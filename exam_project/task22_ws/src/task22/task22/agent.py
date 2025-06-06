import rclpy
import cvxpy as cp
import numpy as np
from time import sleep
from rclpy.node import Node
from das_interfaces.msg import Lidar
from visualization_msgs.msg import Marker
from das_interfaces.msg import Plotting as PlotMsg
from das_interfaces.msg import AggregativeTracking as AggTrackMsg

class Agent(Node):
    """ROS2 node implementing an aggregative tracking agent.
    This class implements a distributed aggregative tracking algorithm where multiple
    agents coordinate to track an intruder while maintaining formation constraints.
    The agent communicates with neighboring agents and uses lidar data for collision
    avoidance.
    Attributes:
        agent_id (int): Unique identifier for this agent.
        alpha (float): Step size parameter for gradient descent.
        u_max (float): Maximum control input magnitude.
        gamma (float): Weight parameter for intruder tracking cost.
        z_0 (np.ndarray): Initial position of the agent.
        gamma_sc (float): Safety control parameter for CBF.
        r_0 (np.ndarray): Reference position for tracking.
        A_i (np.ndarray): Communication weight matrix row for this agent.
        neighbors (list): List of neighboring agent IDs.
        max_iters (int): Maximum number of iterations to run.
        gamma_bar (float): Weight parameter for formation tracking cost.
        gamma_hat (float): Weight parameter for reference tracking cost.
        real_dyn (bool): Whether to use double integrator dynamics.
        intruder (np.ndarray): Current intruder position.
        safety_control (bool): Whether to enable collision avoidance.
        safety_distance (float): Minimum safe distance between agents.
        tracking_tolerance (float): Tolerance for goal reaching in real dynamics.
    """

    def __init__(self):
        """Initialize the aggregative tracking agent."""
        super().__init__(
            "aggregative_tracking_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Load parameters from ROS parameter server
        self.agent_id = self.get_parameter("id").value
        self.alpha = self.get_parameter("alpha").value
        self.u_max = self.get_parameter("u_max").value
        self.gamma = self.get_parameter("gamma").value
        self.z_0 = np.array(self.get_parameter("z0").value)
        self.gamma_sc = self.get_parameter("gamma_sc").value
        self.r_0 = np.array(self.get_parameter("r_0").value)
        self.A_i = np.array(self.get_parameter("A_i").value)
        self.neighbors = self.get_parameter("neighbors").value
        self.max_iters = self.get_parameter("max_iters").value
        self.gamma_bar = self.get_parameter("gamma_bar").value
        self.gamma_hat = self.get_parameter("gamma_hat").value
        self.real_dyn = self.get_parameter("real_dynamics").value
        self.intruder = np.array(self.get_parameter("intruder").value)
        self.safety_control = self.get_parameter("safety_control").value
        self.safety_distance = self.get_parameter("safety_distance").value
        communication_time = self.get_parameter("communication_time").value
        self.tracking_tolerance = self.get_parameter("tracking_tolerance").value
        
        self.delta_T = communication_time / 10
        self.d = len(self.z_0)
        self.pos = self.z_0.copy()
        self.vel = np.zeros(self.d)

        self.k = 0
        self.z = np.zeros((self.max_iters + 1, self.d))
        self.s = np.zeros((self.max_iters + 1, self.d))
        self.v = np.zeros((self.max_iters + 1, self.d))
        self.u_ref = np.zeros((self.max_iters + 1, self.d))
        self.safe_u = np.zeros((self.max_iters + 1, self.d))
        self.gradient_direction = np.zeros((self.max_iters + 1, self.d))
        self.safe_gradient_direction = np.zeros((self.max_iters + 1, self.d))
        self.cost = np.zeros((self.max_iters))
        self.visible_neighbors = {}

        self.z[0] = self.z_0
        self.s[0] = self.z_0
        _, _, self.v[0] = self.local_cost_function(self.z_0, self.intruder, self.s[0], self.r_0, self.gamma, self.gamma_bar, self.gamma_hat)

        self.received_data = {}
        for j in self.neighbors:
            self.received_data[j] = {}

        for j in self.neighbors:
            self.create_subscription(AggTrackMsg, f"/topic_{j}", self.listener_callback, 10)
            
        self.create_subscription(Lidar,f'/agent_{self.agent_id}/lidar',self.lidar_callback,10)
        
        self.publisher = self.create_publisher(AggTrackMsg, f"/topic_{self.agent_id}", 10)
        self.dynamics_publisher = self.create_publisher(AggTrackMsg, f"/dynamics_topic_{self.agent_id}", 10)
        self.grad_publisher = self.create_publisher(PlotMsg, f"/plotting_{self.agent_id}", 10)

        self.timer = self.create_timer(communication_time, self.timer_callback)
        print(f"Agent {self.agent_id}: setup completed!")

    def listener_callback(self, msg):
        """Callback for receiving messages from neighboring agents."""
        j = int(msg.id)
        k = int(msg.k)
        if j in self.neighbors:
            self.received_data[j][k] = {
                'z': np.array(msg.z),
                's': np.array(msg.s),
                'v': np.array(msg.v)
            }
            
    def lidar_callback(self, msg):
        """Callback for processing lidar sensor data."""
        self.visible_neighbors = {
            id_: (dist, horiz, vert)
            for id_, dist, horiz, vert in zip(msg.detected_ids, msg.distances, msg.horizontal_angles, msg.vertical_angles)
        }

    def timer_callback(self):
        """Main timer callback that orchestrates the aggregative tracking algorithm.
        Handles the synchronization of iterations, processes received messages,
        executes one iteration of the aggregative tracking algorithm, and
        publishes the updated state.
        """
        if self.k == 0:
            self._publish_current_state()
            print(f"Agent {self.agent_id}: Iter {self.k:3d} - Published initial state \n z={self.z[self.k]}, s={self.s[self.k]}")
            self.k += 1
        else:
            if self._check_messages_received(self.k - 1):
                self._process_iteration()
                
                self._publish_current_state()
                print(f"Agent {self.agent_id}: Iter {self.k:3d} \n z={self.z[self.k]}, s={self.s[self.k]}")
                self.k += 1

                if self.k >= self.max_iters:
                    print(f"\nAgent {self.agent_id}: Max iterations reached")
                    rclpy.shutdown()
            else:
                missing = [j for j in self.neighbors if self.k - 1 not in self.received_data[j]]

    def _check_messages_received(self, iteration):
        """Check if messages from all neighbors have been received for a given iteration."""
        for j in self.neighbors:
            if iteration not in self.received_data[j]:
                return False
        return True

    def _publish_current_state(self, z=None, dynamics=False):
        """Publish the current agent state to other agents."""
        msg = AggTrackMsg()
        msg.id = self.agent_id
        msg.k = self.k
        msg.s = self.s[self.k].tolist()
        msg.v = self.v[self.k].tolist()
        if not dynamics: 
            msg.z = self.z[self.k].tolist()
            self.publisher.publish(msg)
        else:
            msg.z = z.tolist()
        self.dynamics_publisher.publish(msg)

    def _process_iteration(self):
        """Process one iteration of the aggregative tracking algorithm."""
        k = self.k

        neighbor_data = {}
        for j in self.neighbors:
            neighbor_data[j] = self.received_data[j][k-1]
        
        self.aggregative_tracking(
            i=self.agent_id,
            A=self.A_i,
            N_i=self.neighbors,
            k=k-1,
            z=self.z,
            v=self.v,
            s=self.s,
            intruder=self.intruder,
            r_0=self.r_0,
            gamma=self.gamma,
            gamma_bar=self.gamma_bar,
            gamma_hat=self.gamma_hat,
            gamma_sc=self.gamma_sc,
            received_info=neighbor_data
        )

    def local_phi_function(self, agent_i):
        """Compute the local phi function and its gradient for aggregative tracking.
        Args:
            agent_i (np.ndarray): Current position of the agent.
        Returns:
            tuple: A tuple containing:
                - phi_i (np.ndarray): Value of the phi function.
                - grad_phi_i (np.ndarray): Gradient of the phi function.
        """
        phi_i = agent_i
        grad_phi_i = np.ones(self.d)
        return phi_i, grad_phi_i

    def local_cost_function(self, agent_i, intruder_i, sigma, r_0, gamma_i, gamma_bar_i, gamma_hat_i):
        """Compute the local cost function and its gradients.
        Args:
            agent_i (np.ndarray): Current position of the agent.
            intruder_i (np.ndarray): Current position of the intruder.
            sigma (np.ndarray): Aggregative variable for formation control.
            r_0 (np.ndarray): Reference position.
            gamma_i (float): Weight for intruder tracking.
            gamma_bar_i (float): Weight for formation tracking.
            gamma_hat_i (float): Weight for reference tracking.
        Returns:
            tuple: A tuple containing:
                - local_cost (float): Value of the local cost function.
                - grad_1 (np.ndarray): Gradient with respect to agent position.
                - grad_2 (np.ndarray): Gradient with respect to sigma.
        """
        agent_to_intruder = np.linalg.norm(agent_i - intruder_i)**2 
        agent_to_sigma = np.linalg.norm(agent_i - sigma)**2
        intruder_to_r_0 = np.linalg.norm(intruder_i - r_0)**2
        local_cost = gamma_i * agent_to_intruder + gamma_bar_i * agent_to_sigma + gamma_hat_i * intruder_to_r_0
        grad_1 = 2 * (gamma_i * (agent_i - intruder_i) + gamma_bar_i * (agent_i - sigma))
        grad_2 = - 2 * gamma_bar_i * (agent_i - sigma)
        return local_cost, grad_1, grad_2
    
    def compute_cbf(self, x_i, x_j, delta):
        """Compute the Control Barrier Function (CBF) for collision avoidance.
        Args:
            x_i (np.ndarray): Position of agent i.
            x_j (np.ndarray): Position of agent j.
            delta (float): Minimum safe distance.
        Returns:
            tuple: A tuple containing:
                - V_s_ij (float): CBF value.
                - grad_x_i (np.ndarray): CBF gradient with respect to x_i.
                - grad_x_j (np.ndarray): CBF gradient with respect to x_j.
        """
        diff = x_i - x_j
        V_s_ij = np.linalg.norm(diff)**2 - delta**2
        grad_x_i = 2 * diff
        grad_x_j = -2 * diff
        return V_s_ij, grad_x_i, grad_x_j
        
    def safe_control(self, u_ref, x_i, neighbor_positions, delta, gamma_sc, u_max):
        """Compute safe control input using Control Barrier Functions.
        Solves a quadratic program to find the control input closest to the
        reference control while satisfying safety constraints.
        Args:
            u_ref (np.ndarray): Reference control input.
            x_i (np.ndarray): Current position of the agent.
            neighbor_positions (list): List of neighbor positions.
            delta (float): Minimum safe distance.
            gamma_sc (float): CBF gain parameter.
            u_max (float): Maximum control input magnitude.
        Returns:
            np.ndarray: Safe control input that satisfies CBF constraints.
        """
        u = cp.Variable(self.d)
        objective = cp.Minimize(cp.sum_squares(u - u_ref))
        constraints = []
        for x_j in neighbor_positions:
            V_s, grad_x_i, _ = self.compute_cbf(x_i, x_j, delta)
            constraints.append(-grad_x_i @ u - 0.5 * gamma_sc * V_s <= 0)
        constraints.append(cp.norm(u, 2) <= u_max)
        prob = cp.Problem(objective, constraints)
        solvers_to_try = [cp.ECOS, cp.SCS, cp.OSQP]
        for solver in solvers_to_try:
            try:
                prob.solve(solver=solver)
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    return u.value
            except:
                continue
        self.get_logger().warn(f"Agent {self.agent_id}: No solver found a feasible solution, returning reference control")
        return u_ref 
    
    def compute_neighbor_positions(self, z, visible_neighbors):
        """Compute neighbor positions from lidar data.
        Args:
            z (np.ndarray): Current agent position.
            visible_neighbors (dict): Dictionary of visible neighbors with distance/angle data.
        Returns:
            list: List of neighbor positions in Cartesian coordinates.
        """
        neighbor_positions = []
        for neighbor_id, (distance, horiz_angle, vert_angle) in visible_neighbors.items():
            x = z[0] + distance * np.cos(vert_angle) * np.cos(horiz_angle)
            y = z[1] + distance * np.cos(vert_angle) * np.sin(horiz_angle)
            if self.d == 3:
                z_pos = z[2] + distance * np.sin(vert_angle)
                neighbor_positions.append(np.array([x, y, z_pos]))
                self.publish_marker(x, y, z_pos)
            else:
                neighbor_positions.append(np.array([x, y]))
                self.publish_marker(x, y, 0.0)
        return neighbor_positions

    def publish_marker(self, x, y, z):
        """Publish a visualization marker for the agent."""
        if not hasattr(self, 'marker_pub'):
            self.marker_pub = self.create_publisher(Marker, f'/agent_{self.agent_id}/marker', 10)
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"agent_{self.agent_id}"
        marker.id = self.agent_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = [1.0, 0.0, 0.0][self.agent_id % 3]  # Cycle through red, green, blue
        marker.color.g = [0.0, 1.0, 0.0][self.agent_id % 3]
        marker.color.b = [0.0, 0.0, 1.0][self.agent_id % 3]
        
    def dynamics(self, z, u_ref, delta_T):
        """Simple integrator dynamics model.
        Args:
            z (np.ndarray): Current position.
            u_ref (np.ndarray): Control input.
            delta_T (float): Time step.
        Returns:
            np.ndarray: Updated position.
        """
        return z + delta_T * u_ref 
    
    def real_dynamics(self, pos, vel, acc, delta_T):
        """Double integrator dynamics model.
        Args:
            pos (np.ndarray): Current position.
            vel (np.ndarray): Current velocity.
            acc (np.ndarray): Acceleration input.
            delta_T (float): Time step.
        Returns:
            tuple: A tuple containing:
                - pos_next (np.ndarray): Updated position.
                - vel_next (np.ndarray): Updated velocity.
        """
        pos_next = pos + delta_T * vel
        vel_next = vel + delta_T * acc
        return pos_next, vel_next
    
    def pid_position_controller(self, desired_pos, current_pos, current_vel, integral_error, Kp=1.0, Ki=0.01, Kd=0.5, max_integral=0.5):
        """PID controller for position tracking.
        Args:
            desired_pos (np.ndarray): Desired position.
            current_pos (np.ndarray): Current position.
            current_vel (np.ndarray): Current velocity.
            integral_error (np.ndarray): Accumulated integral error.
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            max_integral (float): Maximum integral error magnitude.
        Returns:
            tuple: A tuple containing:
                - acc_command (np.ndarray): Acceleration command.
                - integral_error (np.ndarray): Updated integral error.
        """
        error = desired_pos - current_pos
        derivative = -current_vel
        integral_error += error * self.delta_T
        integral_error = np.clip(integral_error, -max_integral, max_integral)
        acc_command = Kp * error + Ki * integral_error + Kd * derivative
        return acc_command, integral_error
    
    def has_reached_goal(self, pos, goal):
        """Check if the agent has reached the goal position."""
        return np.linalg.norm(pos - goal) < self.tracking_tolerance
    
    def publish_plotting_data(self, l_i, grad_1_l, grad_phi, grad_2_l_i):
        """Publish data for plotting and analysis.
        Args:
            l_i (float): Local cost function value.
            grad_1_l (np.ndarray): Gradient of cost w.r.t. agent position.
            grad_phi (np.ndarray): Gradient of phi function.
            grad_2_l_i (np.ndarray): Gradient of cost w.r.t. sigma.
        """
        plot_msg = PlotMsg()
        plot_msg.id = self.agent_id
        plot_msg.k = self.k
        plot_msg.cost = float(l_i) 
        plot_msg.grad_1 = grad_1_l.tolist()   
        plot_msg.grad_phi = grad_phi.tolist()
        plot_msg.grad_2 = grad_2_l_i.tolist()
        self.grad_publisher.publish(plot_msg)
  
    def aggregative_tracking(self, i, A, N_i, k, z, v, s, intruder, r_0, gamma, gamma_bar, gamma_hat, gamma_sc, received_info):
        """Main aggregative tracking algorithm implementation. Executes one
        iteration of the distributed aggregative tracking algorithm.
        Updates agent position, aggregative variable (s), and dual variable (v)
        based on local cost gradients and neighbor information.
        Args:
            i (int): Agent ID.
            A (np.ndarray): Communication weight matrix row.
            N_i (list): List of neighbor agent IDs.
            k (int): Current iteration number.
            z (np.ndarray): Position trajectory array.
            v (np.ndarray): Dual variable trajectory array.
            s (np.ndarray): Aggregative variable trajectory array.
            intruder (np.ndarray): Intruder position.
            r_0 (np.ndarray): Reference position.
            gamma (float): Intruder tracking weight.
            gamma_bar (float): Formation tracking weight.
            gamma_hat (float): Reference tracking weight.
            gamma_sc (float): Safety control parameter.
            received_info (dict): Information received from neighbors.
        """
        _, grad_1_l_i, _ = self.local_cost_function(z[k], intruder, s[k], r_0, gamma, gamma_bar, gamma_hat)
        _, grad_phi_i = self.local_phi_function(z[k])

        if not self.real_dyn:
            # --------------- Position Control ---------------
            self.u_ref[k] = - self.alpha * (grad_1_l_i + grad_phi_i * v[k])
            neighbor_positions = self.compute_neighbor_positions(z[k], self.visible_neighbors)
            if self.safety_control:
                self.safe_u[k] = self.safe_control(self.u_ref[k], z[k], neighbor_positions, self.safety_distance, gamma_sc, self.u_max)
            else:
                self.safe_u[k] = self.u_ref[k]
            z[k+1] = self.dynamics(z[k], self.safe_u[k], self.delta_T)
        else:
            # --------------- Double Integrator Dynamics + PID low lever controller ----------------
            self.gradient_direction[k] = - self.alpha * (grad_1_l_i + grad_phi_i * v[k])
            neighbor_positions = self.compute_neighbor_positions(z[k], self.visible_neighbors)
            if self.safety_control:
                self.safe_gradient_direction[k] = self.safe_control(self.gradient_direction[k], z[k], neighbor_positions, self.safety_distance, gamma_sc, self.u_max)
            else:
                self.safe_gradient_direction[k] = self.gradient_direction[k]
            z[k+1] = z[k] + self.delta_T * self.safe_gradient_direction[k]
            integral_error = 0
            while not self.has_reached_goal(self.pos, z[k+1]):
                acc_cmd, integral_error = self.pid_position_controller(z[k+1], self.pos, self.vel, integral_error)
                self.pos, self.vel = self.real_dynamics(self.pos, self.vel, acc_cmd, self.delta_T)
                self._publish_current_state(self.pos, True)

        s[k+1] = A[i] * s[k]
        for j in N_i:
            s_k_j = received_info[j]['s']
            s[k+1] += A[j] * s_k_j
        s[k+1] += z[k+1] - z[k]
        
        v[k+1] = A[i] * v[k]
        for j in N_i:
            v_k_j = received_info[j]['v']
            v[k+1] += A[j] * v_k_j
            
        _, _, grad_2_l_i_new = self.local_cost_function(z[k+1], intruder, s[k+1], r_0, gamma, gamma_bar, gamma_hat)
        l_i, _, grad_2_l_i_old = self.local_cost_function(z[k], intruder, s[k], r_0, gamma, gamma_bar, gamma_hat)
        v[k+1] += grad_2_l_i_new - grad_2_l_i_old

        self.publish_plotting_data(l_i, grad_1_l_i, grad_phi_i, grad_2_l_i_old)    

def main(args=None):
    rclpy.init(args=args)
    aggregative_tracking_agent = Agent()
    aggregative_tracking_agent.get_logger().info(
        f"Agent {aggregative_tracking_agent.agent_id:d}: Waiting for sync..."
    )
    sleep(1)
    aggregative_tracking_agent.get_logger().info("GO!")
    try:
        rclpy.spin(aggregative_tracking_agent)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Done")
    finally:
        aggregative_tracking_agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
