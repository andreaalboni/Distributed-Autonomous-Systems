import rclpy
import numpy as np
from time import sleep
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat

# Can be improved - especially the casting part

class Agent(Node):
    def __init__(self):
        super().__init__(
            "aggregative_tracking_agent",
            allow_undeclared_parameters = True,
            automatically_declare_parameters_from_overrides = True,
        )

        # Get parameters from the launch file
        self.agent_id = self.get_parameter("id").value
        self.alpha = self.get_parameter("alpha").value
        self.gamma = self.get_parameter("gamma_i").value
        self.z_0 = np.array(self.get_parameter("z0").value)
        self.r_0 = np.array(self.get_parameter("r_0").value)
        self.A_i = np.array(self.get_parameter("A_i").value)
        self.neighbors = self.get_parameter("neighbors").value
        self.max_iters = self.get_parameter("max_iters").value
        self.gamma_bar = self.get_parameter("gamma_bar_i").value
        self.intruder = np.array(self.get_parameter("intruder").value)
        communication_time = self.get_parameter("communication_time").value
        self.delta_T = communication_time / 10       # Discretization step, it may be decoupled from communication time

        self.k = 0
        self.z = np.zeros((self.max_iters, len(self.z_0)))
        self.s = np.zeros((self.max_iters, len(self.z_0)))
        self.v = np.zeros((self.max_iters, len(self.z_0)))
        self.cost = np.zeros((self.max_iters))
        
        # Initialization
        #self.z[0] = self.z_0
        #self.s[0] = self.z_0   # phi_i(z_i) = z_i: regular barycenter
        #self.v[0] = self.local_cost_function(self.z_0, self.intruder, self.s[0], self.r_0, self.gamma, self.gamma_bar)

        print(f"I am agent: {self.agent_id:d}")
        
        # Empty dictionary to store the messages received from each neighbor j
        self.received_data = {j: [] for j in self.neighbors}

        # Subscribe to topic_j
        for j in self.neighbors:
            self.create_subscription(MsgFloat, f"/topic_{j}", self.listener_callback, 10)
        
        # Create topic_i
        self.publisher = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)
        
        self.timer = self.create_timer(communication_time, self.timer_callback)
        print(f"Agent {self.agent_id}: setup completed!")

    '''
        # publish the updated message
        msg.data = [float(self.agent_id), float(self.k), *self.x_i]
        self.publisher.publish(msg)
    '''

    def listener_callback(self, msg):
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.received_data[j].append(msg_j)
        return None

    def timer_callback(self):
        msg = MsgFloat()

        if self.k == 0:
            msg.data = [float(self.agent_id), float(self.k), *self.z[0], *self.s[0], *self.v[0]]
            self.publisher.publish(msg)
            self.k += 1
            print(f"Iter: {self.k:3d}")
        else:
            pass


    def local_phi_function(self, agent_i):
        phi_i = agent_i
        grad_phi_i = 1
        return phi_i, grad_phi_i

    def local_cost_function(self, agent_i, intruder_i, sigma, r_0, gamma_i, gamma_bar_i):
        '''
        Cost function a lezione:
        l_i = gamma_i * (norma della distanza agente - intruder)**2 +                       --> garantisce agente vada sull'intruder
                gamma_bar_i * (norma della distanza agente - sigma)**2 +               --> garantisce formazione sia il più possibile compatta
                norma della distanza r_0 - intruder (che io farei che sia il baricentro degli intruder)
        sigma(z) = sum (phi_i(z_i)) / N     --> calcolo sigma
        phi_i(z_i): per calcolo sigma normale: = z_i, per weighted barycenter: = weight_i * z_i
        
        # agents has shape (2,)
        # intruder has shape (2,)
        # sigma has shape (2,)
        
        # grad_1 is the gradient of the cost function with respect to the intruder
        # grad_2 is the gradient of the cost function with respect to the barycenter
        '''
        agent_to_intruder = np.linalg.norm(agent_i - intruder_i)**2  
        agent_to_sigma = np.linalg.norm(agent_i - sigma)**2
        intruder_to_r_0 = np.linalg.norm(intruder_i - r_0)**2
        
        local_cost = gamma_i * agent_to_intruder + gamma_bar_i * agent_to_sigma + intruder_to_r_0
        
        # grad_1 is the gradient of the cost function with respect to the agent
        grad_1 = 2 * (gamma_i * (agent_i - intruder_i) + gamma_bar_i * (agent_i - sigma))
        # grad_2 is the gradient of the cost function with respect to sigma
        grad_2 = - 2 * gamma_bar_i * (agent_i - sigma)
        
        return local_cost, grad_1, grad_2


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
    except SystemExit:  # <--- process the exception
        rclpy.logging.get_logger("Quitting").info("Done")
    finally:
        aggregative_tracking_agent.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()