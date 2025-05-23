import rclpy
import numpy as np
from time import sleep
from rclpy.node import Node
from das_interfaces.msg import AggregativeTracking as AggTrackMsg

class Agent(Node):
    def __init__(self):
        super().__init__(
            "aggregative_tracking_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.agent_id = self.get_parameter("id").value
        self.alpha = self.get_parameter("alpha").value
        self.gamma = self.get_parameter("gamma").value
        self.z_0 = np.array(self.get_parameter("z0").value)
        self.r_0 = np.array(self.get_parameter("r_0").value)
        self.A_i = np.array(self.get_parameter("A_i").value)
        self.neighbors = self.get_parameter("neighbors").value
        self.max_iters = self.get_parameter("max_iters").value
        self.gamma_bar = self.get_parameter("gamma_bar").value
        self.gamma_hat = self.get_parameter("gamma_hat").value
        self.intruder = np.array(self.get_parameter("intruder").value)
        communication_time = self.get_parameter("communication_time").value
        self.delta_T = communication_time / 10

        self.k = 0
        self.z = np.zeros((self.max_iters + 1, len(self.z_0)))
        self.s = np.zeros((self.max_iters + 1, len(self.z_0)))
        self.v = np.zeros((self.max_iters + 1, len(self.z_0)))
        self.cost = np.zeros((self.max_iters))

        self.z[0] = self.z_0
        self.s[0] = self.z_0
        _, _, self.v[0] = self.local_cost_function(self.z_0, self.intruder, self.s[0], self.r_0, self.gamma, self.gamma_bar, self.gamma_hat)

        self.received_data = {}
        for j in self.neighbors:
            self.received_data[j] = {}

        for j in self.neighbors:
            self.create_subscription(AggTrackMsg, f"/topic_{j}", self.listener_callback, 10)
        
        self.publisher = self.create_publisher(AggTrackMsg, f"/topic_{self.agent_id}", 10)
        
        self.timer = self.create_timer(communication_time, self.timer_callback)
        print(f"Agent {self.agent_id}: setup completed!")

    def listener_callback(self, msg):
        j = int(msg.id)
        k = int(msg.k)
        if j in self.neighbors:
            self.received_data[j][k] = {
                'z': np.array(msg.z),
                's': np.array(msg.s),
                'v': np.array(msg.v)
            }
            #print(f"Agent {self.agent_id}: Received message from agent {j} for iteration {k}")

    def timer_callback(self):
        if self.k == 0:
            self._publish_current_state()
            print(f"Agent {self.agent_id}: Iter {self.k:3d} - Published initial state")
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
                #print(f"Agent {self.agent_id}: Waiting for iter {self.k-1} from agents {missing}")

    def _check_messages_received(self, iteration):
        for j in self.neighbors:
            if iteration not in self.received_data[j]:
                return False
        return True

    def _publish_current_state(self):
        msg = AggTrackMsg()
        msg.id = self.agent_id
        msg.k = self.k
        msg.z = self.z[self.k].tolist()
        msg.s = self.s[self.k].tolist()
        msg.v = self.v[self.k].tolist()
        self.publisher.publish(msg)

    def _process_iteration(self):
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
            received_info=neighbor_data
        )

    def local_phi_function(self, agent_i):
        phi_i = agent_i
        grad_phi_i = 1
        return phi_i, grad_phi_i

    def local_cost_function(self, agent_i, intruder_i, sigma, r_0, gamma_i, gamma_bar_i, gamma_hat_i):
        agent_to_intruder = np.linalg.norm(agent_i - intruder_i)**2 
        agent_to_sigma = np.linalg.norm(agent_i - sigma)**2
        intruder_to_r_0 = np.linalg.norm(intruder_i - r_0)**2
        local_cost = gamma_i * agent_to_intruder + gamma_bar_i * agent_to_sigma + gamma_hat_i * intruder_to_r_0
        grad_1 = 2 * (gamma_i * (agent_i - intruder_i) + gamma_bar_i * (agent_i - sigma))
        grad_2 = - 2 * gamma_bar_i * (agent_i - sigma)
        return local_cost, grad_1, grad_2
    
    def aggregative_tracking(self, i, A, N_i, k, z, v, s, intruder, r_0, gamma, gamma_bar, gamma_hat, received_info):
        _, grad_1_l_i, _ = self.local_cost_function(z[k], intruder, s[k], r_0, gamma, gamma_bar, gamma_hat)
        _, grad_phi_i = self.local_phi_function(z[k])
        
        z[k+1] = z[k] - self.alpha * (grad_1_l_i + grad_phi_i * v[k])
    
        s[k+1] = A[i] * s[k]
        for j in N_i:
            s_k_j = received_info[j]['s']
            s[k+1] += A[j] * s_k_j
        s[k+1] += z[k+1] - z[k]
        
        v[k+1] = A[i] * v[k]
        for j in N_i:
            v_k_j = received_info[j]['v']
            v[k+1] += A[i] * v_k_j
            
        _, _, grad_2_l_i_new = self.local_cost_function(z[k+1], intruder, s[k+1], r_0, gamma, gamma_bar, gamma_hat)
        _, _, grad_2_l_i_old = self.local_cost_function(z[k], intruder, s[k], r_0, gamma, gamma_bar, gamma_hat)
        v[k+1] += grad_2_l_i_new - grad_2_l_i_old

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
