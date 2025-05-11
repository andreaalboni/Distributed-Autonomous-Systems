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
        self.alpah = self.get_parameter("alpha").value
        self.z_i = np.array(self.get_parameter("z0").value)
        self.A_i = np.array(self.get_parameter("A_i").value)
        self.neighbors = self.get_parameter("neighbors").value
        self.max_iters = self.get_parameter("max_iters").value
        self.intruder = np.array(self.get_parameter("intruder").value)
        communication_time = self.get_parameter("communication_time").value
        self.delta_T = communication_time / 10       # Discretization step, it may be decoupled from communication time
        
        self.k = 0

        print(f"I am agent: {self.agent_id:d}")
        print(f"alpha: {self.alpah}")
        print(f"Initial position z0: {self.z_i}")
        print(f"Adjacency matrix row A_i: {self.A_i}")
        print(f"Neighbors: {self.neighbors}")
        print(f"Max iterations: {self.max_iters}")
        print(f"Intruder position: {self.intruder}")
        print(f"Communication time: {communication_time}")
        print(f"Time step delta_T: {self.delta_T}")

        # Subscribe to topic_j
        for j in self.neighbors:
            self.create_subscription(MsgFloat, f"/topic_{j}", self.listener_callback, 10)
        
        # Empty dictionary to store the messages received from each neighbor j
        self.received_data = {j: [] for j in self.neighbors}

        # Create topic_i
        self.publisher = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)
        
        self.timer = self.create_timer(communication_time, self.timer_callback)
        print(f"Agent {self.agent_id}: setup completed!")


    def listener_callback(self, msg):
        pass

    def timer_callback(self):
        pass


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