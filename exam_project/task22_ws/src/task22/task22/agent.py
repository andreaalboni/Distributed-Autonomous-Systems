import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from time import sleep

# Can be improved - especially the casting part

class Agent(Node):
    def __init__(self):
        super().__init__(
            "formation_control_ros_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters from the launch file
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("neighbors").value
        self.z_i = np.array(self.get_parameter("z0").value)
        self.max_iters = self.get_parameter("max_iters").value
        self.alpah = self.get_parameter("alpha").value
        communication_time = self.get_parameter("communication_time").value
        self.delta_T = communication_time / 10       # Discretization step, it may be decoupled from communication time
        self.k = 0

        print(f"I am agent: {self.agent_id:d}")

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