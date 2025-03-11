import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray as MsgFloat
from time import sleep


class Agent(Node):
    def __init__(self):
        super().__init__(
            "parametric_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("neighbors").value
        self.inital_value = self.get_parameter("xzero").value

        self.get_logger().info(f"I am agent: {self.agent_id:d}")
        self.get_logger().info(f"My neighbors are: {self.neighbors:d}")
        self.get_logger().info(f"My initial value is: {self.inital_value:d}")

        # Create a publisher
        self.publisher_ = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)
        self.timer_ = self.create_timer(1.0, self.timer_callback) # Check timer to see if we can publish

        # Create a subscriber
        for j in self.neighbors:
            self.create_subscription(MsgFloat, f"/topic_{j}", self.listener_callback, 10)
        self.received_data = {j: [] for j in self.neighbors}
        
    def listener_callback(self, msg):
        pass

    def timer_callback(self):
        pass

def main():
    rclpy.init()

    anAgent = Agent()
    sleep(1)
    anAgent.get_logger("Starting").info("GO!")

    try:
        rclpy.spin(anAgent)
    except SystemExit:  # <--- process the exception
        rclpy.logging.get_logger("Quitting").info("Done")

    anAgent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
