import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.subscription = self.create_subscription(
            String, "aTopic", self.listener_callback, 10
        )

    def listener_callback(self, msg):
        self.get_logger().info(f"I heard: {msg.data}")


def main():
    rclpy.init()

    aMinSubs = MinimalSubscriber()

    rclpy.spin(aMinSubs)
    aMinSubs.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
