import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("minimal_publisher")
        self.publisher = self.create_publisher(String, "aTopic", 10)
        time_period = 0.5
        self.timer = self.create_timer(
            timer_period_sec=time_period, callback=self.timer_callback
        )
        self.iter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f"Hello world {self.iter:d}"
        self.iter += 1
        self.publisher.publish(msg)
        self.get_logger().info(f"Publishing: {msg.data}")


def main():
    rclpy.init()

    aMinPubl = MinimalPublisher()

    rclpy.spin(aMinPubl)
    aMinPubl.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
