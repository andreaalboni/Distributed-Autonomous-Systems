import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from time import sleep

class Agent(Node):
    def __init__(self, id):
        super().__init__('parametric_agent', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.id = id
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.id}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')