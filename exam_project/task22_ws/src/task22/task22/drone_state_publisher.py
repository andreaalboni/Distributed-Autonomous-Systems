#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import time
import numpy as np

class DroneStatePublisher(Node):
    """
    ROS2 node for publishing drone state as TF2 transforms.
    """
    
    def __init__(self):
        super().__init__('drone_state_publisher')
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.timer = self.create_timer(0.05, self.publish_drone_state)  # 20hz
        
        # Initialize drone state variables
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0  # Start 1 meter above ground
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        self.start_time = time.time()
        self.get_logger().info('Drone State Publisher started')
        
    def publish_drone_state(self):
        """Publish drone state as TF transforms."""
        # Calculate animated movement
        elapsed_time = time.time() - self.start_time
        
        # Create a circular flight pattern
        radius = 2.0
        angular_speed = 0.5  # rad/s
        
        self.x = radius * math.cos(angular_speed * elapsed_time)
        self.y = radius * math.sin(angular_speed * elapsed_time)
        self.z = 1.0 + 0.5 * math.sin(0.5 * elapsed_time)  # Oscillate up and down
        
        # Add some gentle tilting in the direction of movement
        # Tilt into the turn
        self.roll = 0.1 * math.sin(angular_speed * elapsed_time)
        self.pitch = 0.1 * math.cos(angular_speed * elapsed_time)
        self.yaw = angular_speed * elapsed_time + math.pi/2  # Point in direction of travel
        
        # Set rotation - convert roll, pitch, yaw to quaternion
        cy = math.cos(self.yaw * 0.5)
        sy = math.sin(self.yaw * 0.5)
        cp = math.cos(self.pitch * 0.5)
        sp = math.sin(self.pitch * 0.5)
        cr = math.cos(self.roll * 0.5)
        sr = math.sin(self.roll * 0.5)
        
        # transform message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'        
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = DroneStatePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
