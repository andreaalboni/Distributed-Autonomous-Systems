#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
import random

class PointVisualizerNode(Node):
    """
    ROS2 node for visualizing 3D points in Foxglove.
    """
    
    def __init__(self):
        super().__init__('point_visualizer_node')
        
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            'visualization_markers',
            10
        )
        
        self.timer = self.create_timer(1.0, self.publish_points)
        self.get_logger().info('Point Visualizer Node started')
        
    def publish_points(self):
        """Generate and publish random 3D points as markers."""
        marker_array = MarkerArray()
        
        # Create 10 random points
        for i in range(10):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "points"
            marker.id = i            
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = random.uniform(-5.0, 5.0)
            marker.pose.position.y = random.uniform(-5.0, 5.0)
            marker.pose.position.z = random.uniform(-5.0, 5.0)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0            
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2            
            marker.color.r = random.uniform(0.0, 1.0)
            marker.color.g = random.uniform(0.0, 1.0)
            marker.color.b = random.uniform(0.0, 1.0)
            marker.color.a = 1.0  # Alpha (transparency)

            marker_array.markers.append(marker)
        
        self.marker_publisher.publish(marker_array)
        self.get_logger().info(f'Published {len(marker_array.markers)} points')

def main(args=None):
    rclpy.init(args=args)
    node = PointVisualizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()