#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from das_interfaces.msg import AggregativeTracking as AggTrackMsg
import re
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer')
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_publisher = self.create_publisher(
            MarkerArray, 
            'agent_trajectories', 
            10
        )
        self.agent_states = {}
        self.agent_trajectories = {}
        self.discovered_agents = set()
        
        # QoS for subscribers (to ensure we get messages even if published before we subscribe)
        qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )
        self.create_timer(1.0, self.discover_agents)
        self.create_timer(0.05, self.publish_visualizations)
    
    def discover_agents(self):
        topic_names_and_types = self.get_topic_names_and_types()
        
        pattern = r'/topic_(\d+)'
        
        for topic_name, _ in topic_names_and_types:
            match = re.match(pattern, topic_name)
            if match and topic_name not in self.discovered_agents:
                agent_id = int(match.group(1))
                self.get_logger().info(f'Discovered new agent with ID {agent_id}')
                
                # Create a subscription for this agent
                self.create_subscription(
                    AggTrackMsg,
                    topic_name,
                    lambda msg, id=agent_id: self.agent_callback(msg, id),
                    10
                )
                
                self.agent_trajectories[agent_id] = []                
                self.discovered_agents.add(topic_name)
    
    def agent_callback(self, msg, agent_id):
        self.agent_states[agent_id] = {
            'z': msg.z,
            'k': msg.k
        }
        
        # Update trajectory
        if len(msg.z) >= 2:
            position = [0.0, 0.0, 0.0]
            for i in range(len(msg.z)):
                position[i] = msg.z[i]                
            self.agent_trajectories[agent_id].append(position)
    
    def publish_visualizations(self):
        current_time = self.get_clock().now().to_msg()
        
        # Publish TF transforms
        for agent_id, state in self.agent_states.items():
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = 'world'
            t.child_frame_id = f'agent_{agent_id}/base_link'
            if len(state['z']) >= 2:
                t.transform.translation.x = state['z'][0]
                t.transform.translation.y = state['z'][1]
                if len(state['z']) >= 3:
                    t.transform.translation.z = state['z'][2]
                else:
                    t.transform.translation.z = 0.0
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0            
            self.tf_broadcaster.sendTransform(t)
        
        # Publish trajectory markers
        marker_array = MarkerArray()
        for agent_id, trajectory in self.agent_trajectories.items():
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = current_time
            marker.ns = 'agent_trajectories'
            marker.id = agent_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # Line width            
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 1.0
            
            for point in trajectory:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                marker.points.append(p)
            
            marker_array.markers.append(marker)
        
        # Publish all markers
        self.marker_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = Visualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()