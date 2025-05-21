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
        super().__init__(
            "visualizer",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
                
        intruders = np.array(self.get_parameter("intruders").value)
        self.r_0 = np.array(self.get_parameter("r_0").value)
        self.world_size = self.get_parameter("world_size").value
        
        self.d = len(self.r_0)
        self.intruders = [intruders[i:i+self.d] for i in range(0, len(intruders), self.d)]
        
        self.get_logger().info(f'\033[92mintruders: {self.intruders}\033[0m')
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.agent_trajectories_publisher = self.create_publisher(
            MarkerArray, 
            'agent_trajectories', 
            10
        )
        self.intruder_publisher = self.create_publisher(
            MarkerArray, 
            'intruders', 
            10
        )
        self.marker_publisher = self.create_publisher(
            Marker, 
            'r_0', 
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
        # self.publish_r_0(self.r_0)
        self.publish_intruders(self.intruders)
    
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
    
    def publish_r_0(self, r_0):
        """Publish r_0 as a single marker."""
        
        position = [0.0, 0.0, 0.0]
        
        for i in range(len(r_0)):
            position[i]=r_0[i]
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "r_0"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_publisher.publish(marker)
        self.get_logger().info('Published r_0 marker')
        
    def publish_intruders(self, intruders):
        """Generate and publish random 3D points as markers."""
        marker_array = MarkerArray()
            
        for i in range(len(intruders)):
            
            
            position = [0.0, 0.0, 0.0]
            
            self.get_logger().info(f"intruder: {intruders[i]}")
            
            for j in range(len(intruders[i])):
                position[j] = intruders[i][j]
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "points"
            marker.id = i            
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0            
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2            
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0  # Alpha (transparency)

            marker_array.markers.append(marker)
        
        self.intruder_publisher.publish(marker_array)
        self.get_logger().info(f'Published {len(marker_array.markers)} points')

        
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
        self.marker_array_publisher.publish(marker_array)

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