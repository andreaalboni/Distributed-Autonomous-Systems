#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import re
from std_msgs.msg import String
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

class URDFManager(Node):
    def __init__(self):
        super().__init__('urdf_manager')
        
        # QoS profile for URDF publication (to ensure Foxglove sees it)
        urdf_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )
        
        # Publish a topic that Foxglove can recognize for the combined URDF
        self.urdf_publisher = self.create_publisher(
            String,
            '/robot_description',
            urdf_qos
        )
        
        # Dictionary to track known agents
        self.discovered_agents = set()
        
        # Timer to check for new agents
        self.create_timer(1.0, self.discover_agents)
        
        # Get the URDF content
        try:
            pkg_share = get_package_share_directory('task22')
            urdf_file = os.path.join(pkg_share, 'urdf', 'simple_quad.urdf')
            with open(urdf_file, 'r') as file:
                self.urdf_content = file.read()
            self.get_logger().info(f'Loaded URDF from {urdf_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {str(e)}')
            self.urdf_content = ""
    
    def discover_agents(self):
        # Pattern for agent namespaces (looking for TF frames)
        # This checks for transform topics which indicate an agent
        topic_names_and_types = self.get_topic_names_and_types()
        
        # Look for topics matching the agent pattern
        # We need to detect each agent's transform publisher
        agent_pattern = r'/tf'
        frame_pattern = r'agent_(\d+)/base_link/'
        
        # First see if we have tf data
        has_tf = False
        for topic_name, types in topic_names_and_types:
            if topic_name == '/tf':
                has_tf = True
                break
        
        if has_tf:
            # Get active TF frames from the /tf_static topic or parameter
            # For simplicity, we'll just check for known agent namespaces
            # In a real implementation, you would listen to /tf and extract frame IDs
            
            # For now, simulate discovery based on the existing nodes
            current_agents = set()
            node_names_and_namespaces = self.get_node_names_and_namespaces()
            
            for node_name, namespace in node_names_and_namespaces:
                match = re.match(r'/agent_(\d+)', namespace)
                if match:
                    agent_id = int(match.group(1))
                    current_agents.add(agent_id)
            
            # Check for new agents
            new_agents = current_agents - self.discovered_agents
            if new_agents:
                self.discovered_agents.update(new_agents)
                self.get_logger().info(f'Discovered new agents: {new_agents}')
                self.update_combined_urdf()
    
    def update_combined_urdf(self):
        # For Foxglove visualization, we can create a combined URDF
        # or configure Foxglove to recognize the individual URDFs
        
        # Method 1: Generate a combined URDF (simplified for clarity)
        # In a real implementation, you might want to create a more sophisticated URDF
        combined_urdf = self.urdf_content
        
        # Publish the URDF on a topic Foxglove will recognize
        msg = String()
        msg.data = combined_urdf
        self.urdf_publisher.publish(msg)
        self.get_logger().info('Published updated URDF')
        
        # Method 2 (alternative): Publish individual URDFs with frame prefixes
        # This would involve creating parameter overrides or topic publications
        # that Foxglove can recognize with the correct frame prefixes
        # Left as an exercise for a more complete implementation

def main(args=None):
    rclpy.init(args=args)
    node = URDFManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()