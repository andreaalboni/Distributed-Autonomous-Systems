import rclpy
from rclpy.node import Node
import numpy as np
import re
from das_interfaces.msg import AggregativeTracking as AggTrackMsg
from das_interfaces.msg import Lidar
from .utils import simulate_lidar_scan, calculate_heading_from_movement

class Lidars(Node):
    def __init__(self):
        super().__init__(
            "lidars",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.d = self.get_parameter("d").value
        self.fov_range = self.get_parameter("fov_range").value
        self.fov_vertical = self.get_parameter("fov_vertical").value
        self.fov_horizontal = self.get_parameter("fov_horizontal").value        
        self.agent_positions = {}
        self.agent_headings = {}
        self.agent_prev_positions = {}
        self.discovered_agents = set()
        self.agent_trajectories = {}        
        self.lidar_publishers = {}        
        self.create_timer(1.0, self.discover_agents)
        self.create_timer(0.5, self.compute_and_publish_lidar)

    def discover_agents(self):
        topic_names_and_types = self.get_topic_names_and_types()
        pattern = r'/topic_(\d+)'
        for topic_name, _ in topic_names_and_types:
            match = re.match(pattern, topic_name)
            if match and topic_name not in self.discovered_agents:
                agent_id = int(match.group(1))
                self.get_logger().info(f'Discovered new agent with ID {agent_id}')
                self.create_subscription(AggTrackMsg,topic_name,lambda msg,id=agent_id: self.agent_callback(msg, id),10)
                self.lidar_publishers[agent_id] = self.create_publisher(Lidar,f'/agent_{agent_id}/lidar',10)
                self.agent_positions[agent_id] = None
                self.agent_headings[agent_id] = 0.0 if self.d == 2 else [0.0, 0.0]  # Initialize headings
                self.agent_prev_positions[agent_id] = None
                self.discovered_agents.add(topic_name)

    def agent_callback(self, msg, agent_id):
        new_position = np.array(msg.z)
        new_heading = calculate_heading_from_movement(new_position,self.agent_prev_positions[agent_id],self.d)
        if new_heading is not None:
            self.agent_headings[agent_id] = new_heading
        self.agent_prev_positions[agent_id] = self.agent_positions[agent_id]
        self.agent_positions[agent_id] = new_position

    def compute_and_publish_lidar(self):
        if len(self.agent_positions) < 2:
            return
        ids = sorted(self.agent_positions.keys())
        positions = [self.agent_positions[i] for i in ids]
        headings = np.array([self.agent_headings[i] for i in ids])
        if any(p is None for p in positions):
            return
        agents_array = np.vstack(positions)
        distances, horizontal_angles, vertical_angles = simulate_lidar_scan(
            agents_array,
            headings,
            self.fov_horizontal,
            self.fov_vertical,
            self.fov_range,
            self.d
        )
        for i, agent_id in enumerate(ids):
            detected_ids = []
            detected_dists = []
            detected_h_angles = []
            detected_v_angles = []
            for j, other_id in enumerate(ids):
                if i != j and not np.isnan(distances[i, j]):
                    detected_ids.append(other_id)
                    detected_dists.append(distances[i, j])
                    detected_h_angles.append(horizontal_angles[i, j])
                    detected_v_angles.append(vertical_angles[i, j])
            msg = Lidar()
            msg.id = agent_id
            msg.detected_ids = detected_ids
            msg.distances = detected_dists
            msg.horizontal_angles = detected_h_angles
            msg.vertical_angles = detected_v_angles
            if agent_id in self.lidar_publishers:
                self.lidar_publishers[agent_id].publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = Lidars()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()