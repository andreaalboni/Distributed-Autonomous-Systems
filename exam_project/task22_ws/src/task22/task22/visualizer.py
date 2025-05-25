#!/usr/bin/env python3
import re
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from das_interfaces.msg import AggregativeTracking as AggTrackMsg
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

class Visualizer(Node):
    def __init__(self):
        super().__init__(
            "visualizer",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )
                
        self.d = self.get_parameter("d").value
        self.r_0 = np.array(self.get_parameter("r_0").value)
        self.fov_range = self.get_parameter("fov_range").value
        self.world_size = self.get_parameter("world_size").value
        self.fov_vertical = self.get_parameter("fov_vertical").value
        self.fov_horizontal = self.get_parameter("fov_horizontal").value

        self.prev_heading = {}
        
        intruders = np.array(self.get_parameter("intruders").value)
        initial_agent_positions = np.array(self.get_parameter("initial_agent_positions").value)
        self.intruders = [intruders[i:i+self.d] for i in range(0, len(intruders), self.d)]
        self.initial_agent_positions = [initial_agent_positions[i:i+self.d] for i in range(0, len(initial_agent_positions), self.d)]
        
        self.trajectories_publisher = self.create_publisher(MarkerArray, 'trajectories', self.qos)
        self.intruders_publisher = self.create_publisher(MarkerArray, 'intruders', self.qos)
        self.marker_publisher = self.create_publisher(Marker, 'r_0', self.qos)
        self.grid_publisher = self.create_publisher(Marker, 'world_grid', self.qos)
        
        self.agent_states = {}
        self.agent_trajectories = {}
        self.discovered_agents = set()
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0, self.discover_agents)
        self.create_timer(0.05, self.publish_visualizations)
        
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_grid_tf()
        self.publish_world_grid()
        self.publish_r_0()
        self.publish_intruders()
    

    def create_fov_marker(self, agent_id, position, heading):
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'fov'
        marker.id = agent_id + 1000  # Unique ID for FOV markers
        marker.action = Marker.ADD

        # For 2D case
        if self.d == 2:
            marker.type = Marker.TRIANGLE_LIST
            z = 0.0
            segments = 32  # Number of triangles to approximate the sector
            
            # Convert FOV angle from degrees to radians
            angle_rad = np.radians(float(self.fov_horizontal))
            # Calculate the starting angle (centered around heading)
            start_angle = heading - angle_rad/2
            
            # Create vertices for triangle fan
            center = Point(x=position[0], y=position[1], z=z)
            
            for i in range(segments):
                theta1 = start_angle + (angle_rad * i / segments)
                theta2 = start_angle + (angle_rad * (i + 1) / segments)
                
                # First point of triangle (center)
                marker.points.append(center)
                
                # Second point of triangle (first radius)
                marker.points.append(Point(
                    x=position[0] + self.fov_range * np.cos(theta1),
                    y=position[1] + self.fov_range * np.sin(theta1),
                    z=z
                ))
                
                # Third point of triangle (second radius)
                marker.points.append(Point(
                    x=position[0] + self.fov_range * np.cos(theta2),
                    y=position[1] + self.fov_range * np.sin(theta2),
                    z=z
                ))
            
        # For 3D case 
        else:
            marker.type = Marker.TRIANGLE_LIST
            z = position[2]
            segments = 32  # Number of segments for the circular approximation
            v_angle_rad = np.radians(float(self.fov_vertical))
            h_angle_rad = np.radians(float(self.fov_horizontal))
            for i in range(segments):
                theta1 = heading - h_angle_rad/2 + (h_angle_rad * i / segments)
                theta2 = heading - h_angle_rad/2 + (h_angle_rad * (i + 1) / segments)
                for v_angle in [-v_angle_rad/2, v_angle_rad/2]:
                    p0 = Point(x=position[0], y=position[1], z=z)  # Cone apex
                    p1 = Point(
                        x=position[0] + self.fov_range * np.cos(theta1) * np.cos(v_angle),
                        y=position[1] + self.fov_range * np.sin(theta1) * np.cos(v_angle),
                        z=z + self.fov_range * np.sin(v_angle)
                    )
                    p2 = Point(
                        x=position[0] + self.fov_range * np.cos(theta2) * np.cos(v_angle),
                        y=position[1] + self.fov_range * np.sin(theta2) * np.cos(v_angle),
                        z=z + self.fov_range * np.sin(v_angle)
                    )
                    marker.points.extend([p0, p1, p2])
                if i % (segments/4) == 0:  # Create fewer vertical planes for cleaner visualization
                    p_top = Point(
                        x=position[0] + self.fov_range * np.cos(theta1) * np.cos(v_angle_rad/2),
                        y=position[1] + self.fov_range * np.sin(theta1) * np.cos(v_angle_rad/2),
                        z=z + self.fov_range * np.sin(v_angle_rad/2)
                    )
                    p_bottom = Point(
                        x=position[0] + self.fov_range * np.cos(theta1) * np.cos(-v_angle_rad/2),
                        y=position[1] + self.fov_range * np.sin(theta1) * np.cos(-v_angle_rad/2),
                        z=z + self.fov_range * np.sin(-v_angle_rad/2)
                    )
                    marker.points.extend([p0, p_top, p_bottom])

        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Set color (semi-transparent)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.3

        return marker

    def discover_agents(self):
        topic_names_and_types = self.get_topic_names_and_types()
        pattern = r'/dynamics_topic_(\d+)'
        for topic_name, _ in topic_names_and_types:
            match = re.match(pattern, topic_name)
            if match and topic_name not in self.discovered_agents:
                agent_id = int(match.group(1))
                self.get_logger().info(f'Discovered new agent with ID {agent_id}')
                self.get_logger().info(f'Subscribing to {topic_name}')
                self.create_subscription(
                    AggTrackMsg,
                    topic_name,
                    lambda msg, id=agent_id: self.agent_callback(msg, id),
                    10
                )
                self.agent_trajectories[agent_id] = []
                self.discovered_agents.add(topic_name)

    def agent_callback(self, msg, agent_id):
        self.agent_states[agent_id] = {'z': msg.z, 'k': msg.k}
        pos = [0.0, 0.0, 0.0]
        for i in range(len(msg.z)):
            pos[i] = msg.z[i]
        self.agent_trajectories[agent_id].append(pos)
    
    def publish_static_grid_tf(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id  = "world"   # parent
        tf.child_frame_id   = "grid"    # child
        tf.transform.translation.x = float(self.r_0[0])
        tf.transform.translation.y = float(self.r_0[1])
        tf.transform.translation.z = min(np.array(self.intruders)[:,2]) - 0.5 if len(self.r_0) > 2 else 0.0
        tf.transform.rotation.w = 1.0   # identity quaternion
        self.static_broadcaster.sendTransform(tf)

    def publish_world_grid(self):
        size = float(self.world_size) * 1.25    
        step = 1.0                             
        half = size / 2.0
        grid = Marker()
        grid.header.frame_id = "grid"          # the frame we placed at r_0
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.ns = "world_grid"
        grid.id = 0
        grid.type = Marker.LINE_LIST
        grid.action = Marker.ADD
        grid.scale.x = 0.03                    # line thickness
        grid.color.r = 0.6
        grid.color.g = 0.6
        grid.color.b = 0.6
        grid.color.a = 1.0
        x = -half
        while x <= half:
            p1 = Point(); p1.x = x;  p1.y = -half; p1.z = 0.0
            p2 = Point(); p2.x = x;  p2.y =  half; p2.z = 0.0
            grid.points.extend([p1, p2])
            x += step
        y = -half
        while y <= half:
            p1 = Point(); p1.x = -half; p1.y = y; p1.z = 0.0
            p2 = Point(); p2.x =  half; p2.y = y; p2.z = 0.0
            grid.points.extend([p1, p2])
            y += step
        grid.pose.orientation.w = 1.0
        self.grid_publisher.publish(grid)

    def publish_r_0(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "r_0"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.r_0[0])
        marker.pose.position.y = float(self.r_0[1])
        marker.pose.position.z = float(self.r_0[2]) if len(self.r_0) > 2 else 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_publisher.publish(marker)
        
    def publish_intruders(self):
        marker_array = MarkerArray()
        for i in range(len(self.intruders)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "points"
            marker.id = i            
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.intruders[i][0]
            marker.pose.position.y = self.intruders[i][1]
            marker.pose.position.z = float(self.intruders[i][2]) if len(self.intruders[i]) > 2 else 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0            
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0  
            marker_array.markers.append(marker)
        self.intruders_publisher.publish(marker_array)
      
    def create_drone_marker(self, frame_id, marker_id, position, stamp):
        markers = []
        body = Marker()
        body.header.frame_id = frame_id
        body.header.stamp = stamp
        body.ns = "drone_body"
        body.id = marker_id
        body.type = Marker.CYLINDER
        body.action = Marker.ADD
        body.pose.position.x = position[0]
        body.pose.position.y = position[1]
        z = position[2] if len(position) >= 3 else 0.0
        body.pose.position.z = z
        body.pose.orientation.w = 1.0
        body.scale.x = 0.4  # diameter
        body.scale.y = 0.4  # diameter
        body.scale.z = 0.1  # height
        body.color.r = 0.7
        body.color.g = 0.7
        body.color.b = 0.7
        body.color.a = 1.0
        markers.append(body)
        arm_length = 0.9
        arm_width = 0.075
        arm_height = 0.075
        z_offset = 0.05  # Offset above the body
        for i in range(4):
            arm = Marker()
            arm.header.frame_id = frame_id
            arm.header.stamp = stamp
            arm.ns = "drone_arms"
            arm.id = marker_id * 10 + i + 1  # Unique ID for each arm
            arm.type = Marker.CYLINDER
            arm.action = Marker.ADD
            arm.pose.position.x = position[0]
            arm.pose.position.y = position[1]
            arm.pose.position.z = z + z_offset
            angle = np.pi/4 + (np.pi/2 * i)
            arm.pose.orientation.w = np.cos(angle/2)
            arm.pose.orientation.z = np.sin(angle/2)
            arm.scale.x = arm_length  # length
            arm.scale.y = arm_width   # width
            arm.scale.z = arm_height  # height
            arm.color.r = 0.3
            arm.color.g = 0.3
            arm.color.b = 0.3
            arm.color.a = 1.0
            markers.append(arm)
        return markers

    def publish_visualizations(self):
        current_time = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        
        # Publish initial agent positions until agents are discovered
        if not self.agent_states:
            for i, pos in enumerate(self.initial_agent_positions):
                drone_markers = self.create_drone_marker('world', i, pos, current_time)
                marker_array.markers.extend(drone_markers)

        active_positions = []
        for agent_id, state in self.agent_states.items():
            if len(state['z']) >= 2:
                position = [
                    state['z'][0],
                    state['z'][1],
                    state['z'][2] if len(state['z']) >= 3 else 0.0
                ]
                active_positions.append(position)
                # Calculate motion direction
                heading = 0.0
                idx = self.agent_trajectories[agent_id].index(position)
                curr_pos = position
                if idx > 0:
                    prev_pos = self.agent_trajectories[agent_id][idx-1]
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    if abs(dx) > 0.001 or abs(dy) > 0.001:
                        heading = np.arctan2(dy, dx)
                        self.prev_heading[agent_id] = heading
                    elif agent_id in self.prev_heading:
                        # Use the last valid heading if agent is not moving
                        heading = self.prev_heading[agent_id]
                drone_markers = self.create_drone_marker(
                    'world',
                    agent_id,
                    position,
                    current_time
                )
                marker_array.markers.extend(drone_markers)
                # Add FOV marker for this agent with heading
                fov_marker = self.create_fov_marker(agent_id, position, heading)
                marker_array.markers.append(fov_marker)

                # Add agent trajectories
                if agent_id in self.agent_trajectories:
                    traj_marker = Marker()
                    traj_marker.header.frame_id = 'world'
                    traj_marker.header.stamp = current_time
                    traj_marker.ns = 'agent_trajectories'
                    traj_marker.id = agent_id
                    
                    if len(self.agent_trajectories[agent_id]) > 1:
                        traj_marker.type = Marker.LINE_STRIP
                    else:
                        traj_marker.type = Marker.POINTS
                    traj_marker.action = Marker.ADD
                    traj_marker.scale.x = 0.05  # Line width            
                    traj_marker.color.r = 0.5
                    traj_marker.color.g = 0.5
                    traj_marker.color.b = 0.5
                    traj_marker.color.a = 1.0
                    for point in self.agent_trajectories[agent_id]:
                        p = Point()
                        p.x = point[0]
                        p.y = point[1]
                        p.z = point[2]
                        traj_marker.points.append(p)
                    marker_array.markers.append(traj_marker)

        all_agents_present = len(self.agent_states) == len(self.agent_trajectories)
        if all_agents_present and active_positions:
            barycentre = np.mean(active_positions, axis=0)
            
            # Store barycentre trajectory
            if not hasattr(self, 'barycentre_trajectory'):
                self.barycentre_trajectory = []
            self.barycentre_trajectory.append(barycentre)

            # Barycentre marker
            bary_marker = Marker()
            bary_marker.header.frame_id = 'world'
            bary_marker.header.stamp = current_time
            bary_marker.ns = 'barycentre'
            bary_marker.id = 999  # Unique ID for barycentre
            bary_marker.type = Marker.TRIANGLE_LIST
            # Create hexagon vertices
            r = 0.5
            vertices = []
            for i in range(6):
                angle = i * np.pi / 3
                vertices.extend([
                    Point(x=0.0, y=0.0, z=0.0),  # center
                    Point(x=r*np.cos(angle), y=r*np.sin(angle), z=0.0),
                    Point(x=r*np.cos(angle + np.pi/3), y=r*np.sin(angle + np.pi/3), z=0.0)
                ])
            bary_marker.points = vertices
            bary_marker.action = Marker.ADD
            bary_marker.pose.position.x = barycentre[0]
            bary_marker.pose.position.y = barycentre[1]
            bary_marker.pose.position.z = barycentre[2]
            bary_marker.pose.orientation.w = 1.0
            bary_marker.scale.x = 0.4
            bary_marker.scale.y = 0.4
            bary_marker.scale.z = 0.4
            bary_marker.color.g = 1.0  # Green
            bary_marker.color.a = 0.75
            marker_array.markers.append(bary_marker)

            # Barycentre trajectory
            bary_traj = Marker()
            bary_traj.header.frame_id = 'world'
            bary_traj.header.stamp = current_time
            bary_traj.ns = 'barycentre_trajectory'
            bary_traj.id = 1000  # Unique ID for barycentre trajectory
            if len(self.barycentre_trajectory) > 1:
                bary_traj.type = Marker.LINE_STRIP
            else:
                bary_traj.type = Marker.POINTS
            bary_traj.action = Marker.ADD
            bary_traj.scale.x = 0.05
            bary_traj.color.g = 0.7  # Light green
            bary_traj.color.a = 0.7
            for point in self.barycentre_trajectory:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                bary_traj.points.append(p)
            marker_array.markers.append(bary_traj)

        self.trajectories_publisher.publish(marker_array)
        current_time = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

def main(args=None):
    rclpy.init(args=args)
    node = Visualizer()
    
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