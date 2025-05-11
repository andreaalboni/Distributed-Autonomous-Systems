from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
import os

def generate_launch_description():
    pkg_share = FindPackageShare(package='task22').find('task22')    
    urdf_file = os.path.join(pkg_share, 'urdf', 'simple_quad.urdf')    
    robot_description_content = Command([
        FindExecutable(name='xacro'),
        ' ',
        urdf_file
    ])
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_description_content, value_type=str)
        }]
    )
    
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
    )
    
    drone_state_publisher_node = Node(
        package='task22',
        executable='drone_state_publisher',
        name='drone_state_publisher',
        output='screen',
    )
    
    foxglove_bridge_node = ExecuteProcess(
        cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml'],
        output='screen'
    )
    
    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node,
        drone_state_publisher_node,
        foxglove_bridge_node
    ])