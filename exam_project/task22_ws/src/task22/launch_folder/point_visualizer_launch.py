from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='task22',
            executable='point_visualizer',
            name='point_visualizer_node',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml'],
            output='screen'
        )
    ])