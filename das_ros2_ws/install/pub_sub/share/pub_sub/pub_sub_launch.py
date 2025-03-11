from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    talker_node = Node(
        package="pub_sub",
        executable="talker",
        output="screen",
        prefix='xterm -title "The Talker" -fa "Monospace" -fs 12 -hold -e',
    )

    listener_node = Node(
        package="pub_sub",
        executable="listener",
        output="screen",
        prefix='xterm -title "The Listener" -fa "Monospace" -fs 12 -hold -e',
    )
    return LaunchDescription([talker_node, listener_node])
