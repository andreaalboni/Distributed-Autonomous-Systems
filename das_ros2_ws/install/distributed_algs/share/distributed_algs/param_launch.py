from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx

N = 4
G = nx.complete_graph(N)
x = np.random.randint(low=0, high=100, size=N)

def generate_launch_description():
    node_list = []

    for i in range(N):
        node_list.append(
            Node(
                package="distributed_algs",
                namespace=f"agent_{i}",
                executable="generic_agent",
                parameters=[
                    {
                        "id": i,
                        "neighbors": list(G.neighbors(i)),
                        "xzero": int(x[i]),
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{i}"  -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
            )
        )

    return LaunchDescription(node_list)