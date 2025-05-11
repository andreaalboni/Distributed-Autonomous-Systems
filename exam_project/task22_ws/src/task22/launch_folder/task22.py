import numpy as np
from utils import *
from config import PARAMETERS
from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    ALPHA = 0.0001
    MAXITERS = 2000
    COMM_TIME = 5e-2
    N = PARAMETERS['num_intruders']

    graph_type = 'cycle'
    _, adj, A = generate_graph(N, type=graph_type)

    intruders, agents = generate_agents_and_intruders()
    z_0 = agents

    node_list = []
    package_name = "task22"

    for i in range(N):
        z_0_i = z_0[i]
        intruder_i = intruders[i]
        N_i = np.nonzero(adj[:, i])[0].tolist()

        node_list.append(
            Node(
                package=package_name,
                namespace=f"agent_{i}",
                executable="generic_agent",
                parameters=[
                    {
                        "id": i,
                        "z0": z_0_i,
                        "alpha": ALPHA,
                        "neighbors": N_i,
                        "max_iters": MAXITERS,
                        "intruder": intruder_i,
                        "communication_time": COMM_TIME,
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{i}"  -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
            )
        )
    return LaunchDescription(node_list)

generate_launch_description()