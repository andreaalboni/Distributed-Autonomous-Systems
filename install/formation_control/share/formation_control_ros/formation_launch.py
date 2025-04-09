from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np


def generate_launch_description():
    MAXITERS = 200
    COMM_TIME = 5e-2  # communication time period
    NN = 6  # number of agents
    n_x = 2  # dimension of each x_i

    # Weight matrix to control inter-agent distances
    L = 2
    D = 2 * L
    H = np.sqrt(3) * L

    # minimally rigid 2*N-3 (only for regular polygons)
    distances = np.array(
        [
            [0, L, 0, D, H, L],
            [L, 0, L, 0, D, 0],
            [0, L, 0, L, 0, D],
            [D, 0, L, 0, L, 0],
            [H, D, 0, L, 0, L],
            [L, 0, D, 0, L, 0],
        ]
    )

    # Adjacency matrix
    Adj = distances > 0

    # definite initial positions
    x_init = np.random.rand(n_x * NN, 1)

    node_list = []  # Append here your nodes
    package_name = "formation_control_ros"

    for ii in range(NN):
        distances_ii = distances[:, ii].tolist()

        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        ii_index = ii * n_x + np.arange(n_x)
        x_init_ii = x_init[ii_index].flatten().tolist()

        node_list.append(
            Node(
                package=package_name,
                namespace=f"agent_{ii}",
                executable="generic_agent",
                parameters=[
                    {
                        "id": ii,
                        "communication_time": COMM_TIME,
                        "neighbors": N_ii,
                        "xzero": x_init_ii,
                        "dist": distances_ii,
                        "maxT": MAXITERS,
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{ii}"  -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
            )
        )

    return LaunchDescription(node_list)
