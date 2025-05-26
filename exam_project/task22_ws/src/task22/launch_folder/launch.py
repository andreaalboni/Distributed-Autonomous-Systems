import numpy as np
import networkx as nx
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from task22.utils import generate_agents_and_intruders, generate_graph, compute_r_0

PARAMETERS = {
    'num_intruders': 3,
    'world_size': 20,
    'd': 2,
    'intruder_radius': 10.0,
    'radius_spawn_agent': 5.0,
    'noise_r_0': 0.0,
    'graph_type': 'cycle',
    'max_iters': 2500,
    'alpha': 0.1,
    'gamma': 15,
    'gamma_bar': 3,
    'gamma_hat': 1,
    'gamma_sc': 10,  # Safety control gain
    'fov_horizontal': 180,    # Horizontal Field of View in degrees
    'fov_vertical': 30,       # Vertical Field of View in degrees
    'fov_range': 3.0,         # Range of the Field of View
    'safety_distance': 2.0,  # Safety distance for agents
    "u_max": 100.0,  # Maximum control input
    'communication_time': 1e-2,
}

def generate_launch_description():
    _, adj, A = generate_graph(PARAMETERS['num_intruders'], type=PARAMETERS['graph_type'])
    intruders, agents = generate_agents_and_intruders(PARAMETERS['num_intruders'], 
                                                      PARAMETERS['world_size'], 
                                                      PARAMETERS['radius_spawn_agent'], 
                                                      PARAMETERS['intruder_radius'],
                                                      PARAMETERS['d'])
    
    r_0 = compute_r_0(intruders, PARAMETERS['noise_r_0'], PARAMETERS['world_size'], PARAMETERS['d']).tolist()
    z_0 = agents

    gamma = PARAMETERS['gamma'] * np.ones(len(agents))
    gamma_bar = PARAMETERS['gamma_bar'] * np.ones(len(agents))
    gamma_hat = PARAMETERS['gamma_hat'] * np.ones(len(agents))

    node_list = []
    package_name = "task22"
    
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        arguments=['--ros-args', '--log-level', 'ERROR']
    )
    node_list.append(foxglove_bridge_node)

    visualizer_node = Node(
        package='task22',
        executable='visualizer',
        name='visualizer',
        output='screen',
        parameters=[{
            # Flatten the intruders array and pass as a list
            'initial_agent_positions': z_0.flatten().tolist(),
            'intruders': intruders.flatten().tolist(),
            'r_0': r_0,
            'world_size': PARAMETERS['world_size'],
            'd': PARAMETERS['d'],
            'fov_horizontal': PARAMETERS['fov_horizontal'],
            'fov_vertical': PARAMETERS['fov_vertical'],
            'fov_range': PARAMETERS['fov_range'],
        }],
    )
    node_list.append(visualizer_node)
    
    lidars_node = Node(
        package='task22',
        executable='lidars',
        name='lidars',
        output='screen',
        parameters=[{
            'd': PARAMETERS['d'],
            'fov_horizontal': PARAMETERS['fov_horizontal'],
            'fov_vertical': PARAMETERS['fov_vertical'],
            'fov_range': PARAMETERS['fov_range'],
        }],
    )
    node_list.append(lidars_node)

    for i in range(PARAMETERS['num_intruders']):
        gamma_i = float(gamma[i])
        A_i = A[i].tolist()
        z_0_i = z_0[i].tolist()
        gamma_bar_i = float(gamma_bar[i])
        gamma_hat_i = float(gamma_hat[i])
        intruder_i = intruders[i].tolist()
        N_i = np.nonzero(adj[:, i])[0].tolist()

        node_list.append(
            TimerAction(period=5.0,
                actions=[
                    Node(
                        package=package_name,
                        namespace=f"agent_{i}",
                        executable="agent",
                        parameters=[
                            {
                                "id": i,
                                "A_i": A_i,
                                "r_0": r_0,
                                "z0": z_0_i,
                                "gamma": gamma_i,
                                "alpha": float(PARAMETERS['alpha']),
                                "neighbors": N_i,
                                "max_iters": int(PARAMETERS['max_iters']),
                                "intruder": intruder_i,
                                "gamma_bar": gamma_bar_i,
                                "gamma_hat": gamma_hat_i,
                                "gamma_sc": float(PARAMETERS['gamma_sc']),
                                "safety_distance": float(PARAMETERS['safety_distance']),
                                "u_max": float(PARAMETERS['u_max']),
                                "communication_time": float(PARAMETERS['communication_time']),
                            }
                        ],
                        output="screen",
                        prefix=f'xterm -geometry 100x3+{0}+{100 + i*100} -fa "Monospace" -fs 11 -title "agent_{i}" -fg white -bg black -hold -e',
                    )
                ]
            )
        )

    return LaunchDescription(node_list)