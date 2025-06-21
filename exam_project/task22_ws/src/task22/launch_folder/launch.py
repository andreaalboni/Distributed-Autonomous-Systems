import numpy as np
import networkx as nx
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import TimerAction
from task22.utils import generate_agents_and_intruders, generate_graph, compute_r_0

PARAMETERS = {
    'num_intruders': 5,
    'world_size': 20,
    'd': 3,
    'intruder_radius': 10.0,
    'radius_spawn_agent': 5.0,
    'noise_r_0': 0.0,
    'graph_type': 'star', # Options: 'path', 'cycle', 'star', 'erdos_renyi'
    'max_iters': 5000,
    'alpha': 0.1,
    'gamma': 15,
    'gamma_bar': 3,
    'gamma_hat': 1,
    
    'safety_control': True,
        'gamma_sc': 20,             # Safety control gain
        'fov_horizontal': 360,      # Horizontal Field of View in degrees
        'fov_vertical': 360,        # Vertical Field of View in degrees
        'fov_range': 3.0,           # Range of the Field of View
        'safety_distance': 3.0,     # Safety distance for agents
    
    'real_dynamics': True,      # Use real dynamics for agents
        'u_max': 100.0,         # Maximum control input
        'tracking_tolerance': 1e-2,
    
    'communication_time': 1e-2,
}

def generate_launch_description():
    """
    Generates a ROS 2 launch description for the task22 project.
    This function sets up and launches multiple nodes required for the simulation.
    Nodes launched:
        - foxglove_bridge: Provides a bridge for visualization and debugging.
        - visualizer: Visualizes agent and intruder positions and simulation state.
        - plotter: Handles plotting of simulation results.
        - lidars: Simulates lidar sensors for agents.
        - agent_{i} (one per intruder): Runs the agent logic for each intruder, with parameters specific to each agent.
    Returns:
        LaunchDescription: A ROS 2 LaunchDescription object containing all configured nodes and actions.
    """
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

    fov_hor_angle = 0.0
    fov_vert_angle = 0.0
    if PARAMETERS['fov_horizontal'] == 360:
        fov_hor_angle = 360
    else:
        fov_hor_angle = PARAMETERS['fov_horizontal'] % 360
    if PARAMETERS['fov_vertical'] == 360:
        fov_vert_angle = 360
    else:
        fov_vert_angle = PARAMETERS['fov_vertical'] % 360

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
            'safety_control': PARAMETERS['safety_control'],
                'fov_horizontal': fov_hor_angle,
                'fov_vertical': fov_vert_angle,
                'fov_range': PARAMETERS['fov_range'],
        }],
    )
    node_list.append(visualizer_node)
    
    plotter_node = Node(
        package='task22',
        executable='plotter',
        name='plotter',
        output='screen',
        parameters=[{
            'max_iters': PARAMETERS['max_iters'],
            'num_intruders': PARAMETERS['num_intruders'],
            'd': PARAMETERS['d'],
        }],
    )
    node_list.append(plotter_node)
    
    lidars_node = Node(
        package='task22',
        executable='lidars',
        name='lidars',
        output='screen',
        parameters=[{
            'd': PARAMETERS['d'],
            'safety_control': PARAMETERS['safety_control'],
            'fov_horizontal': fov_hor_angle,
            'fov_vertical': fov_vert_angle,
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
                                "neighbors": N_i,
                                "intruder": intruder_i,
                                "gamma_bar": gamma_bar_i,
                                "gamma_hat": gamma_hat_i,
                                "alpha": float(PARAMETERS['alpha']),
                                "u_max": float(PARAMETERS['u_max']),
                                "max_iters": int(PARAMETERS['max_iters']),
                                "gamma_sc": float(PARAMETERS['gamma_sc']),
                                "real_dynamics": PARAMETERS['real_dynamics'],
                                'safety_control': PARAMETERS['safety_control'],
                                "safety_distance": float(PARAMETERS['safety_distance']),
                                "tracking_tolerance": float(PARAMETERS['tracking_tolerance']),
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