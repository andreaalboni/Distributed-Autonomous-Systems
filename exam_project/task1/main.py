import numpy as np
from utils_graph import *
from utils_visualization import *
from utils_world_generation import *
from cost_functions import *
from gradient_tracking import gradient_tracking_algorithm


def main():
    
    PARAMETERS = {
        'num_targets': 3,
        'ratio_at': 7,              # ratio of agents to targets
        'world_size': [5, 5],
        'radius_fov': np.inf,
        'graph_type': 'cycle',
        'noise_level': 0.1,
        'bias': 0.0,
        'p_er': 0.5,
    }
    
    # Initialization
    targets, agents = generate_agents_and_targets(PARAMETERS['num_targets'],
                                                  PARAMETERS['ratio_at'],
                                                  PARAMETERS['world_size'])
    
    G, adj, A = generate_graph(len(agents), type=PARAMETERS['graph_type'], p_er=PARAMETERS['p_er'])
    real_distances, noisy_distances = get_distances(agents, targets)
    
    # Run traking algorithm
    z, cost, norm_grad_cost, prova = gradient_tracking_algorithm(agents,
                                                                 targets,
                                                                 noisy_distances,
                                                                 adj,
                                                                 A,
                                                                 local_cost_function_task1)
    
    # Visualization
    visualize_world(agents, targets, world_size=PARAMETERS['world_size'])
    plot_gradient_traking_results(z, cost, norm_grad_cost, prova, agents, targets, PARAMETERS['graph_type'])
    animate_world_evolution(agents, targets, type=PARAMETERS['graph_type'], z_history=z, world_size=PARAMETERS['world_size'])

if __name__ == "__main__":
    main()