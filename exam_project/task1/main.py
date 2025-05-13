import numpy as np
from utils_graph import *
from utils_visualization import *
from utils_world_generation import *
from cost_functions import *
from config import PARAMETERS
from save_and_load import save_evolution_data
from gradient_tracking import gradient_tracking_algorithm


def main():
    
    save = False
    # Initialization
    targets, agents = generate_agents_and_targets()
    real_distances, noisy_distances = get_distances(agents, targets)
    graph_type = 'cycle'
    G, adj, A = generate_graph(len(agents), type=graph_type, p_er=PARAMETERS['p_er'])
    
    # Run traking algorithm
    z, cost, norm_grad_cost, prova = gradient_tracking_algorithm(agents,
                                                                 targets,
                                                                 noisy_distances,
                                                                 adj,
                                                                 A,
                                                                 local_cost_function_task2)
    
    # Visualization
    visualize_world(agents, targets, world_size=PARAMETERS['world_size'])
    plot_gradient_traking_results(z, cost, norm_grad_cost, prova, agents, targets, graph_type)
    print(f"Final error norm: {np.linalg.norm(PARAMETERS['world_size'] * z[-1,0] - PARAMETERS['world_size'] * targets)}")
    animate_world_evolution(agents, targets, type=graph_type, z_history=z, world_size=PARAMETERS['world_size'])
    
    # Save data if requested
    if save:
        save_evolution_data(agents, targets, z, type=graph_type, world_size=PARAMETERS['world_size'])


if __name__ == "__main__":
    main()