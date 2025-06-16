import numpy as np
from utils_graph import *
from cost_functions import *
from termcolor import colored
from utils_visualization import *
from utils_world_generation import *
from gradient_tracking import gradient_tracking_method

def main():
    
    PARAMETERS = {
        'num_targets': 3,
        'ratio_at': 4,              # Ratio of agents to targets
        'd': 2,                     # Dimension of the world 
        'world_size': 5,            # Size of the world (world_size x world_size)
        'radius_fov': np.inf,       # Radius of the field of view
        'noise_level': 0.0,         # Noise level for distance measurements
        'bias': 0.0,                # Bias for distance measurements  
        'graph_type': 'path',       # 'cycle', 'star', 'erdos_renyi', 'path'
        'max_iters': 10000,
    }
    
    task_to_run = ['1.1', '1.2']
    
    task_settings = {
        '1.1': [local_cost_function_task1, 1.5e-4],
        '1.2': [local_cost_function_task2, 0.04],
    }
    
    # Initialization
    targets, agents = generate_agents_and_targets(PARAMETERS['num_targets'],
                                                  PARAMETERS['ratio_at'],
                                                  PARAMETERS['world_size'],
                                                  PARAMETERS['d'],
                                                  PARAMETERS['radius_fov'])

    G, adj, A = generate_graph(len(agents), PARAMETERS['graph_type'])
    
    
    for task in task_to_run:
        # Run traking algorithm
        print(colored(f"\n---------------- Starting task {task} ----------------\n", "green"))
        cost_function = task_settings[task][0]
        alpha = task_settings[task][1]
        
        if task == '1.1':
            Q, b = quadratic_cost_function_param(PARAMETERS['d'])  # Generate Q and b for task 1.1
    
        real_distances, noisy_distances = get_distances(agents, 
                                                        targets, 
                                                        PARAMETERS['noise_level'],
                                                        PARAMETERS['bias'],
                                                        PARAMETERS['radius_fov'],
                                                        PARAMETERS['world_size'],
                                                        task)
        
        z_hystory, cost, norm_grad_cost, norm_error = gradient_tracking_method(agents,
                                                                               targets,
                                                                               noisy_distances,
                                                                               adj,
                                                                               A,
                                                                               cost_function,
                                                                               alpha,
                                                                               PARAMETERS['max_iters'],
                                                                               task,
                                                                               Q if task == '1.1' else None,
                                                                               b if task == '1.1' else None)
        
        # Visualization
        #visualize_world(agents, targets, PARAMETERS['world_size'], PARAMETERS['d'])
        plot_gradient_tracking_results(z_hystory, cost, norm_grad_cost, agents, targets, norm_error, task)
        animate_world_evolution(agents, targets, z_hystory, PARAMETERS['graph_type'], PARAMETERS['world_size'], PARAMETERS['d'])

if __name__ == "__main__":
    main()