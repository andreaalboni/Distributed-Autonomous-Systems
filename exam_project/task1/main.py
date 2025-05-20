import numpy as np
from utils_graph import *
from utils_visualization import *
from utils_world_generation import *
from cost_functions import *
from gradient_tracking import gradient_tracking_method
from termcolor import colored

def main():
    
    PARAMETERS = {
        'num_targets': 2,
        'ratio_at': 5,
        'd': 2,
        'world_size': 5,
        'radius_fov': np.inf,
        'noise_level': 0.0,
        'bias': 0.0,
        'graph_type': 'cycle',
    }
    
    task_to_run = ['1.1', '1.2']
    
    task_functions = {
        '1.1': local_cost_function_task1,
        '1.2': local_cost_function_task2,
    }
    
    #TODO: flag to print some data like adj, distances, etc..
    
    
    # Initialization
    targets, agents = generate_agents_and_targets(PARAMETERS['num_targets'],
                                                  PARAMETERS['ratio_at'],
                                                  PARAMETERS['world_size'],
                                                  PARAMETERS['d'],
                                                  PARAMETERS['radius_fov'])

    G, adj, A = generate_graph(len(agents), PARAMETERS['graph_type'])
    
    real_distances, noisy_distances = get_distances(agents, 
                                                    targets, 
                                                    PARAMETERS['noise_level'],
                                                    PARAMETERS['bias'],
                                                    PARAMETERS['radius_fov'],
                                                    PARAMETERS['world_size'])
    
    for task in task_to_run:
        # Run traking algorithm
        # TODO: remove parm
        print(colored(f"\n----------------Starting task {task}----------------\n ", 'green'))
        cost_function = task_functions[task]
        z_hystory, cost, norm_grad_cost, prova, norm_error = gradient_tracking_method(agents,
                                                                                      targets,
                                                                                      noisy_distances,
                                                                                      adj,
                                                                                      A,
                                                                                      cost_function)
        
        # Visualization
        visualize_world(agents, targets, PARAMETERS['world_size'], PARAMETERS['d'])
        plot_gradient_tracking_results(z_hystory, cost, norm_grad_cost, prova, agents, targets, norm_error)
        animate_world_evolution(agents, targets, z_hystory, PARAMETERS['graph_type'], PARAMETERS['world_size'], PARAMETERS['d'])

if __name__ == "__main__":
    main()