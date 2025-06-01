from utils_graph import *
from utils_visualization import *
from utils_world_generation import *
from cost_function import *
from aggregative_tracking import aggregative_tracking_method

def main():
    
    PARAMETERS = {
        'num_intruders': 5,
        'world_size': 20,
        'd': 3,
        'intruder_radius': 10.0,
        'radius_spawn_agent': 5.0,
        'noise_r_0': 0.0,
        'graph_type': 'cycle',
        'gamma': 15,
        'gamma_bar': 3,
        'gamma_hat': 1,
        'max_iters': 12000,
    }
        
    # Initialization
    intruders, agents = generate_agents_and_intruders(PARAMETERS['num_intruders'],
                                                  PARAMETERS['world_size'],
                                                  PARAMETERS['radius_spawn_agent'],
                                                  PARAMETERS['d'],
                                                  PARAMETERS['intruder_radius'])

    visualize_world(agents, intruders, PARAMETERS['noise_r_0'], PARAMETERS['world_size'], PARAMETERS['d'])    
    G, adj, A = generate_graph(len(agents), PARAMETERS['graph_type'])
    # visualize_graph(G)
    
    cost, z, r_0, total_grad_cost = aggregative_tracking_method(agents,
                                                                intruders,
                                                                A,
                                                                adj,
                                                                PARAMETERS['noise_r_0'],
                                                                PARAMETERS['world_size'],
                                                                PARAMETERS['d'],
                                                                PARAMETERS['gamma'],
                                                                PARAMETERS['gamma_bar'],
                                                                PARAMETERS['gamma_hat'],
                                                                PARAMETERS['max_iters'])
    
    # Visualization
    plot_aggregative_tracking_results(cost, total_grad_cost)
    animate_world_evolution(intruders, z, r_0, PARAMETERS['world_size'], PARAMETERS['d'])
    return

if __name__ == "__main__":
    main()