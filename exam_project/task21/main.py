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
    
    cost, z, r_0 = aggregative_tracking_method(agents,
                                               intruders,
                                               A,
                                               adj,
                                               PARAMETERS['noise_r_0'],
                                               PARAMETERS['world_size'],
                                               PARAMETERS['d'])
    
    # Visualization
    plot_aggregative_tracking_results(cost)
    animate_world_evolution(intruders, z, r_0, PARAMETERS['world_size'], PARAMETERS['d'])
    return

if __name__ == "__main__":
    main()