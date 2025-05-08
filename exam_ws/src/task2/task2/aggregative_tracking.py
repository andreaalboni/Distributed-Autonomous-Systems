import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import PARAMETERS
from utils import *


def aggregative_tracking_method(): 
    targets, agents = generate_agents_and_targets()
    print(get_distances(agents, targets))


    G, Adj, A = generate_graph(len(agents), type='cycle')
    print("Adjacency Matrix:")
    print(Adj)  
    print("Graph:")
    print(G)

    sigma = compute_aggregative_variale(agents)
    visualize_world(agents, targets, sigma)

    plot_graph_with_connections(G)
    local_cost, gradient = local_cost_function(agents[0], targets[0], sigma)
    print("Local Cost:", local_cost)
    print("Gradient:", gradient)

def main(): 
    aggregative_tracking_method()

if __name__ == "__main__":
    main()