import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import PARAMETERS
from utils import *


def aggregative_tracking_method(): 
    targets, agents = generate_agents_and_targets()
    visualize_world(agents, targets)
    print(get_distances(agents, targets))
    return


    G, Adj, A = generate_graph(len(agents), type='cycle')
    print("Adjacency Matrix:")
    print(Adj)  
    print("Graph:")
    print(G)

    visualize_world(agents, targets)

    plot_graph_with_connections(G)


def main(): 
    aggregative_tracking_method()

if __name__ == "__main__":
    main()