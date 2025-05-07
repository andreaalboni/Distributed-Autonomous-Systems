# Aggregative Optimization problem -> implement the algorithm Aggregative Tracking algorithm

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import PARAMETERS
from utils import *



def main(): 
    targets, agents = generate_agents_and_targets()



    G, Adj, A = generate_graph(len(agents), type='cycle')
    print("Adjacency Matrix:")
    print(Adj)  
    print("Graph:")
    print(G)

    visualize_world(agents, targets)

    plot_graph_with_connections(G)

if __name__ == "__main__":
    main()