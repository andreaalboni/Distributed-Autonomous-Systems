import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import PARAMETERS
from utils import *


def aggregative_tracking_method(max_iters=200, alpha=0.01): 
    intruders, agents = generate_agents_and_intruders()
    visualize_world(agents, intruders)
    graph_type = 'cycle'
    _, adj, A = generate_graph(len(agents), type=graph_type)
    # visualize_graph(G)
    
    cost = np.zeros((max_iters))
    z = np.zeros((max_iters, len(agents), len(agents[0])))
    s = np.zeros((max_iters, ))
    v = np.zeros((max_iters, ))
    
    # Initialization
    z[0] = agents
        
    for i in range(len(agents)):
        _, s[0, i] = local_cost_function(z[0, i], agents[i], noisy_distances[i])

    # alpha = alpha / params['world_size'][0]
    # Ch 6 p 14 
    for k in range(max_iters - 1):
        for i in range(len(agents)):
            z[k+1, i] = A[i, i] * z[k, i]
            N_i = np.nonzero(adj[i])[0]
            for j in N_i:
                z[k+1, i] += A[i, j] * z[k, j]    
            z[k+1, i] -= alpha * s[k, i]
        
        for i in range(len(agents)):
            s[k+1, i] = A[i, i] * s[k, i]
            N_i = np.nonzero(adj[i])[0]
            for j in N_i:
                s[k+1, i] += A[i, j] * s[k, j]
                
            _, grad_l_i_new = local_cost_function(z[k+1, i], agents[i], real_distances[i])
            l_i, grad_l_i_old = local_cost_function(z[k, i], agents[i], real_distances[i])
            s[k+1, i] += grad_l_i_new - grad_l_i_old
            
            total_grad += s[k+1,i]
            cost[k] += l_i

def main(): 
    aggregative_tracking_method()

if __name__ == "__main__":
    main()