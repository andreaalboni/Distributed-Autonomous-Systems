import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import *

def gradient_tracking_method(params, max_iters=400, alpha=0.001):
    targets, agents = generate_agents_and_targets(
        num_targets=params['num_targets'],
        ratio_at=params['ratio_at'],
        world_size=params['world_size'],
        radius_fov=params['radius_fov']
    )
    visualize_world(agents, targets, world_size=params['world_size'])
    real_distances, noisy_distances = get_distances(agents, targets)
    G, adj, A = generate_graph(len(agents), type='cycle', p_er=0.5)
    
    z_opt = get_targets_real_positions(targets)

    cost = np.zeros((max_iters))
    z = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    s = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    
    # Randomly initialize z[0]
    z[0] = np.random.uniform(params['world_size'][0]/2, params['world_size'][1]/2, z[0].shape)
        
    for i in range(len(agents)):
        _, s[0, i] = local_cost_function(z[0, i], agents[i], noisy_distances[i])
    print("s[0]: ", s[0])
    
    print("adj: ", adj)
    print("A: ", A)
    
    print(adj-A)

    # ch.6 p.14
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
            
            _, grad_l_i_new = local_cost_function(z[k+1, i], agents[i], noisy_distances[i])
            l_i, grad_l_i_old = local_cost_function(z[k, i], agents[i], noisy_distances[i])
            s[k+1, i] += grad_l_i_new - grad_l_i_old

            cost[k] += l_i
    
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
    ax = axes[0]
    ax.semilogy(np.arange(max_iters-1), cost[:-1])
    ax = axes[1]
    for i in range(len(targets)):
        errors = [np.linalg.norm(z[t, 0, i] - z_opt[i]) for t in range(max_iters-1)]
        ax.semilogy(np.arange(max_iters-1), errors, label=f'Target {i+1}')
        ax.legend()
    plt.show()
    
    animate_world_evolution(agents, targets, world_size=params['world_size'], z_hystory=z)
    return z, cost

def main():
    params = get_default_params()
    z_hystory, cost_hystory = gradient_tracking_method(params)

if __name__ == "__main__":
    main()