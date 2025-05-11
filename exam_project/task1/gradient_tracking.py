import numpy as np
from utils import *
from config import PARAMETERS
import matplotlib.pyplot as plt
from save_and_load import save_evolution_data

# extend the length of the print of numpy arrays
np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)


def gradient_tracking_method(max_iters=2000, alpha=0.035, save=False):
    targets, agents = generate_agents_and_targets()
    # visualize_world(agents, targets, world_size=params['world_size'])
    real_distances, noisy_distances = get_distances(agents, targets)
    graph_type = 'cycle'
    G, adj, A = generate_graph(len(agents), type=graph_type)
    # visualize_graph(G)

    cost = np.zeros((max_iters))
    norm_grad_cost = np.zeros((max_iters))
    prova = np.zeros((max_iters))
    z = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    s = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    
    # Randomly initialize z[0] - Normalized
    # z[0] = 1.0/2 * np.ones((z[0].shape))
    z[0] = np.random.uniform(0.0, 1.0, z[0].shape)
    # z[0] = np.zeros((len(agents), len(targets), targets.shape[1]))

    # initialize gradients
    for i in range(len(agents)):
        _, s[0, i] = local_cost_function(z[0, i], agents[i], noisy_distances[i])

    # alpha = alpha / params['world_size'][0]
    # Ch 6 p 14 
    for k in range(max_iters - 1):        
        total_grad = np.zeros_like(s[0])
        
        # position update
        for i in range(len(agents)):
            z[k+1, i] = A[i, i] * z[k, i]
            N_i = np.nonzero(adj[i])[0]
            for j in N_i: # for each neighbor
                z[k+1, i] += A[i, j] * z[k, j]    
            z[k+1, i] -= alpha * s[k, i]
        
        # gradient update
        for i in range(len(agents)):
            s[k+1, i] = A[i, i] * s[k, i]
            N_i = np.nonzero(adj[i])[0]
            for j in N_i:
                s[k+1, i] += A[i, j] * s[k, j]
                
            _, grad_l_i_new = local_cost_function(z[k+1, i], agents[i], real_distances[i])
            l_i, grad_l_i_old = local_cost_function(z[k, i], agents[i], real_distances[i])
            s[k+1, i] += grad_l_i_new - grad_l_i_old
            # if (np.any(np.isnan(grad_l_i_new)) or np.any(np.isnan(grad_l_i_old))):
            #     print(f"NaN in gradient update for agent {i} at iteration {k+1}")
            
            total_grad += s[k+1,i]
            cost[k] += l_i

        norm_grad_cost[k] = np.linalg.norm(total_grad / len(agents))
        prova[k] = np.linalg.norm(s[k,0])

    print("Normalized values:")
    print(f"estimates: {z[-1,0]}")
    print(f"targets: {targets}")

    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
    ax = axes[0]
    ax.semilogy(np.arange(max_iters-1), cost[:-1], color='violet')
    ax.set_title('Cost vs Iteration')
    ax.set_xlabel('Iteration')
    
    ax = axes[1]
    ax.semilogy(np.arange(max_iters-1), norm_grad_cost[:-1], color='cyan')
    ax.semilogy(np.arange(max_iters-1), prova[:-1], color='purple')
    ax.set_title('Gradient of the cost vs Iteration')
    ax.set_xlabel('Iteration')
    
    """
    ax = axes[1]
    for i in range(len(targets)):
        for j in range(len(agents)):
            errors = [np.linalg.norm(z[t, j, i] - z_opt[i]) for t in range(max_iters-1)]
            ax.semilogy(np.arange(max_iters-1), errors)
    # ax.legend()
    ax.set_title('Estimation error vs Iteration')
    ax.set_xlabel('Iteration') 
    """
    plt.show()
    
    # print(f"z optimal: {z_opt}")
    # print(f"estimated positions of targets: {z[-1, :, :, :]}")
    if save:
        save_evolution_data(agents, targets, z, type=graph_type, world_size=PARAMETERS['world_size'])
    animate_world_evolution(agents, targets, type=graph_type, z_history=z)
    return z, cost


def main():
    gradient_tracking_method()

if __name__ == "__main__":
    main()