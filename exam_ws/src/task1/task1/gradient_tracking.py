import numpy as np
from utils import *
from config import PARAMETERS
import matplotlib.pyplot as plt
from save_and_load import save_evolution_data


def gradient_tracking_method(max_iters=1500, alpha=0.001, save=False):
    targets, agents = generate_agents_and_targets()
    # visualize_world(agents, targets, world_size=params['world_size'])
    real_distances, noisy_distances = get_distances(agents, targets)
    graph_type = 'erdos_renyi'
    _, adj, A = generate_graph(len(agents), type=graph_type)
    # visualize_graph(G)
    
    z_opt = get_targets_real_positions(targets)
    
    cost = np.zeros((max_iters))
    z = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    s = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    
    # Randomly initialize z[0]
    # z[0] = params['world_size'][0]/2 * np.ones((z[0].shape))
    z[0] = np.random.uniform(0.0, min(PARAMETERS['world_size']), z[0].shape)
        
    for i in range(len(agents)):
        _, s[0, i] = local_cost_function(z[0, i], agents[i], noisy_distances[i])

    # alpha = alpha / params['world_size'][0]
    # Ch 6 p 14
    for k in range(max_iters - 1):
        # ------------------ Armijo Linesearch: ------------------
        # def evaluate_total_cost_function(z_k):
        #     total_cost = 0
        #     for i in range(len(agents)):
        #         l_i, _ = local_cost_function(z_k[i], agents[i], real_distances[i])
        #         total_cost += l_i
        #     return total_cost
        # 
        # first_comp = second_comp = 0
        # for i in range(len(agents)):
        #     first_comp += s[k, i][0][0]
        #     second_comp += s[k, i][0][1] 
        # 
        # grad_total_cost = np.array([first_comp, second_comp]) / len(agents)
        # dir_der = - grad_total_cost.T @ grad_total_cost
        # 
        # if k > max_iters / 5:    
        #     alpha = Armijo_linesearch(
        #         f = evaluate_total_cost_function,
        #         search_direction = -grad_total_cost,
        #         z0 = z[k],
        #         fz0 = evaluate_total_cost_function(z[k]),
        #         directional_derivative_z0 = dir_der,
        #         alpha_init = 1e-3
        #     )
        # --------------------------------------------------------
        
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
            cost[k] += l_i
    
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
    ax = axes[0]
    ax.semilogy(np.arange(max_iters-1), cost[:-1])
    ax.set_title('Cost vs Iteration')
    ax.set_xlabel('Iteration')
    ax = axes[1]
    for i in range(len(targets)):
        for j in range(len(agents)):
            errors = [np.linalg.norm(z[t, j, i] - z_opt[i]) for t in range(max_iters-1)]
            ax.semilogy(np.arange(max_iters-1), errors)
    # ax.legend()
    ax.set_title('Estimation error vs Iteration')
    ax.set_xlabel('Iteration')
    plt.show()
    
    # print(f"z optimal: {z_opt}")
    # print(f"estimated positions of targets: {z[-1, :, :, :]}")
    if save:
        save_evolution_data(agents, targets, z, type=graph_type)
    animate_world_evolution(agents, targets, type=graph_type, z_hystory=z)
    return z, cost

def doll_gradient_tracking_method(max_iters=400, alpha=0.001):
    # -------------- Not working yet ------------------
    targets, agents = generate_agents_and_targets(
        num_targets=params['num_targets'],
        ratio_at=params['ratio_at'],
        world_size=params['world_size'],
        radius_fov=params['radius_fov']
    )
    #visualize_world(agents, targets, world_size=params['world_size'])
    real_distances, noisy_distances = get_distances(agents, targets)
    G, adj, A = generate_graph(len(agents), type='erdos_renyi', p_er=0.5)
    #visualize_graph(G)
    
    Q = np.zeros((len(agents), len(targets), len(targets)))
    for i in range(len(agents)):
        Q[i] = np.eye(len(targets)) * 3
    
    z_opt = get_targets_real_positions(targets)
    
    cost = np.zeros((max_iters))
    z = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    s = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    
    z[0] = np.random.uniform(params['world_size'][0]/2, params['world_size'][1]/2, z[0].shape)
        
    for i in range(len(agents)):
        _, s[0, i] = doll_cost_function(z[0, i], Q[i], noisy_distances[i])

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
            
            _, grad_l_i_new = doll_cost_function(z[k+1, i], Q[i], noisy_distances[i])
            _, grad_l_i_old = doll_cost_function(z[k, i], Q[i], noisy_distances[i])
            s[k+1, i] += grad_l_i_new - grad_l_i_old
            
            l_i, _ = doll_cost_function(z[k, i], Q[i], noisy_distances[i])
            cost[k] += l_i
    
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
    ax = axes[0]
    ax.semilogy(np.arange(max_iters-1), cost[:-1])
    ax.set_title('Cost vs Iteration')
    ax.set_xlabel('Iteration')
    ax = axes[1]
    for i in range(len(targets)):
        for j in range(len(agents)):
            errors = [np.linalg.norm(z[t, j, i] - z_opt[0, i]) for t in range(max_iters-1)]
            ax.semilogy(np.arange(max_iters-1), errors, label=f'Agent {j+1}, Target {i+1}')
    #ax.legend()
    ax.set_title('Estimation error vs Iteration')
    ax.set_xlabel('Iteration')
    ax.grid(True)
    plt.show()
    
    animate_world_evolution(agents, targets, world_size=params['world_size'], z_hystory=z)
    return z, cost

def main():
    gradient_tracking_method()

if __name__ == "__main__":
    main()