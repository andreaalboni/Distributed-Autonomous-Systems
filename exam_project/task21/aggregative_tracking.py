import numpy as np
from cost_function import *

def compute_agents_barycenter(agents):
    """Compute the barycenter (mean position) of a set of agents."""
    sigma = np.mean(agents, axis=0)
    return sigma

def compute_r_0(intruders, noise_radius, world_size, d):
    """Generates a random point near the barycenter of intruders within a specified noise radius."""
    barycenter_intruders = compute_agents_barycenter(intruders)
    if (noise_radius == 0.0):
        return barycenter_intruders
    while True:
        r_0_candidate = np.random.uniform(0, world_size, size=d)
        if (np.linalg.norm(r_0_candidate - barycenter_intruders) <= noise_radius and
            np.linalg.norm(r_0_candidate - barycenter_intruders) >= noise_radius/10):
                return r_0_candidate

def aggregative_tracking_method(agents, intruders, A, adj, noise_radius, world_size, d, gamma, gamma_bar, gamma_hat, max_iters, alpha=0.0001): 
    """
    Perform distributed aggregative tracking for a multi-agent system.
    Each agent updates its state based on local and neighbor information to track intruders
    while minimizing a local cost function and aggregating information from its neighbors.
    Args:
        agents (np.ndarray): Initial positions of agents, shape (n_agents, state_dim).
        intruders (np.ndarray): Positions of intruders, shape (n_agents, state_dim).
        A (np.ndarray): Weight matrix for consensus, shape (n_agents, n_agents).
        adj (np.ndarray): Adjacency matrix indicating agent connectivity, shape (n_agents, n_agents).
        noise_radius (float): Noise radius for intruder detection.
        world_size (float): Size of the environment.
        d (float): Distance parameter for cost computation.
        gamma (float or np.ndarray): Regularization parameter(s).
        gamma_bar (float or np.ndarray): Regularization parameter(s).
        gamma_hat (float or np.ndarray): Regularization parameter(s).
        max_iters (int): Number of iterations to run the algorithm.
        alpha (float, optional): Step size for gradient updates. Defaults to 0.0001.
    Returns:
        tuple: Tuple containing:
            - cost (np.ndarray): Cost at each iteration, shape (max_iters,).
            - z (np.ndarray): Agent states at each iteration, shape (max_iters, n_agents, state_dim).
            - r_0 (np.ndarray): Initial detection radii for intruders.
            - norm_grad_cost (np.ndarray): Norm of the gradient of the cost at each iteration, shape (max_iters,).
    """
    cost = np.zeros((max_iters))
    norm_grad_cost = np.zeros((max_iters))
    z = np.zeros((max_iters, len(agents), len(agents[0])))
    s = np.zeros((max_iters, len(agents), len(agents[0])))
    v = np.zeros((max_iters, len(agents), len(agents[0])))
    gamma = gamma * np.ones(len(agents))
    gamma_bar = gamma_bar * np.ones(len(agents))
    gamma_hat = gamma_hat * np.ones(len(agents))

    r_0 = compute_r_0(intruders, noise_radius, world_size, d)
    
    # Initialization
    z[0] = agents
    s[0] = agents   # phi_i(z_i) = z_i: regular barycenter
    for i in range(len(agents)):
        _, _, v[0, i] = local_cost_function(z[0, i], intruders[i], s[0, i], r_0, gamma[i], gamma_bar[i], gamma_hat[i])

    # Ch 8 p 7/11 
    for k in range(max_iters - 1):
        total_grad = np.zeros_like(v[0])
        for i in range(len(agents)):
            _, grad_1_l_i, _ = local_cost_function(z[k, i], intruders[i], s[k, i], r_0, gamma[i], gamma_bar[i], gamma_hat[i])    # in z_i^{k}, s_i^{k}
            _, grad_phi_i = local_phi_function(z[k, i])
            z[k+1, i] = z[k, i] - alpha * ( grad_1_l_i + grad_phi_i * v[k, i] )
            total_grad += grad_1_l_i
        
        for i in range(len(agents)):
            s[k+1, i] = A[i, i] * s[k, i]
            N_i = np.nonzero(adj[i])[0]
            for j in N_i:
                s[k+1, i] += A[i, j] * s[k, j]
            # phi_i(z_i^{k+1})
            s[k+1, i] += z[k+1, i] - z[k, i]
            
        for i in range(len(agents)):
            v[k+1, i] = A[i, i] * v[k, i]
            N_i = np.nonzero(adj[i])[0]
            for j in N_i:
                v[k+1, i] += A[i, j] * v[k, j]
            _, _, grad_2_l_i_new = local_cost_function(z[k+1, i], intruders[i], s[k+1, i], r_0, gamma[i], gamma_bar[i], gamma_hat[i])    # in z_i^{k+1}, s_i^{k+1}
            l_i, _, grad_2_l_i_old = local_cost_function(z[k, i], intruders[i], s[k, i], r_0, gamma[i], gamma_bar[i], gamma_hat[i])    # in z_i^{k}, s_i^{k}
            v[k+1, i] += grad_2_l_i_new - grad_2_l_i_old
            
            total_grad_2 = np.zeros_like(v[0])
            for j in range(len(agents)):
                total_grad_2 += grad_2_l_i_old
            _, grad_phi_i = local_phi_function(z[k, i])
            total_grad += grad_phi_i * total_grad_2 / len(agents)
            cost[k] += l_i
        
        norm_grad_cost[k] = np.linalg.norm(total_grad)

    return cost, z, r_0, norm_grad_cost