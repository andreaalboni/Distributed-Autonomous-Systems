import numpy as np

def gradient_tracking_method(agents, targets, noisy_distances, adj, A, local_cost_function, alpha, max_iters):
    """
    Implements the gradient tracking algorithm for distributed optimization.
    Args:
        agents (np.ndarray): Array of agent positions or states.
        targets (np.ndarray): Array of target positions.
        noisy_distances (list or np.ndarray): Noisy distance measurements for each agent.
        adj (np.ndarray): Adjacency matrix representing the communication graph.
        A (np.ndarray): Weight matrix for consensus updates.
        local_cost_function (callable): Function to compute local cost and gradient.
        alpha (float): Step size for the gradient update.
        max_iters (int): Maximum number of iterations.
    Returns:
        tuple: 
            z (np.ndarray): Trajectory of agent estimates over iterations.
            cost (np.ndarray): Accumulated local costs at each iteration.
            norm_grad_cost (np.ndarray): Norm of the aggregated gradient at each iteration.
            norm_error (np.ndarray): Norm of the error between agent estimates and targets.
    """
    cost = np.zeros((max_iters))
    norm_grad_cost = np.zeros((max_iters, len(targets)))
    norm_error = np.zeros((max_iters, len(agents), len(targets)))
    z = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    s = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    
    # Randomly initialize z[0]
    z[0] = np.random.uniform(0.0, 1.0, z[0].shape)

    for i in range(len(agents)):
        _, s[0, i] = local_cost_function(z[0, i], agents[i], noisy_distances[i])

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
                
            _, grad_l_i_new = local_cost_function(z[k+1, i], agents[i], noisy_distances[i])
            l_i, grad_l_i_old = local_cost_function(z[k, i], agents[i], noisy_distances[i])
            s[k+1, i] += grad_l_i_new - grad_l_i_old
            total_grad += grad_l_i_old
            cost[k] += l_i

        for i in range(len(agents)):
            for j in range(len(targets)):
                norm_error[k, i, j] += np.linalg.norm(z[k, i, j] - targets[j])
        norm_grad_cost[k] = np.linalg.norm(total_grad)
    
    return z, cost, norm_grad_cost, norm_error