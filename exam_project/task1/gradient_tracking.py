import numpy as np

def gradient_tracking_algorithm(agents, targets, noisy_distances, adj, A, local_cost_function, max_iters=2250, alpha=0.025):
    """
    Implements the gradient tracking algorithm for distributed optimization.
    """
    cost = np.zeros((max_iters))
    norm_grad_cost = np.zeros((max_iters))
    prova = np.zeros((max_iters))
    z = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    s = np.zeros((max_iters, len(agents), len(targets), targets.shape[1]))
    
    # Randomly initialize z[0]
    z[0] = np.random.uniform(0.0, 1.0, z[0].shape)

    # initialize gradients
    for i in range(len(agents)):
        _, s[0, i] = local_cost_function(agents[i], z[0, i], noisy_distances[i])
    
    # Gradient tracking algorithm
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
            
            _, grad_l_i_new = local_cost_function(agents[i], z[k+1, i], noisy_distances[i])
            l_i, grad_l_i_old = local_cost_function(agents[i], z[k, i], noisy_distances[i])
            s[k+1, i] += grad_l_i_new - grad_l_i_old
            
            total_grad += s[k+1,i]
            cost[k] += l_i

        norm_grad_cost[k] = np.linalg.norm(total_grad / len(agents))
        prova[k] = np.linalg.norm(s[k,0])

    return z, cost, norm_grad_cost, prova