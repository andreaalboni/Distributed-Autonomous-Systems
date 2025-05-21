import numpy as np

# extend the length of the print of numpy arrays
np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)

def gradient_tracking_method(agents, targets, noisy_distances, adj, A, local_cost_function, alpha, max_iters=10250):
    cost = np.zeros((max_iters))
    norm_grad_cost = np.zeros((max_iters, len(targets)))
    norm_error = np.zeros((max_iters, len(agents), len(targets)))
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
                
            _, grad_l_i_new = local_cost_function(z[k+1, i], agents[i], noisy_distances[i])
            l_i, grad_l_i_old = local_cost_function(z[k, i], agents[i], noisy_distances[i])
            s[k+1, i] += grad_l_i_new - grad_l_i_old
            # if (np.any(np.isnan(grad_l_i_new)) or np.any(np.isnan(grad_l_i_old))):
            #     print(f"NaN in gradient update for agent {i} at iteration {k+1}")
            
            total_grad += grad_l_i_old
            cost[k] += l_i

        for i in range(len(agents)):
            for j in range(len(targets)):
                norm_error[k, i, j] += np.linalg.norm(z[k, i, j] - targets[j])
        norm_grad_cost[k] = np.linalg.norm(total_grad)
    
    return z, cost, norm_grad_cost, norm_error