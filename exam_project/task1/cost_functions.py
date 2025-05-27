import numpy as np
np.random.seed(42)

def local_cost_function_task1(z, p_i, distances_i=None):
    p_i = np.asarray(p_i).reshape(1, -1)
    z = np.asarray(z)
    
    d = p_i.shape[1]
    np.random.seed(42)
    eig_max = np.random.randint(low=2)
    eigenvals = np.random.uniform(1, eig_max, d)
    Q = np.diag(eigenvals)
    b = np.random.uniform(0, 1)
    
    diffs = p_i - z
    
    local_cost = 0.0
    local_cost_grad = np.zeros(p_i.shape[1])
    
    for diff in diffs:
        local_cost += diff.T @ Q @ diff + b
        local_cost_grad -= 2 * Q @ diff
    return local_cost, local_cost_grad


def local_cost_function_task2(z, p_i, distances_i):
    # p_i:  position of the agent
    # z:    position of the targets
    # distances_i:  distances to the targets
    num_targets = len(distances_i)
    local_cost = 0
    local_cost_gradient = np.zeros((num_targets, len(p_i)))
    for target in range(num_targets):
        if np.isnan(distances_i[target]):
            local_cost += 0
            local_cost_gradient[target, :] = np.zeros_like(p_i)
        else:
            estimated_distance_squared = np.linalg.norm(z[target] - p_i)**2
            measured_distance_squared = distances_i[target]**2
            local_cost += (measured_distance_squared - estimated_distance_squared)**2 
            
            # Gradient evaluation
            local_cost_gradient[target, :] = 4 * (estimated_distance_squared - measured_distance_squared) * (z[target] - p_i)
    return local_cost, local_cost_gradient