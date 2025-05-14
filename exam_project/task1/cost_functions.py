import numpy as np

def local_cost_function_task1(z_ref, z, distances=None):
    Q = np.eye(len(z)) * 10
    diff = z_ref - z
    cost = np.linalg.norm(diff.T @ Q @ diff, 'fro')**2
    grad = 2 * Q @ diff
    return cost, grad


def local_cost_function_task2(z_ref, z, distances_i):
    # p_i:  position of the agent
    # z:    position of the target
    # distances_i:  distances to the targets
    num_targets = len(distances_i)
    local_cost = 0
    local_cost_gradient = np.zeros((num_targets, len(z)))
    for target in range(num_targets):
        if np.isnan(distances_i[target]):
            local_cost += 0
            local_cost_gradient[target, :] = np.nan * np.zeros_like(z)
        else:
            estimated_distance_squared = np.linalg.norm(z_ref[target] - z)**2
            measured_distance_squared = distances_i[target]**2
            local_cost += (measured_distance_squared - estimated_distance_squared)**2 
            
            # Gradient evaluation
            #print(4 * (estimated_distance_squared - measured_distance_squared) * (z[target] - p_i))
            local_cost_gradient[target, :] = 4 * (estimated_distance_squared - measured_distance_squared) * (z_ref[target] - z)
    return local_cost, local_cost_gradient