import numpy as np
np.random.seed(42)

def local_cost_function_task1(z, p_i, distances_i=None):
    """
    Compute the local cost and its gradient for task 1.1

    Args:
        z (array-like): Input variable(s) for the cost function.
        p_i (array-like): Reference point(s) for the cost function.
        distances_i (array-like, optional): Not used.

    Returns:
        tuple: Local cost (float) and its gradient (ndarray).
    """
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
    """
    Computes the local cost and its gradient for an agent based on estimated and measured distances to multiple targets.

    Args:
        z (np.ndarray): Array of target positions, shape (num_targets, dim).
        p_i (np.ndarray): Position of the agent, shape (dim,).
        distances_i (np.ndarray): Measured distances from the agent to each target, shape (num_targets,).

    Returns:
        tuple: 
            local_cost (float): The total local cost for the agent.
            local_cost_gradient (np.ndarray): Gradient of the local cost with respect to agent position, shape (num_targets, dim).
    """
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