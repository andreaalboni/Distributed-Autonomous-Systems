import numpy as np

def local_cost_function(agent_i, intruder_i, sigma, r_0, gamma_i, gamma_bar_i, gamma_hat_i):
    """
    Computes the local cost function and its gradients for an agent in a multi-agent system.
    The cost function is defined as:
        l_i = gamma_i * ||agent_i - intruder_i||^2 
            + gamma_bar_i * ||agent_i - sigma||^2
            + gamma_hat_i * ||intruder_i - r_0||^2
    Where:
        - The first term encourages the agent to approach the intruder.
        - The second term encourages the agent to stay close to the formation barycenter (sigma).
        - The third term encourages the intruder to stay close to the barycenter of the intruders (r_0).
    Args:
        agent_i (np.ndarray): Position of the agent, shape (2,).
        intruder_i (np.ndarray): Position of the intruder, shape (2,).
        sigma (np.ndarray): Barycenter of the agents (formation center), shape (2,).
        r_0 (np.ndarray): Barycenter of the intruders, shape (2,).
        gamma_i (float): Weight for the agent-intruder distance term.
        gamma_bar_i (float): Weight for the agent-formation barycenter distance term.
        gamma_hat_i (float): Weight for the intruder-barycenter distance term.
    Returns:
        tuple:
            local_cost (float): Value of the local cost function.
            grad_1 (np.ndarray): Gradient of the cost function with respect to the agent position, shape (2,).
            grad_2 (np.ndarray): Gradient of the cost function with respect to the barycenter (sigma), shape (2,).
    """
    agent_to_intruder = np.linalg.norm(agent_i - intruder_i)**2  
    agent_to_sigma = np.linalg.norm(agent_i - sigma)**2
    intruder_to_r_0 = np.linalg.norm(intruder_i - r_0)**2
    local_cost = gamma_i * agent_to_intruder + gamma_bar_i * agent_to_sigma + gamma_hat_i * intruder_to_r_0
    # grad_1 is the gradient of the cost function with respect to the agent
    grad_1 = 2 * (gamma_i * (agent_i - intruder_i) + gamma_bar_i * (agent_i - sigma))
    # grad_2 is the gradient of the cost function with respect to sigma
    grad_2 = - 2 * gamma_bar_i * (agent_i - sigma)
    return local_cost, grad_1, grad_2

def local_phi_function(agent_i):
    """Computes the local phi function and its gradient for a given agent."""
    phi_i = agent_i
    grad_phi_i = np.ones(agent_i.shape)
    return phi_i, grad_phi_i