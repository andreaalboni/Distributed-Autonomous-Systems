import numpy as np

def local_cost_function(agent_i, intruder_i, sigma, r_0, gamma_i, gamma_bar_i, gamma_hat_i):
    '''
    Cost function a lezione:
    l_i = gamma_i * (norma della distanza agente - intruder)**2 +                  --> garantisce agente vada sull'intruder
            gamma_bar_i * (norma della distanza agente - sigma)**2 +               --> garantisce formazione sia il più possibile compatta
            norma della distanza r_0 - intruder (che io farei che sia il baricentro degli intruder)
    sigma(z) = sum (phi_i(z_i)) / N     --> calcolo sigma
    phi_i(z_i): per calcolo sigma normale: = z_i, per weighted barycenter: = weight_i * z_i
    
    # agents has shape (2,)
    # intruder has shape (2,)
    # sigma has shape (2,)
    
    # grad_1 is the gradient of the cost function with respect to the intruder
    # grad_2 is the gradient of the cost function with respect to the barycenter
    '''
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
    phi_i = agent_i
    grad_phi_i = 1
    return phi_i, grad_phi_i