import numpy as np
from utils import *
from config import PARAMETERS
import matplotlib.pyplot as plt


def aggregative_tracking_method(max_iters=2000, alpha=0.0001): 
    intruders, agents = generate_agents_and_intruders()
    # visualize_world(agents, intruders)
    graph_type = 'cycle'
    _, adj, A = generate_graph(len(agents), type=graph_type)
    # visualize_graph(G)
    
    cost = np.zeros((max_iters))
    z = np.zeros((max_iters, len(agents), len(agents[0])))
    s = np.zeros((max_iters, len(agents), len(agents[0])))
    v = np.zeros((max_iters, len(agents), len(agents[0])))
    gamma = 15 * np.ones(len(agents))
    gamma_bar = 3 * np.ones(len(agents))

    r_0 = compute_r_0(intruders)
    sigma = compute_agents_barycenter(agents)

    # Initialization
    z[0] = agents
    s[0] = agents   # phi_i(z_i) = z_i: regular barycenter
    for i in range(len(agents)):
        _, _, v[0, i] = local_cost_function(z[0, i], intruders[i], s[0, i], r_0, gamma[i], gamma_bar[i])

    # Ch 8 p 7/11 
    for k in range(max_iters - 1):
        for i in range(len(agents)):
            _, grad_1_l_i, _ = local_cost_function(z[k, i], intruders[i], s[k, i], r_0, gamma[i], gamma_bar[i])    # in z_i^{k}, s_i^{k}
            _, grad_phi_i = local_phi_function(z[k, i])
            z[k+1, i] = z[k, i] - alpha * ( grad_1_l_i + grad_phi_i * v[k, i] )
        
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
            _, _, grad_2_l_i_new = local_cost_function(z[k+1, i], intruders[i], s[k+1, i], r_0, gamma[i], gamma_bar[i])    # in z_i^{k+1}, s_i^{k+1}
            l_i, _, grad_2_l_i_old = local_cost_function(z[k, i], intruders[i], s[k, i], r_0, gamma[i], gamma_bar[i])    # in z_i^{k}, s_i^{k}
            v[k+1, i] += grad_2_l_i_new - grad_2_l_i_old
            
            cost[k] += l_i
    
    fig, ax = plt.subplots(figsize=(8, 6), nrows=1, ncols=1)
    ax.semilogy(np.arange(max_iters-1), cost[:-1], color='violet')
    ax.set_title('Cost vs Iteration')
    ax.set_xlabel('Iteration')
    plt.show()
    
    animate_world_evolution(intruders, z, s, sigma, r_0)

def main(): 
    aggregative_tracking_method()

if __name__ == "__main__":
    main()