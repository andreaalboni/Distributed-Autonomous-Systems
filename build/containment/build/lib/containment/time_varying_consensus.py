import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

#######################################################

# Doesn't work properly

#######################################################

NN = 10
maxiters = 100

X = np.zeros((maxiters, NN))
Xinit = np.random.uniform(size=(NN))
X[0] = Xinit

def create_random_graph(N, p_ER=0.5):
    G = nx.erdos_renyi_graph(N, p_ER)
    # Create a random weight matrix
    Adj = nx.adjacency_matrix(G).toarray()
    weighted_Adj = Adj + np.eye(N)
    # Our goal is to make the weighted_Adj matrix doubly stochastic
    while any(abs(np.sum(weighted_Adj, axis=0) - 1) > 1e-8) and any(abs(np.sum(weighted_Adj, axis=1) - 1) > 1e-8):
        weighted_Adj = weighted_Adj / np.sum(weighted_Adj, axis=1, keepdims=True)
        weighted_Adj = weighted_Adj / np.sum(weighted_Adj, axis=0, keepdims=True)
        weighted_Adj = np.abs(weighted_Adj)
    return G, weighted_Adj

# Create a random graph
G, weighted_Adj = create_random_graph(NN, 0.5) 

print(weighted_Adj @ np.ones((NN)))
print(weighted_Adj.T @ np.ones((NN)))

for k in range(maxiters-1):
    # Generate at every iteration a different graph
    G, weighted_Adj = create_random_graph(NN, 0.5)
    X[k+1] = weighted_Adj @ X[k] 

fig,ax = plt.subplots()
ax.plot(np.arange(maxiters), X)
plt.legend([f'Node {i}' for i in range(NN)])
plt.title('Time-varying consensus')
ax.grid()
plt.show()