import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(0)

def create_graph(N, p_ER=0.5):
    while 1:
        G = nx.erdos_renyi_graph(N, p_ER)
        Adj = nx.adjacency_matrix(G).toarray()
        # Check whether it is strongly connected
        test = np.linalg.matrix_power(Adj + np.eye(NN), NN)
        if np.all(test > 0):
            break
    # Create adjacency matrix
    weighted_Adj = Adj + np.eye(N)
    # Our goal is to make the weighted_Adj matrix doubly stochastic
    while any(abs(np.sum(weighted_Adj, axis=1) - 1) > 1e-10):
        weighted_Adj = weighted_Adj / np.sum(weighted_Adj, axis=1, keepdims=True)
        weighted_Adj = weighted_Adj / np.sum(weighted_Adj, axis=0, keepdims=True)
    return G, weighted_Adj

def cost_fcn(zz, QQ, rr):
    val = 0.5 * QQ * zz**2 + rr * zz
    grad = QQ * zz + rr
    return val, grad

NN = 10
d = 5
Q = []
r = []

for i in range(NN):
    Q.append(np.diag(np.random.uniform(size=(d))))
    r.append(np.random.normal(size=(d)))

Qcentr = sum(Q)
rcentr = sum(r)
z_opt = - rcentr / Qcentr
ell_opt, _ = cost_fcn(z_opt, Qcentr, rcentr)
print("Optimal value:", ell_opt)
print("Optimal solution:", z_opt)

maxIters = 100
cost = np.zeros((maxIters))
z = np.zeros((maxIters, d))
alpha_init = 0.1

Adj, A = create_graph(NN, 0.5)

for k in range(maxIters - 1):
    alpha = alpha_init / (k + 1)
    for i in range(NN):
        z[k+1, i] = A[i, i] * z[k, i]
        N_i = np.nonzero(Adj[i])[0]

        for j in N_i:
            z[k+1, i] += A[i, j] * z[k, j]

        _, grad_ell_i = cost_fcn(z[k+1, i], Q[i], r[i])
        z[k+1, i] -= alpha * grad_ell_i

        ell_i, _ = cost_fcn(z[k, i], Q[i], r[i])
        cost[k] += ell_i

z_avg = np.mean(z, axis=1)

fig, axes = plt.subplots(figsize=(8, 6), nrows=2, ncols=1)
ax = axes[0]
ax.semilogy(np.arange(maxIters - 1), cost[:-1])
ax.axhline(ell_opt, color='r', linestyle='--', label='Optimal Value')
ax.set_title("Distributed Gradient Method")
ax.set_xlabel("Iterations")
ax.legend()

ax = axes[1]
for i in NN:
    ax.plot(np.arange(maxIters), z[:, i] - z_avg, label=f'Node {i}')
ax.set_xlabel("Iterations")
ax.set_title("Distributed Gradient Method - Node Values")
plt.show()