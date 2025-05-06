import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(0)


def create_graph(NN, p_er):
    ONES = np.ones((NN, NN))
    while 1:
        G = nx.erdos_renyi_graph(NN, p_er)
        Adj = nx.adjacency_matrix(G).toarray()
        test = np.linalg.matrix_power(Adj + np.eye(NN), NN)
        if np.all(test > 0):
            break

    A = Adj + np.eye(NN)

    while any(abs(np.sum(A, axis=1) - 1) > 1e-10):
        A = A / (A @ ONES)
        A = A / (ONES.T @ A)
        A = np.abs(A)

    return Adj, A


def cost_fcn(zz, QQ, rr):
    val = 0.5 * QQ * (zz**2) + rr * zz
    grad = QQ * zz + rr
    return val, grad


NN = 10
Q = []
r = []

for i in range(NN):
    Q.append(np.random.uniform())
    r.append(np.random.normal())

Qcentr = sum(Q)
rcentr = sum(r)
z_opt = -rcentr / Qcentr
cost_opt, _ = cost_fcn(z_opt, Qcentr, rcentr)
# print(cost_opt)
# quit()

maxIters = 200
cost = np.zeros((maxIters))
z = np.zeros((maxIters, NN))
s = np.zeros((maxIters, NN))
for i in range(NN):
    _, s[0, i] = cost_fcn(z[0, i], Q[i], r[i])

Adj, A = create_graph(NN, 0.5)

# print(np.sum(A, axis=0))
# print(np.sum(A, axis=1))
# quit()


alpha = 0.1

for k in range(maxIters - 1):
    for i in range(NN):
        z[k + 1, i] = A[i, i] * z[k, i]
        N_i = np.nonzero(Adj[i])[0]
        for j in N_i:
            z[k + 1, i] += A[i, j] * z[k, j]

        z[k + 1, i] -= alpha * s[k, i]

    for i in range(NN):
        s[k + 1, i] = A[i, i] * s[k, i]
        N_i = np.nonzero(Adj[i])[0]
        for j in N_i:
            s[k + 1, i] += A[i, j] * s[k, j]

        _, grad_ell_i_new = cost_fcn(z[k + 1, i], Q[i], r[i])
        _, grad_ell_i_old = cost_fcn(z[k, i], Q[i], r[i])
        s[k + 1, i] += grad_ell_i_new - grad_ell_i_old

        ell_i, _ = cost_fcn(z[k, i], Q[i], r[i])
        cost[k] += ell_i

z_avg = np.mean(z, axis=1)

fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
ax = axes[0]
ax.semilogy(np.arange(maxIters - 1), np.abs(cost[:-1] - cost_opt))
# ax.plot(np.arange(maxIters - 1), cost[:-1])
# ax.plot(np.arange(maxIters - 1), cost_opt * np.ones((maxIters - 1)), "r--")

ax = axes[1]
for i in range(NN):
    ax.semilogy(np.arange(maxIters), np.abs(z[:, i] - z_avg))
    # ax.plot(np.arange(maxIters), z[:, i] - z_avg)
    
print(z[-1, :])
plt.show()