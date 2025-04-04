import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def cost_fcn(z, Qlist, rlist):
    n = len(Qlist)
    val = []
    grad = []

    for i in range(n):
        Q_i = Qlist[i]
        r_i = rlist[i]

        val_i = 0.5 * z.T @ Q_i @ z + r_i.T @ z
        val.append(val_i)
        grad_i = Q_i @ z + r_i
        grad.append(grad_i)

    return val, grad


NN = 10
d = 5
Q = []
r = []

for i in range(NN):
    Q.append(np.diag(np.random.uniform(size=(d))))
    r.append(np.random.normal(size=(d)))

maxIters = 100
cost_incremental = np.zeros((maxIters))
z = np.zeros((maxIters, d))
alpha = 0.1

#batch_size = 1 -> Incremental Gradient Method
#batch_size = NN -> Gradient Method
batch_size = 5

for k in range(maxIters - 1):
    ik = np.random.choice(NN, batch_size, replace=False) # List of indices

    Q_minibatch = [Q[i] for i in ik]
    r_minibatch = [r[i] for i in ik]
    _, direction = cost_fcn(z[k], Q_minibatch, r_minibatch)
    direction = sum(direction)

    z[k + 1] = z[k] - alpha * direction
    cost, _ = cost_fcn(z[k], Q, r)
    cost_incremental[k] = sum(cost)


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(maxIters - 1), cost_incremental[:-1], label="Incremental cost")
ax.set_title("Incremental Gradient Method with Minibatch")
ax.set_xlabel("Iterations")
ax.legend()
plt.show()
