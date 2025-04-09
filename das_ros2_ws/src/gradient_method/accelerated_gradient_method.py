import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def cost_fcn(zz, QQ, rr):
    val = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    return val, grad


d = 5
Q = np.diag(np.random.uniform(size=(d)))
r = np.random.normal(size=(d))


z_opt = -np.linalg.inv(Q) @ r
ell_opt, _ = cost_fcn(z_opt, Q, r)

maxIters = 1000
zinit = np.random.normal(size=(d))

cost_GM = np.zeros((maxIters))
z = np.zeros((maxIters, d))
xi = np.zeros((maxIters, d))
v = np.zeros((maxIters, d))
y = np.zeros((maxIters, d))

z[0] = zinit
xi[0] = zinit
alpha3 = 0
alpha2 = 0.1
alpha1 = 0

for k in range(maxIters - 1):
    v[k] = (1 + alpha3) * z[k] - alpha3 * xi[k]
    _, grad = cost_fcn(v[k], Q, r)
    y[k] = -alpha2 * grad

    z[k + 1] = (1 + alpha1) * z[k] - alpha1 * xi[k] + y[k]
    xi[k + 1] = z[k]

    cost_GM[k], _ = cost_fcn(v[k], Q, r)

###
cost_HB = np.zeros((maxIters))
z = np.zeros((maxIters, d))
xi = np.zeros((maxIters, d))
v = np.zeros((maxIters, d))
y = np.zeros((maxIters, d))

z[0] = zinit
xi[0] = zinit
alpha3 = 0.1
alpha2 = 0.1
alpha1 = 0.1

for k in range(maxIters - 1):
    v[k] = (1 + alpha3) * z[k] - alpha3 * xi[k]
    _, grad = cost_fcn(v[k], Q, r)
    y[k] = -alpha2 * grad

    z[k + 1] = (1 + alpha1) * z[k] - alpha1 * xi[k] + y[k]
    xi[k + 1] = z[k]

    cost_HB[k], _ = cost_fcn(v[k], Q, r)

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(np.arange(maxIters - 1), np.abs(cost_GM[:-1] - ell_opt), label="GM")
ax.semilogy(np.arange(maxIters - 1), np.abs(cost_HB[:-1] - ell_opt), label="HB")
ax.set_title("Accelerated Gradient Method (Heavy Ball) vs Gradient Method")
ax.set_xlabel("Iterations")
ax.legend()
plt.show()
