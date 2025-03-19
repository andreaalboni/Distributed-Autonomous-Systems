import numpy as np
from matplotlib import pyplot as plt

d = 4
Q = np.diag(np.random.uniform(size=d))
r = np.random.uniform(size=d)

maxiter = 1000
z = np.zeros((maxiter, d))
alpha = 1e-2

# Code a QP and solve it using gradient method
def cost_function(z, Q, r):
    cost = 0.5 * z.T @ Q @ z + r.T @ z 
    grad = Q @ z + r
    return cost, grad

for k in range(maxiter-1):
    _,grad = cost_function(z[k], Q, r)
    z[k+1] = z[k] - alpha * grad

fig,ax = plt.subplots()
ax.plot(z)
plt.show()