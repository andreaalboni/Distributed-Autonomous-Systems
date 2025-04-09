import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils_formation_part import formation_vect_field, inter_distance_error, animation

NN = 6
L = 2
D = 2*L
distances = np.array(
    [
        [0, L, 0, D, 0, L],
        [L, 0, L, 0, D, 0],
        [0, L, 0, L, 0, D],
        [D, 0, L, 0, L, 0],
        [0, D, 0, L, 0, L],
        [L, 0, D, 0, L, 0],
    ]
)  # We need to have a rigid formation -> place 6 rigid bars between the 6 agents -> 6x6 matrix (regular exagon)

Adj = distances > 0     # No self loops, Connected, Binary Matrix
n_x = 2                 # 2D position
Tmax = 2
dt = 0.01               # Discretization step
horizon = np.arange(0, Tmax + dt, dt)   # Time horizon, uniform grid and step size

# Initial position of the agents
#Xinit = np.zeros((NN*n_x))
Xinit = np.random.uniform(low=-1, high=1, size=(NN*n_x))

res = solve_ivp(
    fun = lambda t, x: formation_vect_field(x, n_x, distances),     # Fun we want to integrate
    t_span = (0, Tmax),
    y0 = Xinit,
    t_eval = horizon,
    method = "RK45",    # Runge-Kutta method of order 5(4) (default method)
)   # Numerical integration of ODEs

XX = res.y.T

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 6))
fig.suptitle("Formation Control")

ax = axes[0]
ax.plot(horizon, XX)
ax.set_xlabel("Time")
ax.set_ylabel("Position")
ax.legend([f"Agent {i+1}" for i in range(NN)])
ax.grid()

ax = axes[1]
err = inter_distance_error(XX, NN, n_x, distances, horizon)
ax.semilogy(horizon, err)
ax.set_xlabel("Time")
ax.set_ylabel("Err")
ax.legend([f"Agent {i+1}" for i in range(NN)])
ax.grid()

ax = axes[2]
animation(
    XX,
    NN,
    n_x,
    horizon,
    Adj,
    ax,
)
plt.show()