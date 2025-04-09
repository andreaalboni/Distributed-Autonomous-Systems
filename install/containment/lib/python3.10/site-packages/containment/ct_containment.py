import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from functions import plot_trajectory, animation
import control

np.random.seed(5)

ANIMATION = True
PLOT_TRAJECTORIES = True
INT_ACT = True

Tmax = 5.0  # simulation time
n_x = 2  # dimension of x_i

NN = 15  # number of agents
n_le = 5  # number of leaders

leaders_velocity = 1  # velocity of the leaders

I_NN = np.eye(NN)
I_nx = np.eye(n_x)

# Network generation
p_ER = 0.7  # edge probability
while 1:
    graph_ER = nx.binomial_graph(NN, p_ER)
    Adj = nx.adjacency_matrix(graph_ER).toarray()
    test = np.linalg.matrix_power((I_NN + Adj), NN)

    if np.all(test > 0):
        print("the graph is connected\n")
        break
    else:
        print("the graph is NOT connected\n")

deg_vec = np.sum(Adj, axis=0)  # degree vector
deg_matrix = np.diag(deg_vec)  # degree matrix
Laplacian = deg_matrix - Adj.T  # Laplacian matrix

# Followers dynamics
n_fo = NN - n_le
L_f = Laplacian[0:n_fo, 0:n_fo]  # submatrix
L_fl = Laplacian[0:n_fo, n_fo:]

# Concatenate both Leaders and Followers dynamics
A_1d = -np.block(
    [
        [L_f, L_fl],
        [np.zeros((n_le, NN))],
    ]
)

# Consider only the leaders in the input B matrix
B_1d = np.block(
    [
        [np.zeros((n_fo, n_le))],
        [np.eye(n_le)],
    ]
)

# Initialize the agents' state
# XX_init = np.block(
#     [
#         [np.zeros((n_x * n_f, 1))],
#         [np.ones((n_x * n_leaders, 1))],
#     ],
# )

XX_init = 10 * np.random.rand(n_x * NN)

X_0 = XX_init
A = np.kron(A_1d, I_nx)
B = np.kron(B_1d, I_nx)
C = np.kron(np.eye(NN), I_nx)  # to comply with StateSpace syntax

################################################
#
# Followers with an integral action
#
if INT_ACT:
    gain_int = 10  # 10, 20, 30
    K_int = -np.block(
        [
            [gain_int * np.eye(n_fo)],
            [np.zeros((n_le, n_fo))],
        ]
    )

    # Setup the extended dynamics
    A_1d_ext = np.block(
        [
            [A_1d, K_int],
            [L_f, L_fl, np.zeros((n_fo, n_fo))],
        ]
    )

    B_1d_ext = np.block(
        [
            [B_1d],
            [np.zeros((n_fo, n_le))],
        ]
    )

    XX_init_ext = np.concatenate([XX_init, np.zeros((n_x * n_fo))])

    X_0 = XX_init_ext
    A = np.kron(A_1d_ext, I_nx)
    B = np.kron(B_1d_ext, I_nx)
    C = np.kron(
        np.block([np.eye(NN), np.zeros((NN, n_fo))]),
        I_nx,
    )

###############################################################################
#
# CONTAINMENT CT Dynamics
#
dt = 0.01  # Sampling time
horizon = np.arange(0.0, Tmax + dt, dt)

XX = np.zeros((len(horizon), A.shape[1]))
XX[0] = X_0

# Leaders' input
UU = leaders_velocity * np.ones((len(horizon), n_x * n_le))

sys = control.StateSpace(A, B, C, 0)  # dx = - L x + [0; I] u, y = [I 0] x
(horizon, XX) = control.forced_response(sys, X0=X_0, U=UU.T, T=horizon)
XX = XX.T

###############################################################################
#
# Visualization
#
if PLOT_TRAJECTORIES:
    for dim in range(n_x):
        plt.figure(dim)
        plt.xlabel("$k$")
        plt.ylabel(f"$x_{{i,{dim}}}^k$")
        plt.grid()
        plot_trajectory(XX, NN, n_x, n_le, horizon, dim)


if ANIMATION and n_x == 2:
    plt.figure("Animation")
    animation(XX, NN, n_x, n_le, horizon, dt=10)

if PLOT_TRAJECTORIES or ANIMATION:
    plt.show()
