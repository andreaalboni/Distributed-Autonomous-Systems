#
# Utils for Formation control
# Ivano Notarnicola
# Bologna, 08/04/2025
#
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
#
# System dynamics
#
def formation_vect_field(xt, n_x, distances):
    N = distances.shape[0]
    xt_reshaped = xt.reshape((N, n_x))
    dxt = np.zeros_like(xt_reshaped)    

    for i in range(N):
        xi = xt_reshaped[i]
        N_i = np.where(distances[i] > 0)[0]    # Indexes of non-zeros elements in row i -> neighbors
        for j in N_i:
            xj = xt_reshaped[j]
            dxt[i] -= (np.linalg.norm(xi - xj)**2 - distances[i, j]**2) * (xi - xj)
    return dxt.reshape(-1)


def inter_distance_error(XX, NN, n_x, distances, horizon):
    err = np.zeros((len(horizon), NN, NN))

    for tt in range(len(horizon)):
        xt = XX[tt].reshape((NN, n_x))
        for i in range(NN):
            N_i = np.where(distances > 0)[0]
            x_i_t = xt[i]
            for j in N_i:
                x_j_t = xt[j]
                err[tt,i,j] = np.linalg.norm(x_i_t - x_j_t) - distances[i, j]
    return err.reshape((len(horizon),-1))


def animation(XX, NN, n_x, horizon, Adj, ax, wait_time=0.05):
    axes_lim = (np.min(XX) - 1, np.max(XX) + 1)

    for tt in range(len(horizon)):
        # plot 2d-trajectories
        ax.plot(
            XX[:, 0 : n_x * NN : n_x],
            XX[:, 1 : n_x * NN : n_x],
            color="tab:gray",
            linestyle="dashed",
            alpha=0.5,
        )

        # plot 2d-formation
        xx_tt = XX[tt].reshape((NN, n_x))

        for ii in range(NN):
            p_prev = xx_tt[ii]

            ax.plot(
                p_prev[0],
                p_prev[1],
                marker="o",
                markersize=15,
                fillstyle="full",
                color="tab:red",
            )

            for jj in range(NN):
                if Adj[ii, jj] & (jj > ii):
                    p_curr = xx_tt[jj]
                    ax.plot(
                        [p_prev[0], p_curr[0]],
                        [p_prev[1], p_curr[1]],
                        linewidth=1,
                        color="steelblue",
                        linestyle="solid",
                    )

        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.axis("equal")
        ax.set_xlabel("first component")
        ax.set_ylabel("second component")
        ax.set_title(f"Simulation time = {horizon[tt]:.2f} s")
        plt.show(block=False)
        plt.pause(wait_time)
        ax.cla()
