import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def animation(XX, NN, n_x, n_le, horizon, dt):
    TT = len(horizon)
    n_fo = NN - n_le

    for tt in range(0, TT - dt, dt):
        xx_tt = XX[tt]
        plt.clf()

        # Plot trajectories
        if tt > dt and tt < TT - 1:
            plt.plot(
                XX[tt - dt : tt + 1, 0 : n_x * n_fo : n_x],
                XX[tt - dt : tt + 1, 1 : n_x * n_fo : n_x],
                linewidth=2,
                color="tab:blue",
            )
            plt.plot(
                XX[n_x * (NN - n_le) : n_x * NN : n_x, tt - dt : tt + 1],
                XX[n_x * (NN - n_le) + 1 : n_x * NN : n_x, tt - dt : tt + 1],
                linewidth=3,
                color="tab:red",
            )

        # Plot convex hull
        leaders_pos = np.reshape(XX[tt, n_x * n_fo : n_x * NN], (n_le, n_x))

        hull = ConvexHull(leaders_pos)
        plt.fill(
            leaders_pos[hull.vertices, 0],
            leaders_pos[hull.vertices, 1],
            "darkred",
            alpha=0.3,
        )
        vertices = np.hstack(
            (hull.vertices, hull.vertices[0])
        )  # add the first in the last position to draw the last line
        plt.plot(
            leaders_pos[vertices, 0],
            leaders_pos[vertices, 1],
            linewidth=2,
            color="darkred",
            alpha=0.7,
        )

        # Plot agent position
        for ii in range(NN):
            index_ii = ii * n_x + np.array(range(n_x))
            p_prev = xx_tt[index_ii]
            agent_color = "blue" if ii < NN - n_le else "red"
            plt.plot(
                p_prev[0],
                p_prev[1],
                marker="o",
                markersize=10,
                fillstyle="full",
                color=agent_color,
            )

        x_lim = (
            np.min(leaders_pos[hull.vertices, 0]) - 1,
            np.max(leaders_pos[hull.vertices, 0]) + 1,
        )
        y_lim = (
            np.min(leaders_pos[hull.vertices, 1]) - 1,
            np.max(leaders_pos[hull.vertices, 1]) + 1,
        )
        plt.title(
            f"Agents Position - Simulation time = {horizon[tt + dt]:.2f} s of {horizon[-1]:.2f} s"
        )
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xlabel("first component")
        plt.ylabel("second component")
        plt.show(block=False)
        plt.pause(0.1)


def plot_trajectory(XX, NN, n_x, n_le, horizon, dim):
    n_fo = NN - n_le

    followers_x_trajectory = XX[:, range(dim, n_x * n_fo, n_x)]
    plt.plot(horizon, followers_x_trajectory, color="tab:blue")

    followers_x_trajectory = XX[:, range(n_x * n_fo + dim, n_x * NN, n_x)]
    plt.plot(horizon, followers_x_trajectory, color="tab:red")

    if dim == 0:
        plt.title("Evolution of the local estimates: x-axis")
    elif dim == 1:
        plt.title("Evolution of the local estimates: y-axis")
    elif dim == 2:
        plt.title("Evolution of the local estimates: z-axis")
    else:
        plt.title(f"Evolution of the local estimates: {dim}-th axis")
