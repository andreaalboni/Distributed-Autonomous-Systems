import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from time import sleep


def formation_vect_field(dt, x_ii, N_ii, data, dist):
    """
    dt    : discretization step
    x_i   : state pf agent i
    N_ii  : list of neighbors
    data  : state of the neighbors
    dist  : coefficient to implement the formation control law
    """
    x_i_dot = np.zeros(x_ii.shape)

    for jj in N_ii:
        x_jj = np.array(data[jj].pop(0)[1:])
        dV_ij = ((x_ii - x_jj).T @ (x_ii - x_jj) - dist[jj] ** 2) * (x_ii - x_jj)
        x_i_dot += -dV_ij

    # Forward Euler discretization
    x_ii += dt * x_i_dot

    return x_ii


def writer(file_name, string):
    """
    inner function for logging
    """
    file = open(file_name, "a")  # "a" stands for "append"
    file.write(string)
    file.close()


class Agent(Node):
    def __init__(self):
        super().__init__(
            "formation_control_ros_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters from the launch file
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("neighbors").value
        x_i = self.get_parameter("xzero").value

        dist = self.get_parameter("dist").value
        self.dist_ii = dist  # np.array(dist)
        self.x_i = np.array(x_i)

        self.maxIters = self.get_parameter("maxT").value
        communication_time = self.get_parameter("communication_time").value
        self.DeltaT = communication_time / 10
        self.k = 0

        print(f"I am agent: {self.agent_id:d}")

        # Subscribe to topic_j
        for j in self.neighbors:
            self.create_subscription(
                MsgFloat,
                f"/topic_{j}",
                self.listener_callback,
                10,
            )
        # initialize an empty dictionary to store the messages received from each neighbor j
        self.received_data = {j: [] for j in self.neighbors}

        # Create topic_i
        self.publisher = self.create_publisher(
            MsgFloat,
            f"/topic_{self.agent_id}",
            10,
        )
        self.timer = self.create_timer(communication_time, self.timer_callback)

        print(f"Agent {self.agent_id}: setup completed!")

    def listener_callback(self, msg):
        """
        When the new msg arrives, move it into the buffer
        """
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.received_data[j].append(msg_j)

        return None

    def timer_callback(self):
        """
        When all the msg have arrived, do the update
        """
        msg = MsgFloat()

        if self.k == 0:  # First iteration: the publisher starts
            msg.data = [float(self.agent_id), float(self.k), *self.x_i]
            self.publisher.publish(msg)
            self.k += 1

            print(f"Iter: {self.k:3d}   x_{self.agent_id:d}: {self.x_i}")

        else:
            all_received = False
            if all(len(self.received_data[j]) > 0 for j in self.neighbors):
                all_received = all(
                    self.k - 1 == self.received_data[j][0][0] for j in self.neighbors
                )

            if all_received:
                self.x_i = formation_vect_field(
                    self.DeltaT,
                    self.x_i,
                    self.neighbors,
                    self.received_data,
                    self.dist_ii,
                )

                # publish the updated message
                msg.data = [float(self.agent_id), float(self.k), *self.x_i]
                self.publisher.publish(msg)

                print(f"Iter: {self.k:3d}   x_{self.agent_id:d}: {self.x_i}")

                # update iteration counter
                self.k += 1

                # Stop if MAXITERS is exceeded
                if self.k > self.maxIters:
                    print("\nMax iters reached")
                    sleep(3)  # [seconds]
                    raise SystemExit


def main(args=None):
    rclpy.init(args=args)

    aFormContrAgent = Agent()
    aFormContrAgent.get_logger().info(
        f"Agent {aFormContrAgent.agent_id:d}: Waiting for sync..."
    )
    sleep(1)
    aFormContrAgent.get_logger().info("GO!")

    try:
        rclpy.spin(aFormContrAgent)
    except SystemExit:  # <--- process the exception
        rclpy.logging.get_logger("Quitting").info("Done")
    finally:
        aFormContrAgent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
