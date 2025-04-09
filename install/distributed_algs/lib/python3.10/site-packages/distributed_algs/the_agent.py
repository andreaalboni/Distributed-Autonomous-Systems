import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray as MsgFloat
from time import sleep

class Agent(Node):
    def __init__(self):
        super().__init__(
            "parametric_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("neighbors").value
        self.inital_value = self.get_parameter("xzero").value
        self.k = 0
        self.x_i = self.inital_value

        print(f"I am agent: {self.agent_id:d}")
        print(f"My neighbors are: {self.neighbors}")
        print(f"My initial value is: {self.inital_value:d}")

        # Create a publisher
        self.publisher_ = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)
        self.timer_ = self.create_timer(1.0, self.timer_callback) # Check timer to see if we can publish

        # Create a subscriber
        for j in self.neighbors:
            self.create_subscription(MsgFloat, f"/topic_{j}", self.listener_callback, 10)
        self.received_data = {j: [] for j in self.neighbors}
        
    def listener_callback(self, msg):
        j = int(msg.data[0]) # The agent that sent the message
        msg_j = msg.data[1:] # The message
        self.received_data[j].append(msg_j)

    def timer_callback(self):
        if self.k == 0:
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.k), float(self.inital_value)] # Compose the msg: ID, k value and NODE value
            self.publisher_.publish(msg)
            self.x_i = self.inital_value
            self.k += 1
        else:
            time_stamp = [self.k-1 == self.received_data[j][0][0] for j in self.neighbors]
            # self.received_data[j][0][0] is the k value of the message received from agent j since we dropped from the msg the agent ID in listener_callback
            if all(time_stamp):
                # Let's compute the maximum value of all the received values
                temp = self.x_i
                for j in self.neighbors:
                    _, x_i = self.received_data[j].pop(0) # Remove the first entry from the corresponding list
                    temp = max(temp, x_i) # Maximum consensus
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.k), float(temp)]
            self.publisher_.publish(msg)
            print(f"Agent {self.agent_id} at iteration {self.k} has value {temp}")
            self.k += 1

            if self.k > 10:
                #print("Maximum iterations reached. Terminating.")
                #self.destroy_node()
                raise SystemExit

def main():
    rclpy.init()
    anAgent = Agent()
    sleep(1)
    try:
        rclpy.spin(anAgent)
    except SystemExit:
        print("Maximum iterations reached. Terminating.")
    anAgent.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
