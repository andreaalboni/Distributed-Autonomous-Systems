#!/usr/bin/env python3
import rclpy
import re
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from das_interfaces.msg import Plotting
import matplotlib.pyplot as plt

# example message structure:
# ---
# id: 0
# k: 138
# cost: 378.3528610545523
# grad_1:
# - 15.427788776285126
# - 18.441648045148415
# grad_phi: 1
# grad_2:
# - 36.46334546143352
# - 31.580665483394196
# ---


class Plotter(Node):
    def __init__(self):
        super().__init__(
            "plotter",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )
        
        self.max_iters = self.get_parameter("max_iters").value
        self.num_intruders = self.get_parameter("num_intruders").value
        self.d = self.get_parameter("d").value
        
        self.discovered_agents = set()
        self.cost_trajectory = np.zeros((self.num_intruders, self.max_iters))
        self.grad_1_trajectory = np.zeros((self.num_intruders, self.max_iters, self.d)) 
        self.grad_2_trajectory = np.zeros((self.num_intruders, self.max_iters, self.d))
        self.grad_phi_trajectory = np.zeros((self.num_intruders, self.max_iters, self.d))
        self.k_trajectory = np.zeros((self.num_intruders, self.max_iters))
        
        self.timer = self.create_timer(1.0, self.discover_agents)

    def discover_agents(self):
        topic_names_and_types = self.get_topic_names_and_types()
        pattern = r'/plotting_(\d+)'
        
        for topic_name, _ in topic_names_and_types:
            match = re.match(pattern, topic_name)
            if match and topic_name not in self.discovered_agents:
                agent_id = int(match.group(1))
                self.get_logger().info(f'Discovered new agent with ID {agent_id}')
                self.get_logger().info(f'Subscribing to {topic_name}')
                self.create_subscription(
                    Plotting,
                    topic_name,
                    lambda msg, id=agent_id: self.agent_callback(msg, id),
                    10
                )
                self.discovered_agents.add(topic_name)

    def agent_callback(self, msg, agent_id):
        self.cost_trajectory[agent_id][msg.k] = msg.cost
        self.grad_1_trajectory[agent_id][msg.k] = np.array(msg.grad_1)
        self.grad_2_trajectory[agent_id][msg.k] = np.array(msg.grad_2)
        self.grad_phi_trajectory[agent_id][msg.k] = np.array(msg.grad_phi)
        self.k_trajectory[agent_id] = msg.k

        for i in range(self.num_intruders):
            if self.k_trajectory[i][-1] >= self.max_iters - 1:
                self.plot_results()

    def plot_results(self):
        total_costs = []
        gradient_norms = []
        
        for iteration in range(self.max_iters):
            # Sum costs from all agents
            cost_sum = 0
            sum_grad_1 = 0
            grad_sum = np.zeros_like(self.grad_1_trajectory[0][0])
            
            for agent_id in range(self.num_intruders):
                cost_sum += self.cost_trajectory[agent_id][iteration]
                sum_grad_1 += self.grad_1_trajectory[agent_id][iteration]
                
                sum_grad_2 = np.zeros_like(self.grad_2_trajectory[0][0])
                for j in range(self.num_intruders):
                    sum_grad_2 += self.grad_2_trajectory[j][iteration]
                 
                grad_sum += sum_grad_1 + sum_grad_2 * self.grad_phi_trajectory[agent_id][iteration] / self.num_intruders     
            
            total_costs.append(cost_sum)
            gradient_norms.append(np.linalg.norm(grad_sum))

        fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
        
        ax1 = axes[0]
        ax1.semilogy(np.arange(1, self.max_iters-1), total_costs[1:-1], color='cornflowerblue')
        ax1.set_title('Total Cost')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        
        ax2 = axes[1]
        ax2.semilogy(np.arange(1, self.max_iters-1), gradient_norms[1:-1], color='indianred')
        ax2.set_title('Gradient Norm')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('$||∇\ell||$')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()
        
        self.get_logger().info(f'Final cost: {total_costs[-1]:.6f}')
        self.get_logger().info(f'Final gradient norm: {gradient_norms[-1]:.6f}')
        
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Plotter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()