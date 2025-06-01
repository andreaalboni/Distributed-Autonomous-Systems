#!/usr/bin/env python3
import rclpy
import re
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from das_interfaces.msg import Plotting
import matplotlib.pyplot as plt


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
                self.cost_trajectory[agent_id] = []
                self.discovered_agents.add(topic_name)

    def agent_callback(self, msg, agent_id):
        
        self.agent_data[agent_id] = {
            'cost': msg.cost,
            'grad_1': np.array(msg.grad_1),
            'grad_2': np.array(msg.grad_2),
            'grad_phi': np.array(msg.grad_phi),
            'iteration': msg.k   
        }
        
        self.cost_trajectory[agent_id].append(self.agent_data[agent_id]['cost'])

        if self.check_completion():
            self.plot_results()

    def check_completion(self):
        if not self.agent_data:
            return False
            
        for agent_id in self.agent_data:
            if self.max_iters != self.agent_data[agent_id]['iteration']:
                return False
        return True

    def plot_results(self):
        iterations = range(1, self.max_iters + 1)
        total_costs = []
        gradient_norms = []
        num_agents = len(self.agent_data)
        
        for iteration in iterations:
            # Sum costs from all agents
            cost_sum = 0.0
            grad_sum = np.zeros_like(self.agent_data[list(self.agent_data.keys())[0]][1]['grad_1'])
            
            for agent_id in self.agent_data:
                data = self.agent_data[agent_id][iteration]
                cost_sum += data['cost']
                
                # grad_1 + grad_2 * grad_phi / num_agents
                agent_grad = data['grad_1'] + data['grad_2'] * data['grad_phi'] / num_agents
                grad_sum += agent_grad
            
            total_costs.append(cost_sum)
            gradient_norms.append(np.linalg.norm(grad_sum))
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Cost plot
        ax1.plot(iterations, total_costs, 'b-', linewidth=2)
        ax1.set_title('Total Cost')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.grid(True)
        ax1.set_yscale('log')
        
        # Gradient norm plot
        ax2.plot(iterations, gradient_norms, 'r-', linewidth=2)
        ax2.set_title('Gradient Norm')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('||∇f||')
        ax2.grid(True)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        self.get_logger().info(f'Final cost: {total_costs[-1]:.6f}')
        self.get_logger().info(f'Final gradient norm: {gradient_norms[-1]:.6f}')

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