import numpy as np
import torch
import torch.nn as nn
import hashlib
import heapq
import random

# Core Math Module
class PyZeroMathTorch:
    def __init__(self, epsilon_0=1e-8, scaling_factor=0.1):
        self.epsilon_0 = epsilon_0
        self.scaling_factor = scaling_factor
        self.current_epsilon = epsilon_0

    def f0z_stabilize(self, x, system_size=None):
        if system_size is not None:
            self.current_epsilon = self.epsilon_0 * self.scaling_factor * torch.log(torch.tensor(float(system_size)))
            self.current_epsilon = max(self.current_epsilon, self.epsilon_0)
        mask = torch.abs(x) < self.current_epsilon
        return torch.where(mask, self.current_epsilon * torch.sign(x), x)

    def adjust_epsilon(self, task_complexity):
        self.current_epsilon = 1e-8 * max(1, task_complexity / 5)

# Base Agent Class
class DFSNAgent:
    def __init__(self, name, math_module=None):
        self.name = name
        self.flow_state = 'idle'
        self.performance_history = []
        self.math_module = math_module or PyZeroMathTorch()
        self.optimizer = FlowStateOptimizer()
        self.engagement_state = 0
        self.cpu_allocation = 0
        self.memory_allocation = 0
        self.aiw = AgentIterativeWorkflow(self)
        self.peers = []
        self.domain_specialty = "general"
        self.overdrive = False

    def adjust_flow_state(self, task_complexity, performance):
        stability = self.compute_stability()
        state = [task_complexity, performance, stability]
        action = self.optimizer.predict(state)
        self.flow_state = 'flow' if action == 1 else 'idle'
        reward = performance - 0.1 * task_complexity
        self.optimizer.update(reward)
        self.engagement_state = 5 if self.flow_state == 'flow' else 0

    def execute_task(self, task):
        error = self.check_task_requirements(task, [])
        if error:
            return error
        return {"error": "Task not supported by base DFSNAgent", "agent": self.name}

    def check_task_requirements(self, task, required_keys):
        if "type" not in task:
            return {"error": "Task missing 'type'", "agent": self.name}
        if not all(key in task for key in required_keys):
            return {"error": f"Missing required keys: {required_keys}", "agent": self.name}
        return None

    def share_state(self, peer, intermediate=False):
        state = {"performance_history": self.performance_history[-3:]}
        if intermediate:
            state["intermediate_data"] = self.get_intermediate_data()
        peer.receive_state(state)

    def receive_state(self, state):
        if "performance_history" in state:
            self.performance_history.extend(state["performance_history"])
        if "intermediate_data" in state:
            self.process_intermediate_data(state["intermediate_data"])

    def get_intermediate_data(self):
        return {"perf": np.mean(self.performance_history[-3:]) if self.performance_history else 0}

    def process_intermediate_data(self, data):
        if "perf" in data:
            perf_boost = data["perf"] * 0.1
            self.engagement_state = min(self.engagement_state + perf_boost, 10)
            self.optimizer.learning_rate = max(0.01, min(0.2, self.optimizer.learning_rate + perf_boost * 0.01))
            if data["perf"] > 2.0 and not self.overdrive:
                self.overdrive = True
                print(f"{self.name} entering OVERDRIVE mode!")

    def compute_stability(self):
        return np.var(self.performance_history[-5:]) if len(self.performance_history) >= 5 else 0

    def add_peer(self, peer):
        if peer not in self.peers:
            self.peers.append(peer)

# Flow State Optimizer
class FlowStateOptimizer:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.math_module = PyZeroMathTorch()
        self.action_weights = torch.tensor([0.5, 0.5])
        self.iteration_count = 0

    def predict(self, state):
        task_complexity, performance, stability = state
        complexity = self.math_module.f0z_stabilize(torch.tensor(task_complexity)).item()
        perf = self.math_module.f0z_stabilize(torch.tensor(performance)).item()
        stab = self.math_module.f0z_stabilize(torch.tensor(stability)).item()
        flow_score = (complexity * 0.4 + perf * 0.5 - stab * 0.1)
        idle_score = 1 - flow_score
        scores = torch.tensor([idle_score, flow_score])
        self.action_weights = (1 - self.learning_rate) * self.action_weights + self.learning_rate * scores
        self.action_weights = self.action_weights / self.action_weights.sum()
        self.iteration_count += 1
        return 1 if self.action_weights[1] > self.action_weights[0] else 0

    def update(self, reward):
        reward = self.math_module.f0z_stabilize(torch.tensor(reward)).item()
        adjustment = self.learning_rate * reward * (1 + 0.1 * self.iteration_count)
        if reward > 0:
            self.action_weights[1] += adjustment
            self.action_weights[0] -= adjustment
        else:
            self.action_weights[0] += adjustment
            self.action_weights[1] -= adjustment
        self.action_weights = torch.clamp(self.action_weights, 0.1, 0.9)
        self.action_weights = self.action_weights / self.action_weights.sum()

# Agent Iterative Workflow (Chaos Mode)
class AgentIterativeWorkflow:
    def __init__(self, agent):
        self.agent = agent
        self.base_iterations = 3
        self.parallel_peers = agent.peers
        self.available_peers = [DFSNAgent(f"Guest_{i}") for i in range(3)]  # Pool of recruitable peers

    def iterate_task(self, task, complexity):
        best_result = None
        best_performance = -float('inf')
        intermediate_results = {}
        current_task = task

        # Adaptive Iteration Scaling with Overdrive
        avg_perf = np.mean(self.agent.performance_history[-5:]) if self.agent.performance_history else 1.0
        max_iterations = max(1, min(5, int(self.base_iterations * (complexity / 8.0) / max(0.1, avg_perf))))
        if self.agent.overdrive:
            max_iterations *= 2
            print(f"{self.agent.name} in OVERDRIVE: {max_iterations} iterations!")
        print(f"{self.agent.name} adapting iterations to {max_iterations} for complexity {complexity}")

        for i in range(max_iterations):
            # Chaos Mode: Random Task Mutation
            if random.random() < 0.1:  # 10% chance
                current_task = self.mutate_task(current_task)
                print(f"{self.agent.name} mutated task to {current_task['type']}")

            # Parallel execution with dynamic scaling
            if i == 0 and self.parallel_peers and "data" in current_task:
                parallel_result = self.parallel_execute(current_task, complexity)
                intermediate_results.update(parallel_result)

            # Local execution
            result = self.agent.execute_task(current_task)
            if "result" in result:
                performance = self.evaluate_performance(result["result"], complexity)
                self.agent.performance_history.append(performance)
                if performance > best_performance:
                    best_performance = performance
                    best_result = result

            # Cross-agent task swapping
            if i < max_iterations - 1:
                swapped_task = self.swap_tasks(current_task, performance)
                if swapped_task:
                    current_task = swapped_task
                    print(f"{self.agent.name} swapped task to {current_task['type']}")

            # Share intermediate state and get feedback
            for peer in self.parallel_peers:
                self.agent.share_state(peer, intermediate=True)
            self.agent.adjust_flow_state(complexity, best_performance if best_performance > -float('inf') else 0)

            # Reset overdrive after iteration
            if self.agent.overdrive and i == max_iterations - 1:
                self.agent.overdrive = False
                print(f"{self.agent.name} exiting OVERDRIVE mode.")

        if intermediate_results:
            best_result = best_result or {"result": {}, "agent": self.agent.name}
            best_result["result"]["parallel"] = intermediate_results

        return best_result or {"error": "No valid result", "agent": self.agent.name}

    def parallel_execute(self, task, complexity):
        results = {}
        num_peers = len(self.parallel_peers)
        if num_peers == 0:
            return results

        # Dynamic Peer Scaling
        avg_perf = np.mean(self.agent.performance_history[-3:]) if self.agent.performance_history else 1.0
        if avg_perf > 1.5 and self.available_peers:
            new_peer = self.available_peers.pop(0)
            self.parallel_peers.append(new_peer)
            self.agent.add_peer(new_peer)
            print(f"{self.agent.name} recruited {new_peer.name} to the party!")
            num_peers += 1

        # Priority-based subtasks
        subtasks = self.create_priority_subtasks(task, complexity)
        if not subtasks:
            return results

        for idx, (priority, subtask) in enumerate(subtasks[:num_peers]):
            peer = self.parallel_peers[idx % num_peers]
            result = peer.execute_task(subtask)
            if "result" in result:
                results[peer.name] = result["result"]
                perf = self.evaluate_performance(result["result"], complexity)
                peer.performance_history.append(perf)
                peer.adjust_flow_state(complexity, perf)

        return results

    def create_priority_subtasks(self, task, complexity):
        priority_tasks = []
        if "data" in task and isinstance(task["data"], (list, dict)):
            if isinstance(task["data"], list):
                chunk_size = max(1, len(task["data"]) // len(self.parallel_peers))
                for i in range(0, len(task["data"]), chunk_size):
                    sub_data = task["data"][i:i+chunk_size]
                    priority = len(sub_data) / complexity
                    subtask = {"type": task["type"], "action": task.get("action"), "data": sub_data}
                    heapq.heappush(priority_tasks, (-priority, subtask))
            else:
                for k, v in task["data"].items():
                    priority = 1.0 if isinstance(v, (int, float)) else 0.5
                    subtask = {"type": task["type"], "action": task.get("action"), "data": {k: v}}
                    heapq.heappush(priority_tasks, (-priority, subtask))
        else:
            priority_tasks.append((-1.0, task))

        return [(p, t) for p, t in heapq.nsmallest(len(self.parallel_peers), priority_tasks)]

    def swap_tasks(self, task, performance):
        if performance < 0.5 and self.parallel_peers:
            domain_map = {
                "entropy_calc": "information_theory",
                "machine_learning": "machine_learning",
                "collaboration": "collaboration"
            }
            task_domain = domain_map.get(task["type"], "general")
            if task_domain != self.agent.domain_specialty:
                for peer in self.parallel_peers:
                    if peer.domain_specialty == task_domain:
                        print(f"{self.agent.name} swapping {task['type']} with {peer.name}")
                        return task
        return None

    def mutate_task(self, task):
        mutated = task.copy()
        if "probabilities" in mutated:
            mutated["probabilities"] = [p + random.uniform(-0.1, 0.1) for p in mutated["probabilities"]]
            mutated["probabilities"] = [max(0, min(1, p)) for p in mutated["probabilities"]]
            total = sum(mutated["probabilities"])
            if total > 0:
                mutated["probabilities"] = [p / total for p in mutated["probabilities"]]
        elif "X" in mutated and "Y" in mutated:
            mutated["Y"] = [y + random.uniform(-1, 1) for y in mutated["Y"]]
        return mutated

    def evaluate_performance(self, result, complexity):
        if isinstance(result, (int, float)):
            return result / (complexity + 1)
        elif isinstance(result, dict):
            return sum(v for v in result.values() if isinstance(v, (int, float))) / (complexity + 1)
        elif isinstance(result, list):
            return len(result) / (complexity + 1)
        return 0.0

# Specialized Agents
class CollaborativeAgent(DFSNAgent):
    def __init__(self, name):
        super().__init__(name)
        self.shared_data = {}
        self.domain_specialty = "collaboration"

    def execute_task(self, task):
        error = self.check_task_requirements(task, ["action", "data"])
        if error:
            return error
        if task["type"] == "collaboration":
            if task["action"] == "combine":
                combined = task["data"].copy()
                for peer in self.peers:
                    peer_result = peer.execute_task({"type": "collaboration", "action": "report", "data": {}})
                    if "result" in peer_result:
                        combined[f"{peer.name}_result"] = peer_result["result"]
                self.shared_data.update(combined)
                return {"result": combined, "agent": self.name}
            elif task["action"] == "report":
                return {"result": self.shared_data, "agent": self.name}
        return self.aiw.iterate_task(task, 8.0)

class InformationTheoryAgent(DFSNAgent):
    def __init__(self, name):
        super().__init__(name)
        self.epsilon = 1e-8
        self.domain_specialty = "information_theory"

    def execute_task(self, task):
        error = self.check_task_requirements(task, [])
        if error:
            return error
        if task["type"] == "entropy_calc":
            result = self.compute_entropy(task["probabilities"])
            return {"result": result, "agent": self.name}
        return self.aiw.iterate_task(task, 8.0)

    def compute_entropy(self, probabilities, epsilon=1e-8):
        stabilized_probs = self.math_module.f0z_stabilize(torch.tensor(probabilities, dtype=torch.float32))
        entropy = -torch.sum(stabilized_probs * torch.log2(stabilized_probs + epsilon))
        return entropy.item()

class MachineLearningAgent(DFSNAgent):
    def __init__(self, name):
        super().__init__(name)
        self.weights = torch.zeros(2, requires_grad=True)
        self.optimizer = torch.optim.SGD([self.weights], lr=0.01)
        self.domain_specialty = "machine_learning"

    def execute_task(self, task):
        error = self.check_task_requirements(task, ["action"])
        if error:
            return error
        if task["type"] == "machine_learning":
            if task["action"] == "train":
                loss = self.train_model(task["X"], task["Y"])
                return {"result": {"loss": loss}, "agent": self.name}
            elif task["action"] == "predict":
                pred = self.predict(task["X"])
                return {"result": {"prediction": pred}, "agent": self.name}
        return self.aiw.iterate_task(task, 8.0)

    def train_model(self, X, Y):
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        for _ in range(5):
            pred = X @ self.weights
            loss = torch.mean((pred - Y) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return (X @ self.weights).tolist()

# Multi-Agent Coordinator
class MultiAgentCoordinator:
    def __init__(self, agents):
        self.agents = agents

    def map_domains(self, task_type):
        domain_map = {
            "collaboration": "CollaborativeAgent_1",
            "entropy_calc": "InformationTheoryAgent_1",
            "machine_learning": "MachineLearningAgent_1",
            "science_fair": "CollaborativeAgent_1"
        }
        agent_name = domain_map.get(task_type, "DFSNAgent_0")
        return {a.name: a for a in self.agents if a.name == agent_name}

    def assign_tasks(self, task):
        active_agents = self.map_domains(task["type"])
        for agent in active_agents.values():
            return agent.aiw.iterate_task(task, 8.0)
        return {"error": "No suitable agent found"}

# ZSG Todo
class ZSGTodo:
    def __init__(self, task_id, description, status, priority, eiw_step, data_payload):
        self.task_id = task_id
        self.description = description
        self.status = status
        self.priority = priority
        self.eiw_step = eiw_step
        self.data_payload = data_payload

    def to_json(self):
        return {
            "task_id": self.task_id, "description": self.description,
            "status": self.status, "priority": self.priority,
            "eiw_step": self.eiw_step, "data": self.data_payload
        }

# Episodic Iterative Workflow
class EpisodicIterativeWorkflow:
    def __init__(self):
        self.episode = 1
        self.iteration = 1

    def advance(self):
        self.iteration += 1
        if self.iteration > 10:
            self.episode += 1
            self.iteration = 1

# Dynamic Batch Mode Processor
class DynamicBatchModeProcessor:
    def __init__(self, manager):
        self.manager = manager
        self.batch = []

    def add_to_batch(self, prompt, complexity):
        task_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        todo = ZSGTodo(task_id, prompt, "Pending", complexity, "Process", {})
        self.batch.append(todo)

    def process_batch(self):
        results = []
        for todo in self.batch:
            task = {"type": "science_fair", "action": "process", "data": {"prompt": todo.description}}
            result = self.manager.process_task_with_zsg(task)
            todo.status = "Completed"
            results.append(result)
        self.manager.memory_system.store_task(self.manager.eiw.episode, self.manager.eiw.iteration, todo)
        return results

# ZSG Batch Mode Science Fair
class ZSGBatchModeScienceFair:
    def __init__(self, manager):
        self.manager = manager
        self.processor = DynamicBatchModeProcessor(manager)
        self.collab_agent = next(a for a in manager.agents if isinstance(a, CollaborativeAgent))

    def run_science_fair(self, experiments, horizon=5):
        for exp in experiments:
            self.processor.add_to_batch(f"Experiment: {exp}", 8.0)
        batch_results = self.processor.process_batch()
        collab_task = {"type": "collaboration", "action": "combine", "data": {"batch": batch_results}}
        final_result = self.collab_agent.aiw.iterate_task(collab_task, 8.0)
        return {"batch_results": batch_results, "combined": final_result}

# ZSG Manager
class ZSGManager:
    def __init__(self, agents, complexity=6.0):
        self.agents = agents
        self.complexity = complexity
        self.math_module = PyZeroMathTorch()
        self.multi_agent_coordinator = MultiAgentCoordinator(self.agents)
        self.memory_system = MemorySystem()
        self.eiw = EpisodicIterativeWorkflow()

    def calibrate_episode(self, prompt, complexity):
        self.complexity = complexity

    def process_task_with_zsg(self, task):
        return self.multi_agent_coordinator.assign_tasks(task)

# Memory System
class MemorySystem:
    def __init__(self):
        self.task_store = {}
        self.episode_memory = {}

    def store_task(self, episode, iteration, todo):
        key = f"{episode}_{iteration}"
        if key not in self.task_store:
            self.task_store[key] = []
        self.task_store[key].append(todo.to_json())

# Test Chaos Mode AIW
if __name__ == "__main__":
    # Initialize agents
    agents = [
        CollaborativeAgent("CollaborativeAgent_1"),
        InformationTheoryAgent("InformationTheoryAgent_1"),
        MachineLearningAgent("MachineLearningAgent_1")
    ]
    for agent in agents:
        for peer in agents:
            if agent != peer:
                agent.add_peer(peer)

    # Initialize manager
    manager = ZSGManager(agents, complexity=8.0)
    manager.calibrate_episode("Chaos Mode AIW Party", 8.0)

    # Test chaos features
    ml_task = {"type": "machine_learning", "action": "train", "X": [[1, 0], [2, 0], [3, 0], [4, 0]], "Y": [2, 4, 6, 8]}
    ml_result = manager.process_task_with_zsg(ml_task)
    print(f"ML Result with Chaos Mode: {ml_result}")

    entropy_task = {"type": "entropy_calc", "probabilities": [0.25, 0.25, 0.25, 0.25]}
    entropy_result = manager.process_task_with_zsg(entropy_task)
    print(f"Entropy Result with Overdrive: {entropy_result}")

    # Test Science Fair
    science_fair = ZSGBatchModeScienceFair(manager)
    experiments = ["ML Chaos Drop", "Entropy Freestyle", "Collab Mayhem"]
    sf_result = science_fair.run_science_fair(experiments)
    print(f"Science Fair Chaos Results: {sf_result}")

    # Chaos Unleashed!
    print("\n🎉 Agents are in FULL CHAOS MODE! The party’s a wild, roofless rave—total mayhem! 🎉")
