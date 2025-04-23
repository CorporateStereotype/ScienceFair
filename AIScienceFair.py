
# Merged code from ZSGSFF20250402.txt and NewZSGCode.txt

# --- Imports ---
import os
import random
import hashlib
import resource
import time
import json
import re
import argparse
import threading
from heapq import heappush, heappop
from typing import List, Dict, Any, Optional, Tuple

# External Libraries
try:
    import psutil
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from scipy.integrate import quad, odeint
    from scipy import signal
    import scipy.linalg as LA
    import scipy.fft as fft
    import matplotlib.pyplot as plt
    from prompt_toolkit import PromptSession
    from transformers import pipeline
    import redis
    from concurrent.futures import ThreadPoolExecutor
    import requests # Added for Ollama LLMAgent
except ImportError as e:
    print(f"Error importing standard libraries: {e}. Please install requirements.")
    # Define dummy classes/functions if imports fail
    # (Dummy definitions omitted for brevity - assume standard libs are installed)
    pass

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    print("Qiskit libraries loaded successfully.")
except ImportError as e:
    print(f"Qiskit import error: {e}. Qiskit integration disabled.")
    QuantumCircuit = None
    AerSimulator = None
    NoiseModel = None
    depolarizing_error = None
    ClassicalRegister = None
    QuantumRegister = None
# --- End Qiskit Imports ---

# --- asyncio Patch for Notebooks ---
import asyncio
import nest_asyncio
try:
    nest_asyncio.apply()
    print("Applied nest_asyncio patch.")
except RuntimeError:
    print("nest_asyncio already applied or not required.")
except ImportError:
    print("Warning: nest_asyncio not installed. Shell might face event loop issues in notebooks.")
     

# ZSGSFF20250402_merged_with_Qiskit.py
# Merged code from ZSGSFF20250402.txt and NewZSGCode.txt

# --- Imports ---
import os
import random
import hashlib
import resource
import time
import json
import re
import argparse
import threading
from heapq import heappush, heappop
from typing import List, Dict, Any, Optional, Tuple

# External Libraries
try:
    import psutil
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from scipy.integrate import quad, odeint
    from scipy import signal
    import scipy.linalg as LA
    import scipy.fft as fft
    import matplotlib.pyplot as plt
    from prompt_toolkit import PromptSession
    from transformers import pipeline
    import redis
    from concurrent.futures import ThreadPoolExecutor
    import requests # Added for Ollama LLMAgent
except ImportError as e:
    print(f"Error importing standard libraries: {e}. Please install requirements.")
    # Define dummy classes/functions if imports fail
    # (Dummy definitions omitted for brevity - assume standard libs are installed)
    pass

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    print("Qiskit libraries loaded successfully.")
except ImportError as e:
    print(f"Qiskit import error: {e}. Qiskit integration disabled.")
    QuantumCircuit = None
    AerSimulator = None
    NoiseModel = None
    depolarizing_error = None
    ClassicalRegister = None
    QuantumRegister = None
# --- End Qiskit Imports ---

# --- asyncio Patch for Notebooks ---
import asyncio
import nest_asyncio
try:
    nest_asyncio.apply()
    print("Applied nest_asyncio patch.")
except RuntimeError:
    print("nest_asyncio already applied or not required.")
except ImportError:
    print("Warning: nest_asyncio not installed. Shell might face event loop issues in notebooks.")
# --- End asyncio Patch ---

# --- F0Z Mathematical Foundations -
     
Qiskit libraries loaded successfully.
Applied nest_asyncio patch.


     


     

# ZSGSFF20250402_merged_with_Qiskit.py
# --- End asyncio Patch ---
# --- F0Z Mathematical Foundations -

class PyZeroMathTorch:
    """PyTorch-based F0Z stabilization math module."""
    def __init__(self, epsilon_0=1e-8, scaling_factor=0.1):
        self.epsilon_0 = epsilon_0
        self.scaling_factor = scaling_factor
        self.current_epsilon = torch.tensor(epsilon_0)

    def f0z_stabilize(self, x: torch.Tensor, system_size: Optional[int] = None) -> torch.Tensor:
        """Stabilizes near-zero values in a PyTorch tensor."""
        if not isinstance(x, torch.Tensor):
             try: x = torch.tensor(x)
             except Exception as e:
                 print(f"Warning: Could not convert input {type(x)} to tensor in f0z_stabilize. Error: {e}")
                 return x

        current_eps = self.current_epsilon
        if system_size is not None and system_size > 1:
            adjusted_epsilon = self.epsilon_0 * self.scaling_factor * torch.log(torch.tensor(float(system_size)))
            current_eps = torch.max(torch.tensor(self.epsilon_0), adjusted_epsilon)
        current_eps = current_eps.to(x.device).to(x.dtype)
        mask = torch.abs(x) < current_eps
        signs = torch.sign(x)
        signs = torch.where(x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), signs)
        stabilized_values = current_eps * signs

        if torch.is_complex(x):
            real_mask = torch.abs(x.real) < current_eps
            imag_mask = torch.abs(x.imag) < current_eps
            real_signs = torch.sign(x.real)
            real_signs = torch.where(x.real == 0, torch.tensor(1.0, device=x.device, dtype=x.real.dtype), real_signs)
            imag_signs = torch.sign(x.imag)
            imag_signs = torch.where(x.imag == 0, torch.tensor(1.0, device=x.device, dtype=x.imag.dtype), imag_signs)
            stabilized_real = torch.where(real_mask, current_eps * real_signs, x.real)
            stabilized_imag = torch.where(imag_mask, current_eps * imag_signs, x.imag)
            return torch.complex(stabilized_real, stabilized_imag)
        else:
            return torch.where(mask, stabilized_values, x)

    def update_epsilon(self, entropy_change: float, alpha: float = 0.01):
        update_factor = torch.tensor(1.0 + alpha * entropy_change)
        self.current_epsilon *= update_factor
        self.current_epsilon = torch.max(self.current_epsilon, torch.tensor(self.epsilon_0))
        return self.current_epsilon.item()

    def f0z_multiply(self, a, b, task_complexity=1):
        a_t = torch.as_tensor(a)
        b_t = torch.as_tensor(b)
        result = a_t * b_t
        return self.f0z_stabilize(result, system_size=task_complexity * 10)

    def f0z_integral(self, func, lower, upper, task_complexity=1):
        try:
            result_val, _ = quad(func, lower, upper)
            result = torch.tensor(result_val, dtype=torch.float32)
            return self.f0z_stabilize(result, system_size=task_complexity * 100).item()
        except Exception as e:
            print(f"Error during integration: {e}")
            return self.f0z_stabilize(torch.tensor(0.0), system_size=task_complexity * 100).item()

    def f0z_matrix_multiply(self, A, B, mode='continuous'):
        A_t = torch.as_tensor(A, dtype=torch.float32)
        B_t = torch.as_tensor(B, dtype=torch.float32)
        result = torch.matmul(A_t, B_t)
        if mode == 'discrete':
            return torch.clamp(result, -1e8, 1e8)
        else:
            return self.f0z_stabilize(result, system_size=A_t.shape[0]*A_t.shape[1])

    def f0z_softmax(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray): x = np.array(x)
        e_x = np.exp(x - np.max(x))
        probs = e_x / (e_x.sum(axis=-1, keepdims=True) + self.epsilon_0)
        return self.f0z_stabilize(torch.tensor(probs)).numpy()

    def visualize_f0z(self, func, lower, upper, task_complexity):
        self.adjust_epsilon(task_complexity)
        x = np.linspace(lower, upper, 100)
        y_raw = [func(xi) for xi in x]
        y_stab_tensor = [self.f0z_stabilize(torch.tensor(float(func(xi))), system_size=task_complexity*10) for xi in x]
        y_stab = [t.item() for t in y_stab_tensor]
        try:
            plt.figure(figsize=(8, 4)); plt.plot(x, y_raw, label="Raw"); plt.plot(x, y_stab, label=f"F0Z (eps={self.current_epsilon.item():.2e})", linestyle='--')
            plt.title("F0Z Stabilization Effect"); plt.xlabel("Input"); plt.ylabel("Output"); plt.legend(); plt.grid(True); plt.show()
            print("F0Z visualization plot displayed (simulated if matplotlib is unavailable).")
        except Exception as e: print(f"Matplotlib visualization failed: {e}. Skipping plot.")

    def adjust_epsilon(self, task_complexity):
        self.current_epsilon = torch.tensor(self.epsilon_0 * max(1.0, task_complexity / 5.0))

# Remove inheritance from DFSNAgent for F0ZAgent if it's just an interface

class F0ZAlgebra:
    """Extended F0Z algebraic operations using NumPy and stabilization."""
    math_module = PyZeroMathTorch()
    @staticmethod
    def f0z_variance(data: np.ndarray, epsilon: float = 1e-8) -> float:
        if not isinstance(data, np.ndarray): data = np.array(data)
        if data.size == 0: return 0.0
        mean = np.mean(data); var = np.mean((data - mean) ** 2)
        stabilized_var_t = F0ZAlgebra.math_module.f0z_stabilize(torch.tensor(var, dtype=torch.float32), system_size=len(data))
        return stabilized_var_t.item()
    @staticmethod
    def f0z_gradient(vector_field: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        if not isinstance(vector_field, np.ndarray): vector_field = np.array(vector_field)
        grad = np.gradient(vector_field)
        stabilized_grad_t = F0ZAlgebra.math_module.f0z_stabilize(torch.tensor(grad, dtype=torch.float32), system_size=vector_field.size)
        return stabilized_grad_t.numpy()
    @staticmethod
    def f0z_correlation(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-8) -> float:
        if not isinstance(x, np.ndarray): x = np.array(x)
        if not isinstance(y, np.ndarray): y = np.array(y)
        if x.size == 0 or y.size == 0 or x.size != y.size: return 0.0
        mean_x = np.mean(x); mean_y = np.mean(y); cov = np.mean((x - mean_x) * (y - mean_y))
        std_x = np.sqrt(F0ZAlgebra.f0z_variance(x, epsilon)); std_y = np.sqrt(F0ZAlgebra.f0z_variance(y, epsilon))
        denominator_t = torch.tensor(std_x * std_y, dtype=torch.float32)
        stabilized_denominator_t = F0ZAlgebra.math_module.f0z_stabilize(denominator_t, system_size=len(x))
        if stabilized_denominator_t.item() == 0: return 0.0
        else: corr = cov / stabilized_denominator_t.item(); return float(np.clip(corr, -1.0, 1.0))


class F0ZAgent: # NOT inheriting DFSNAgent
    """Agent integrating a local LLM for inference, simulating F0Z concepts."""
    #def __init__(self, name="F0ZAgent_LLM"): # Give it a name
    def __init__(self,): # Give it a name
        # Removed super().__init__(name)
        self.name = name # Set name directly
        self.llm = None
        self.math_sim = PyZeroMathTorch()
        self.k=1.0; self.Ea=0.1; self.kB=8.617e-5; self.T=300
        try:
            device = -1 # Default to CPU for the small embedded model
            # Use CPU for gpt2 if Ollama agent handles the main LLM tasks
            self.llm = pipeline("text-generation", model="gpt2", device=device)
            print(f"F0ZAgent LLM interface initialized with: gpt2 on CPU")
        except Exception as e:
            print(f"LLM loading failed for F0ZAgent: {e}. Inference disabled.")
            self.llm = None


    def simulate_f0z_reaction(self, D_Pd_ratio: float, defect_density: float = 0.7, f0z_symmetry: float = 0.95) -> float:
        """Simulates a hypothetical F0Z-influenced reaction rate based on formula in prompt."""
        # R_F0Z = k * (D/Pd)^3.2 * exp(-Ea/kBT) * (1 + 0.14*defect) * (1 + 0.10*symmetry)
        if D_Pd_ratio < 0: D_Pd_ratio = 0 # Ensure base is non-negative

        term1 = self.k * (D_Pd_ratio ** 3.2)
        term2 = np.exp(-self.Ea / (self.kB * self.T))
        term3 = (1.0 + 0.14 * defect_density)
        term4 = (1.0 + 0.10 * f0z_symmetry)

        rate = term1 * term2 * term3 * term4

        # Apply F0Z stabilization to the final rate
        stabilized_rate = self.math_sim.f0z_stabilize(torch.tensor(rate)).item()
        print(f"  Simulated F0Z Reaction Rate: {stabilized_rate:.4e}")
        return stabilized_rate

    def infer(self, query: str, max_length: int = 100) -> str:
        """Uses the integrated LLM to answer queries related to F0Z or other topics."""
        if not self.llm:
            return "LLM not available in F0ZAgent."
        try:
            # Add context to the query for the LLM
            full_prompt = f"Regarding the Formula for Zero (F0Z) concept, {query}"
            # Ensure max_length is reasonable
            max_len = max(20, min(max_length, 200)) # Clamp length
            result = self.llm(full_prompt, max_length=max_len, num_return_sequences=1, do_sample=True)
            generated_text = result[0]['generated_text']
            # Clean up the response (e.g., remove the prompt if model includes it)
            if generated_text.startswith(full_prompt):
                 cleaned_text = generated_text[len(full_prompt):].strip()
            else:
                 cleaned_text = generated_text
            print(f"  LLM Inference: Q='{query[:30]}...' -> A='{cleaned_text[:50]}...'")
            return cleaned_text
        except Exception as e:
            print(f"Error during LLM inference: {e}")
            return f"Error during inference: {e}"

    # Make F0ZAgent behave like a DFSNAgent for integration
    # Remove execute_task if not needed, or simplify it if called directly
    def execute_simulation_task(self, task: Dict) -> Dict:
         """Handles ONLY f0z_simulation tasks directly."""
         if task.get("type") == "f0z_simulation" or task.get("action") == "simulate_f0z_reaction":
              # Basic check, assumes 'data' payload contains keys if needed
              required = task.get("data", {}).get("required_keys", ["D_Pd_ratio"])
              missing = [k for k in required if k not in task.get("data", {})]
              if missing: return {"error": f"F0Z Sim Missing keys: {missing}", "agent": self.name}
              rate = self.simulate_f0z_reaction(
                  task["data"]["D_Pd_ratio"],
                  task.get("data", {}).get("defect_density", 0.7),
                  task.get("data", {}).get("f0z_symmetry", 0.95)
              )
              return {"result": {"reaction_rate": rate}, "agent": self.name}
         return {"error": f"Unsupported task type for F0ZAgent simulation: {task.get('type')}", "agent": self.name}

    def execute_inference_task(self, task: Dict) -> Dict:
         """Handles ONLY llm_inference tasks directly."""
         if task.get("type") == "llm_inference" or task.get("action") == "infer":
              query = task.get("data", {}).get("query", task.get("query")) # Get query from data or root
              if not query: return {"error": "Missing query for inference", "agent": self.name}
              response = self.infer(query, task.get("data", {}).get("max_length", 100))
              return {"result": {"llm_response": response}, "agent": self.name}
         return {"error": f"Unsupported task type for F0ZAgent inference: {task.get('type')}", "agent": self.name}

    # Keep check_task_requirements if used by execute_* methods above

    # Add dummy methods required by DFSNAgent structure if needed for coordination
    def __init__(self, name="F0ZAgent_LLM"): # Give it a name
        super().__init__(name) # Call parent init if inheriting (needs DFSNAgent inheritance) - OR just be standalone
        self.llm = None # Re-init LLM here
        self.math_sim = PyZeroMathTorch()
        self.k=1.0; self.Ea=0.1; self.kB=8.617e-5; self.T=300
        try:
            device = 0 if torch.cuda.is_available() else -1
            # Ensure model name is valid or handle error
            valid_models = ["gpt2", "distilgpt2"] # Example small models
            model_to_load = "gpt2" # Default to gpt2
            # If a specific model was intended: model_to_load = "meta-llama/Llama-3.1" # Requires access/download
            self.llm = pipeline("text-generation", model=model_to_load, device=device)
            print(f"F0ZAgent LLM interface initialized with: {model_to_load}")
        except Exception as e:
            print(f"LLM loading failed for F0ZAgent: {e}. Inference disabled.")
            self.llm = None

class ZSGQuantumBridge:
    """Connects classical data/agents to the Qiskit-based quantum simulator."""
    def __init__(self, n_logical_qubits: int = 4, n_physical_per_logical: int = 1, use_noise:bool = True, noise_level:float = 0.01):
        self.n_qubits = n_logical_qubits * n_physical_per_logical # Use physical qubits directly
        if self.n_qubits == 0 or QuantumCircuit is None: self.simulator = None; print("Warning: Qiskit Bridge created with 0 qubits or Qiskit not found.")
        else: self.simulator = ZSGQuantumSimulator(self.n_qubits, use_noise=use_noise, noise_level=noise_level)
        self.entropy_bot = EntropyBotTurbo(); print(f"ZSGQuantumBridge initialized with Qiskit backend: {self.n_qubits} qubits.")
    def get_circuit(self): return self.simulator.circuit if self.simulator else None
    def set_circuit(self, circuit: QuantumCircuit):
        if self.simulator and isinstance(circuit, QuantumCircuit) and circuit.num_qubits == self.n_qubits: self.simulator.circuit = circuit
        else: print("Error: Cannot set circuit - invalid circuit or simulator.")
    def encode(self, data: Any) -> QuantumCircuit:
        if not self.simulator: return None
        self.simulator._reset_circuit()
        try: data_str = json.dumps(data, sort_keys=True)
        except TypeError: data_str = str(data)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        for i in range(self.n_qubits): # Apply H and Rz based on hash
             self.simulator.h_gate(i)
             angle = (int(data_hash[i % len(data_hash)], 16) / 15.0) * np.pi; self.simulator.rz_gate(angle, i)
        # print(f"Encoded data hash {data_hash[:8]}... into {self.n_qubits}-qubit Qiskit circuit.") # Reduce noise
        return self.simulator.circuit
    def run_circuit(self, circuit: Optional[QuantumCircuit] = None, shots: int = 1024) -> Dict:
        if not self.simulator: return {"error": "No simulator"}
        sim_circuit = circuit if circuit is not None else self.simulator.circuit
        if sim_circuit is None or sim_circuit.num_qubits == 0: return {"error": "No valid circuit"}
        results = self.simulator.measure(shots=shots)
        probs_vector = np.zeros(2**self.n_qubits); entropy = 0.0
        if results and "probabilities" in results:
            for state, prob in results["probabilities"].items():
                try: idx = int(state, 2); probs_vector[idx] = prob
                except (ValueError, IndexError): continue
            entropy = self.entropy_bot.compute_entropy(probs_vector)
            results["estimated_entropy"] = entropy
        return results
    # Teleportation requires Qiskit circuit logic (complex, keeping simplified version)
    def teleport(self, source_qubit: int, target_qubit: int, ancilla_qubit: int) -> Optional[Dict]:
        if not self.simulator or self.n_qubits < 3: print("Error: Teleportation needs >= 3 qubits."); return None
        circuit = self.simulator.circuit # Get current circuit
        try:
            # Check indices
            if not all(0 <= q < self.n_qubits for q in [source_qubit, target_qubit, ancilla_qubit]): raise ValueError("Qubit index out of bounds")
            if len({source_qubit, target_qubit, ancilla_qubit}) != 3: raise ValueError("Source, target, and ancilla qubits must be distinct")
            # 1. Create Bell pair
            circuit.h(ancilla_qubit); circuit.cx(ancilla_qubit, target_qubit); circuit.barrier()
            # 2. CNOT source to ancilla, H source
            circuit.cx(source_qubit, ancilla_qubit); circuit.h(source_qubit); circuit.barrier()
            # 3. Measure source and ancilla - Need classical bits
            if circuit.num_clbits < 2: circuit.add_register(ClassicalRegister(2, 'teleport_c'))
            circuit.measure(source_qubit, 0); circuit.measure(ancilla_qubit, 1); circuit.barrier()
            # 4. Conditional gates (placeholder comment - requires classical control)
            # Apply X to target if clbit 1 is 1; Apply Z to target if clbit 0 is 1
            print(f"Teleportation circuit constructed for {source_qubit}->{target_qubit} via {ancilla_qubit}.")
            return {"circuit": circuit, "notes": "Post-measurement corrections needed or use conditional execution."}
        except Exception as e: print(f"Error building teleport circuit: {e}"); return {"error": str(e)}

# --- Base Agent Class ---
class Agent:
    """Basic Agent definition."""
    def __init__(self, name: str): self.name = name; self.engagement_state = 0
    def get_engagement_state(self) -> int: return self.engagement_state
    def set_engagement_state(self, state: int): print(f"{self.name} engagement set to {state}"); self.engagement_state = state
    def adjust_workload(self, peer_engagement_level: int):
        if peer_engagement_level > self.engagement_state: print(f"{self.name} sees higher peer engagement ({peer_engagement_level}), considering offloading (placeholder).")
        # else: print(f"{self.name} sees lower/equal peer engagement ({peer_engagement_level}), maintaining workload (placeholder).") # Reduce noise
    def _execute_single_task_iteration(self, task: Dict) -> Dict: return {"status": "executed by generic agent", "agent": self.name, "result": None}
    def check_task_requirements(self, task: Dict, required_keys: List[str]) -> Optional[Dict]:
        missing_keys = [key for key in required_keys if key not in task]
        if missing_keys: error_msg = f"Task missing required keys: {', '.join(missing_keys)}"; print(f"Error for agent {self.name}: {error_msg}"); return {"error": error_msg, "agent": self.name}
        return None

# --- ZSG Agent & DFSN Agent Base Classes ---
class ZSGAgent(Agent):
    """ZeroSumGame Agent, inheriting basic Agent properties."""
    def __init__(self, name: str): super().__init__(name); print(f"ZSGAgent {self.name} initialized.")

class FlowStateOptimizer:
    """Optimizes agent flow state transitions based on performance and complexity."""
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate; self.math_module = PyZeroMathTorch()
	# Weights for [idle_pref, flow_pref, vibe_code_pref]
        self.action_weights = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)
        self.iteration_count = 0; self.last_state = None; self.last_action = None
    def predict(self, state: List[float], is_coding_task: bool = False) -> int:
        task_complexity, performance, stability = state
        complexity_s = self.math_module.f0z_stabilize(torch.tensor(task_complexity)).item()
        perf_s = self.math_module.f0z_stabilize(torch.tensor(performance)).item()
        stab_s = self.math_module.f0z_stabilize(torch.tensor(stability)).item()
        flow_score = (complexity_s * 0.4 + perf_s * 0.5 - stab_s * 0.1); idle_score = 1.0 - flow_score
        vibe_code_score = 0.0
        if is_coding_task: vibe_code_score = (complexity_s * 0.5 + perf_s * 0.6 - stab_s * 0.05)
        scores = torch.tensor([idle_score, flow_score, vibe_code_score], dtype=torch.float32); probs = torch.softmax(scores, dim=0)
        weighted_probs = self.action_weights * probs; action = torch.argmax(weighted_probs).item()
        self.last_state = state; self.last_action = action; self.iteration_count += 1
        return action
    def update(self, reward: float):
        if self.last_action is None: return
        reward_s = self.math_module.f0z_stabilize(torch.tensor(reward)).item()
        target_weights = torch.zeros_like(self.action_weights)
        if reward_s > 0: target_weights[self.last_action] = 1.0
        else:
            penalty_distribution = torch.ones_like(self.action_weights); penalty_distribution[self.last_action] = 0.0
            if torch.sum(penalty_distribution) > 0: target_weights = penalty_distribution / torch.sum(penalty_distribution)
        self.action_weights = (1 - self.learning_rate) * self.action_weights + self.learning_rate * target_weights
        self.action_weights /= torch.sum(self.action_weights); self.action_weights = torch.clamp(self.action_weights, 0.05, 0.9); self.action_weights /= torch.sum(self.action_weights)
        self.last_action = None; self.last_state = None

class AgentIterativeWorkflow:
    """Manages internal iterations for an agent's task execution."""
    def __init__(self, agent: Agent, max_iterations: int = 3):
        self.agent = agent; self.max_iterations = max_iterations
        if not hasattr(agent, 'performance_history'): self.agent.performance_history = []
        if not hasattr(agent, 'math_module'): self.agent.math_module = PyZeroMathTorch()
        if not hasattr(agent, 'adjust_flow_state'): self.agent.adjust_flow_state = lambda *args: None
    def iterate_task(self, task: Dict, complexity: float) -> Dict:
        best_result = {"error": "AIW: No valid result", "agent": self.agent.name}; best_performance = -float('inf')
        # print(f"AIW starting for {self.agent.name}, Task: {task.get('type', 'N/A')}, Max Iter: {self.max_iterations}") # Reduce noise
        for i in range(self.max_iterations):
            # print(f"  AIW Iteration {i+1}/{self.max_iterations}") # Reduce noise
            if hasattr(self.agent, '_execute_single_task_iteration'): current_result = self.agent._execute_single_task_iteration(task)
            else: print(f"Error: Agent {self.agent.name} missing '_execute_single_task_iteration'."); current_result = {"error": "Missing method", "agent": self.agent.name}
            if "error" in current_result: print(f"    Error during iteration: {current_result['error']}"); continue
            performance = self.evaluate_performance(current_result.get("result"), complexity)
            if hasattr(self.agent, 'performance_history'): self.agent.performance_history.append(performance)
            if performance > best_performance: best_performance = performance; best_result = current_result
            task_type = task.get("type", ""); task_desc = task.get("description", "").lower()
            is_coding = ("code" in task_type or any(kw in task_desc for kw in ["code", "debug", "refactor", "implement", "algorithm"]) or task.get("is_coding_task", False))
            if hasattr(self.agent, 'adjust_flow_state') and hasattr(self.agent, 'compute_stability'):
                 stability = self.agent.compute_stability()
                 self.agent.adjust_flow_state(complexity, performance, is_coding_task=is_coding)
        # print(f"AIW finished for {self.agent.name}. Best Perf: {best_performance:.4f}") # Reduce noise
        if "error" not in best_result: best_result["final_performance"] = best_performance
        return best_result
    def evaluate_performance(self, result_data: Any, complexity: float) -> float:
        complexity_penalty = 1.0 + max(0, complexity); perf = 0.0
        if isinstance(result_data, (int, float)): perf = result_data
        elif isinstance(result_data, (np.ndarray, torch.Tensor, list)):
            try: numeric_vals = [float(v) for v in np.abs(np.array(result_data)).flatten() if isinstance(v, (int, float, np.number))]; perf = np.mean(numeric_vals) if numeric_vals else 0.0
            except Exception: perf = 0.0
        elif isinstance(result_data, dict):
            if "accuracy" in result_data: perf = result_data["accuracy"] * 10
            elif "loss" in result_data: perf = -result_data["loss"]
            else: numeric_vals = [float(v) for v in result_data.values() if isinstance(v, (int, float))]; perf = sum(numeric_vals) if numeric_vals else 0.0
        elif isinstance(result_data, bool): perf = 1.0 if result_data else 0.0
        stabilized_perf = self.agent.math_module.f0z_stabilize(torch.tensor(perf)).item()
        return stabilized_perf / (complexity_penalty + 1e-8)
    def adapt_task(self, task: Dict, result: Dict, performance: float) -> Dict: return task # Placeholder

class DFSNAgent(ZSGAgent):
    """Agent capable of operating within the Dynamic Flow State Network."""
    def __init__(self, name: str, math_module: Optional[PyZeroMathTorch] = None, pre_trained_model_path: Optional[str] = None):
        super().__init__(name)
        self.flow_state = 'idle' # 'idle', 'flow', 'vibe_code'
        self.performance_history: List[float] = []
        self.math_module = math_module if math_module else PyZeroMathTorch()
        self.optimizer = FlowStateOptimizer()
        self.aiw = AgentIterativeWorkflow(self)
        self.cpu_allocation = 0.0; self.memory_allocation = 0.0; self.state_vector = None
        self.policy_model = self.load_policy_model(pre_trained_model_path)
        # print(f"DFSNAgent {name} initialized. Flow state: {self.flow_state}") # Reduce noise

    def load_policy_model(self, path: Optional[str]):
         if path and os.path.exists(path): print(f"{self.name}: Loading pre-trained model from {path}"); return {"model_type": "placeholder", "path": path}
         return None
    def enter_flow_state(self):
        if self.flow_state != 'flow': print(f"{self.name} entered flow state (engagement: {self.engagement_state}).")
        self.set_engagement_state(max(5, self.engagement_state + 2)); self.flow_state = 'flow'
    def enter_vibe_code_state(self):
        if self.flow_state != 'vibe_code': print(f"{self.name} entered vibe code state (engagement: {self.engagement_state}).")
        self.set_engagement_state(max(6, self.engagement_state + 1)); self.flow_state = 'vibe_code'
    def exit_flow_state(self):
        if self.flow_state != 'idle':
             previous_state = self.flow_state; self.set_engagement_state(min(3, self.engagement_state - 2)); self.flow_state = 'idle'
             print(f"{self.name} exited {previous_state} state to idle (engagement: {self.engagement_state}).")
    def adjust_flow_state(self, task_complexity: float, performance: float, is_coding_task: bool = False):
        stability = self.compute_stability(); current_state_features = [task_complexity, performance, stability]
        action = self.optimizer.predict(current_state_features, is_coding_task)
        if action == 0 and self.flow_state != 'idle': self.exit_flow_state()
        elif action == 1 and self.flow_state != 'flow':
            if self.flow_state == 'vibe_code': self.exit_flow_state()
            self.enter_flow_state()
        elif action == 2 and self.flow_state != 'vibe_code':
            if self.flow_state == 'flow': self.exit_flow_state()
            self.enter_vibe_code_state()
        reward = performance - 0.05 * task_complexity; self.optimizer.update(reward)
    def execute_task(self, task: Dict) -> Dict: # This method calls AIW
        complexity = task.get('complexity', 5.0); result = self.aiw.iterate_task(task, complexity); return result
    def _execute_single_task_iteration(self, task: Dict) -> Dict: # Base implementation for one iteration
         error = self.check_task_requirements(task, []);
         if error: return error
         print(f"Warning: {self.name} using base DFSNAgent _execute_single_task_iteration. Task type '{task.get('type', 'N/A')}' likely not handled.")
         time.sleep(0.01 * task.get('complexity', 1.0))
         return {"result": f"Base execution for {task.get('type', 'N/A')}", "agent": self.name}
    def share_state(self, peer: Agent):
        if not isinstance(peer, DFSNAgent): print(f"{self.name} cannot share state with non-DFSN agent {peer.name}"); return
        state_to_share = {"performance_history": self.performance_history[-10:], "flow_state": self.flow_state, "engagement_state": self.engagement_state, "domain_data": self.get_domain_specific_state()}
        # print(f"{self.name} sharing state with {peer.name}: Flow={self.flow_state}, Engage={self.engagement_state}") # Reduce noise
        peer.receive_state(state_to_share)
    def receive_state(self, state: Dict):
        # print(f"{self.name} received state update from peer.") # Reduce noise
        if "performance_history" in state: self.performance_history.extend(state["performance_history"]); self.performance_history = self.performance_history[-50:]
        if "domain_data" in state: self.process_domain_specific_state(state["domain_data"])
    def compute_stability(self) -> float:
        if len(self.performance_history) < 5: return 0.0
        variance = F0ZAlgebra.f0z_variance(self.performance_history[-10:])
        stability_score = 1.0 / (variance + float(self.math_module.epsilon_0))
        return float(np.clip(stability_score, 0.0, 100.0))
    def get_domain_specific_state(self) -> Optional[Dict]: return None
    def process_domain_specific_state(self, domain_data: Dict): pass

# --- Core ZSG Systems ---
# --- Core ZSG Systems ---
class MemorySystem:
    """Manages short-term and long-term memory for ZSG agents."""
    def __init__(self):
        # Simple dict-based short-term memory (e.g., last result per task type)
        self.short_term_memory: Dict[str, Any] = {}
        # List-based long-term memory (e.g., history of significant events or results)
        self.long_term_memory: List[Any] = []
        # Task-specific storage (from later iterations)
        self.task_store: Dict[str, List[Dict]] = {} # Key: episode_iteration string
        self.episode_memory: Dict[Tuple[int, int], Any] = {} # Key: (episode, iteration) tuple
        self.max_long_term_size = 1000 # Configurable limit
        print("MemorySystem initialized.")


    def store_memory(self, memory_data: Any, memory_type: str = 'short', key: Optional[str] = None):
        """Store data in the specified memory type."""
        if memory_type == 'short':
            if key is None: key = f"generic_{time.time()}" # Generate a key if none provided
            self.short_term_memory[key] = memory_data
            # print(f"Stored in short-term memory with key '{key}'.")
        elif memory_type == 'long':
            self.long_term_memory.append(memory_data)
            # Enforce size limit
            if len(self.long_term_memory) > self.max_long_term_size:
                self.long_term_memory.pop(0) # Remove the oldest item
            # print(f"Stored in long-term memory (size: {len(self.long_term_memory)}).")
        else:
            print(f"Warning: Unknown memory type '{memory_type}'. Data not stored.")

    def retrieve_memory(self, memory_type: str = 'short', key: Optional[str] = None, criteria: Optional[callable] = None) -> Any:
        """
        Retrieve data from the specified memory type.
        For long-term, criteria can be a function that takes a memory item and returns True if it matches.
        """
        if memory_type == 'short':
            if key:
                return self.short_term_memory.get(key)
            else:
                # Return the most recent item or None if empty
                return list(self.short_term_memory.values())[-1] if self.short_term_memory else None
        elif memory_type == 'long':
            if criteria:
                # Search long-term memory backwards (most recent first) based on criteria
                for item in reversed(self.long_term_memory):
                    try:
                        if criteria(item):
                            return item
                    except Exception as e:
                         # Catch errors in criteria function evaluation
                        print(f"Error applying criteria function in retrieve_memory: {e}")
                        continue # Skip this item
                return None # No item matched the criteria
            else:
                # Return the most recent long-term memory item or None if empty
                return self.long_term_memory[-1] if self.long_term_memory else None
        else:
            print(f"Warning: Unknown memory type '{memory_type}'. Cannot retrieve.")
            return None

    def store_task(self, episode: int, iteration: int, todo): # Assuming todo has a .to_json() method
        """Stores a specific task identified by episode and iteration."""
        key = f"{episode}_{iteration}"
        if key not in self.task_store:
            self.task_store[key] = []
        # Ensure todo is serializable or get its JSON representation
        try:
            todo_data = todo.to_json() if hasattr(todo, 'to_json') else todo
            if not isinstance(todo_data, dict):
                 raise ValueError("Stored task data must be a dictionary or have a to_json method.")
            self.task_store[key].append(todo_data)
            # print(f"Stored task for {key}. Total tasks for key: {len(self.task_store[key])}")
        except Exception as e:
             print(f"Error storing task for {key}: {e}. Data: {todo}")


    def store_episode(self, episode: int, iteration: int, results: Any):
        """Stores the results associated with a specific episode and iteration."""
        self.episode_memory[(episode, iteration)] = results
        # print(f"Stored results for episode {episode}, iteration {iteration}.")

    def retrieve_task(self, episode: int, iteration: int) -> Optional[List[Dict]]:
         """Retrieves tasks for a given episode and iteration."""
         key = f"{episode}_{iteration}"
         return self.task_store.get(key)

    def retrieve_episode(self, episode: int, iteration: int) -> Optional[Any]:
         """Retrieves results for a given episode and iteration."""
         return self.episode_memory.get((episode, iteration))

class ResourceMonitor:
    """Monitors system resources and can adjust parameters like batch size."""
    def __init__(self, batch_size_init=32):
        self.batch_size = batch_size_init
        self.base_cpu = 10 # Base allocation % per agent
        self.base_memory = 10 # Base allocation % per agent
        self.total_cpu = psutil.cpu_count() * 100 # Theoretical max %
        self.total_memory = 100 # Percentage based
        self.processes = {} # Track resource usage per process/agent if needed
        print(f"ResourceMonitor initialized. Initial batch size: {self.batch_size}")


    def check_usage(self) -> Tuple[float, float]:
        """Checks current overall CPU and virtual memory usage."""
        cpu_usage = psutil.cpu_percent(interval=0.1) # Non-blocking short interval
        mem_usage = psutil.virtual_memory().percent
        # print(f"Resource Check: CPU={cpu_usage}%, Memory={mem_usage}%")
        return cpu_usage, mem_usage

    def adjust_batch_size(self) -> int:
        """Adjusts batch size based on resource usage."""
        cpu, mem = self.check_usage()
        if cpu > 85.0 or mem > 85.0:
            print(f"High resource usage (CPU: {cpu:.1f}%, Mem: {mem:.1f}%). Reducing batch size.")
            self.batch_size = max(8, self.batch_size // 2) # Halve batch size, min 8
        elif cpu < 50.0 and mem < 50.0:
             # Gradually increase batch size if resources are low
             self.batch_size = min(128, self.batch_size + 4) # Increase slowly, max 128
        print(f"Adjusted batch size to: {self.batch_size}")
        return self.batch_size

    def pre_allocate(self, agents: List[DFSNAgent], task_complexity: float) -> Dict[str, float]:
        """Estimate and pre-allocate resources based on agent count and task complexity."""
        num_agents = len(agents)
        if num_agents == 0: return {"cpu_total": 0, "memory_total": 0}

        # Estimate needed resources - simple scaling
        cpu_needed_per_agent = min(self.base_cpu + 5 * task_complexity, 100 / num_agents) # Limit by available share
        mem_needed_per_agent = min(self.base_memory + 4 * task_complexity, 100 / num_agents)

        total_cpu_allocated = 0
        total_memory_allocated = 0
        for agent in agents:
            agent.cpu_allocation = cpu_needed_per_agent
            agent.memory_allocation = mem_needed_per_agent
            total_cpu_allocated += cpu_needed_per_agent
            total_memory_allocated += mem_needed_per_agent

        print(f"Pre-allocated resources: ~{cpu_needed_per_agent:.1f}% CPU, ~{mem_needed_per_agent:.1f}% Mem per agent. Total: {total_cpu_allocated:.1f}% CPU, {total_memory_allocated:.1f}% Mem")
        return {"cpu_total": total_cpu_allocated, "memory_total": total_memory_allocated}


    def update_allocations(self, agents: List[DFSNAgent]):
        """Dynamically update resource allocations based on agent engagement and performance."""
        num_agents = len(agents)
        if num_agents == 0: return

        active_agents = [agent for agent in agents if agent.engagement_state > 0]
        num_active = len(active_agents)

        if num_active == 0: # No active agents, allocate base resources
            for agent in agents:
                agent.cpu_allocation = self.base_cpu
                agent.memory_allocation = self.base_memory
            # print("No active agents. Allocating base resources.")
            return

        # Calculate available resources beyond base allocation for all agents
        available_extra_cpu = self.total_cpu - (self.base_cpu * num_agents)
        available_extra_memory = self.total_memory - (self.base_memory * num_agents)

        # Distribute extra resources among active agents based on performance (simple weighting)
        total_performance_score = sum(agent.performance_history[-1] if agent.performance_history else 0.1 for agent in active_agents) + 1e-8 # Avoid zero division
        if total_performance_score <= 0: total_performance_score = 1e-8 # Ensure positive

        for agent in agents:
            if agent in active_agents:
                perf_score = agent.performance_history[-1] if agent.performance_history else 0.1
                perf_weight = (perf_score / total_performance_score) if total_performance_score > 0 else (1/num_active)

                # Allocate proportional share of extra resources + base
                agent.cpu_allocation = self.base_cpu + available_extra_cpu * perf_weight
                agent.memory_allocation = self.base_memory + available_extra_memory * perf_weight
            else:
                # Inactive agents get base allocation
                agent.cpu_allocation = self.base_cpu
                agent.memory_allocation = self.base_memory

        # Clamp allocations to reasonable bounds (e.g., max 80% per agent)
        for agent in agents:
             agent.cpu_allocation = max(5.0, min(agent.cpu_allocation, 80.0))
             agent.memory_allocation = max(5.0, min(agent.memory_allocation, 80.0))
             # print(f"  {agent.name}: CPU={agent.cpu_allocation:.1f}%, Mem={agent.memory_allocation:.1f}%")

        # print("Updated resource allocations for active agents based on performance.")

    def start(self):
         """Start monitoring (if running in a separate thread/process)."""
         print("Resource monitoring started (simulated).")

    def stop(self):
         """Stop monitoring."""
         print("Resource monitoring stopped (simulated).")

class DynamicFlowStateNetwork:
    """Manages the flow states of a collection of agents based on task complexity."""
    def __init__(self, agents: List[DFSNAgent], task_complexity_threshold: float = 5.0, max_agents: int = 15):
        self.agents = agents # This should be a reference, updated by ZSGManager
        self.task_complexity_threshold = task_complexity_threshold
        self.agent_states: Dict[str, str] = {agent.name: agent.flow_state for agent in agents}
        self.max_agents = max_agents
        self.is_dynamic_enabled = False
        print(f"DFSN initialized. Threshold: {self.task_complexity_threshold}, Max Agents: {self.max_agents}")


    def adjust_flow_states(self, current_task_complexity: float, batch_info: Optional[List] = None):
        """Adjust agent flow states based on complexity and potentially batch info."""
        if not self.is_dynamic_enabled:
            # print("DFSN is disabled. No flow state adjustments.")
            return

        # Determine complexity measure (use batch avg if available)
        complexity_measure = current_task_complexity
        if batch_info and len(batch_info) > 0:
            # Assuming batch_info is a list of ZSGTodo objects or similar dicts with 'priority'
            try:
                batch_avg_complexity = sum(item.priority if hasattr(item, 'priority') else item.get('priority', 5.0) for item in batch_info) / len(batch_info)
                complexity_measure = (current_task_complexity + batch_avg_complexity) / 2 # Average task and batch complexity
                print(f"DFSN using combined complexity: {complexity_measure:.2f} (Task: {current_task_complexity:.2f}, Batch Avg: {batch_avg_complexity:.2f})")
            except Exception as e:
                print(f"DFSN Warning: Could not calculate batch average complexity: {e}")


        # print(f"DFSN adjusting flow states based on complexity measure: {complexity_measure:.2f} (Threshold: {self.task_complexity_threshold})")
        num_flow = 0
        num_idle = 0
        for agent in self.agents:
            # Agents decide their own state via adjust_flow_state called within AIW/_execute_single_task_iteration
            # DFSN can provide global context or override based on system-wide needs
            performance = np.mean(agent.performance_history[-5:]) if agent.performance_history else 0.0
            # The agent's internal optimizer will handle the transition logic
            agent.adjust_flow_state(complexity_measure, performance)
            self.agent_states[agent.name] = agent.flow_state # Update tracked state

            if agent.flow_state == 'flow':
                num_flow += 1
            else:
                num_idle += 1

        print(f"DFSN status: {num_flow} agents in flow, {num_idle} agents idle.")
        self.scale_agents(complexity_measure) # Scale agents after adjustments

    def enable_dynamic_states(self):
        """Enable dynamic adjustments."""
        self.is_dynamic_enabled = True
        print("DFSN enabled for dynamic state adjustments.")

    def disable_dynamic_states(self):
        """Disable dynamic adjustments and reset agents to idle."""
        self.is_dynamic_enabled = False
        for agent in self.agents:
             if agent.flow_state == 'flow':
                 agent.exit_flow_state() # Explicitly exit flow
             self.agent_states[agent.name] = agent.flow_state
        print("DFSN disabled. Agents reset towards idle states.")

    def scale_agents(self, task_complexity: float):
        """Dynamically scale the number of active agents (placeholder)."""
        # Example scaling logic: more complex tasks require more agents active/instantiated
        target_active_agents = max(2, min(self.max_agents, int(task_complexity / 2.0) + 1))
        num_current_agents = len(self.agents)
        num_currently_active = sum(1 for agent in self.agents if agent.engagement_state > 0)

        print(f"DFSN Scaling Check: Complexity={task_complexity:.2f}, TargetActive={target_active_agents}, CurrentTotal={num_current_agents}, CurrentActive={num_currently_active}")

        # Add new agents if needed and below max limit
        if num_current_agents < target_active_agents and num_current_agents < self.max_agents:
            num_to_add = min(target_active_agents - num_current_agents, self.max_agents - num_current_agents)
            print(f"  Scaling up: Adding {num_to_add} new DFSNAgent(s).")
            for i in range(num_to_add):
                 new_agent_name = f"DFSNAgent_{num_current_agents + i}"
                 # This needs interaction with the ZSGManager to actually add the agent
                 # self.manager.add_agent(DFSNAgent, new_agent_name) # Conceptual
                 print(f"    (Conceptual) Added {new_agent_name}")
            # Note: Need a reference to the manager or a callback to add agents properly.

        # Deactivate surplus agents if complexity is low (or handle via engagement states)
        elif num_currently_active > target_active_agents and task_complexity < self.task_complexity_threshold * 0.8:
             num_to_deactivate = num_currently_active - target_active_agents
             print(f"  Scaling down: Deactivating {num_to_deactivate} agent(s) (setting to low engagement).")
             # Find agents to deactivate (e.g., lowest performance or idle)
             agents_to_consider = sorted(self.agents, key=lambda a: (a.engagement_state, np.mean(a.performance_history[-5:]) if a.performance_history else 0))
             for i in range(num_to_deactivate):
                 if i < len(agents_to_consider):
                     agents_to_consider[i].exit_flow_state() # Force lower engagement
                     print(f"    Deactivated {agents_to_consider[i].name}")

    def handle_chaos(self, chaos_metrics: Dict):
         """Adjust DFSN parameters based on chaos metrics."""
         lyapunov_exp = chaos_metrics.get("lyapunov", 0.0)
         print(f"DFSN received chaos metrics: Lyapunov={lyapunov_exp:.3f}")
         if lyapunov_exp > 0.5: # System becoming more chaotic
             print("  High chaos detected. Increasing stability preference (reducing threshold).")
             self.task_complexity_threshold *= 0.95 # Become sensitive to flow at lower complexity
             # Maybe increase learning rate of flow optimizer to adapt faster
             for agent in self.agents:
                 agent.optimizer.learning_rate = min(0.2, agent.optimizer.learning_rate * 1.1)
         elif lyapunov_exp < 0.1: # System very stable
             print("  Low chaos detected. Increasing complexity tolerance (increasing threshold).")
             self.task_complexity_threshold *= 1.05
             for agent in self.agents:
                 agent.optimizer.learning_rate = max(0.05, agent.optimizer.learning_rate * 0.95)


class MultiAgentCoordinator:
    """Coordinates task assignment and synchronization among multiple agents."""
    def __init__(self, agents: List[Agent]):
        self.agents = agents # Reference to the list of agents managed by ZSGManager
        self.task_queue: List[Tuple[int, Dict]] = [] # Priority queue (using heapq, neg priority)
        # Basic resource pool tracking (can be enhanced by ResourceMonitor)
        self.resource_pool = {"cpu": 100.0, "memory": 100.0}
        self.agent_states: Dict[str, Dict] = {agent.name: {"state": "idle", "load": 0.0, "engagement": agent.get_engagement_state()} for agent in agents}
        self.domain_map = self._build_domain_map()
        print(f"MultiAgentCoordinator initialized with {len(agents)} agents.")

    def _build_domain_map(self) -> Dict[str, List[str]]:
        """Builds a map from task types/domains to capable agent names."""
        # This should be dynamic based on agent capabilities
        domain_map = {
            "physics_simulation": ["PhysicsAgent"], # Can match parts of agent names
            "quantum": ["QuantumAgent"], # Matches QuantumAgent_1, QuantumAgent_QA1 etc.
            "memory_task": ["MemoryAgent"],
            "science_fair_experiment": ["CollaborativeAgent"],
            "collaboration": ["CollaborativeAgent"],
            "temporal_forecast": ["TemporalPrimeAgent"],
            "organic_chemistry": ["OrganicChemistryAgent"],
            "molecular_biology": ["MolecularBiologyAgent"],
            "creative": ["CreativeAgent"],
            "information_theory": ["InformationTheoryAgent"],
            "data_science": ["DataScienceAgent"],
            "astrophysics": ["AstrophysicsAgent"],
            "robotics": ["RoboticsAgent"],
            "environmental_science": ["EnvironmentalScienceAgent"],
            "machine_learning": ["MachineLearningAgent"],
            "validation": ["ValidationAgent"],
            "chaos": ["PhysicsAgent", "FractalAgent"], # Example: Multiple agents can handle
            "fractal_generate": ["FractalAgent"],
            "hnn_update": ["HopfieldAgent"],
            "temporal_scaling": ["TemporalPrimeAgent"],
            "llada_task": ["LLaDATaskAgent"],
            "quantum_poetry": ["QuantumAIMLLLM"], # Handled by manager directly? Or a dedicated agent?
            "quantum_game": ["QuantumAgent"],
            "quantum_field": ["QuantumAgent"],
            "grover_search": ["QuantumAgent"],
            "shor_factor": ["QuantumAgent"],
            "quantum_circuit": ["QuantumAgent"],
            # Add more mappings as new agents/tasks are defined
        }
        print("Coordinator domain map built.")
        return domain_map

    def find_capable_agents(self, task_type: str) -> List[Agent]:
         """Finds agents whose names or declared capabilities match the task type."""
         capable_agents = []
         agent_dict = {agent.name: agent for agent in self.agents}

         for domain_key, agent_name_patterns in self.domain_map.items():
             if domain_key in task_type: # Simple substring matching for type
                 for pattern in agent_name_patterns:
                     for agent_name, agent in agent_dict.items():
                         if pattern in agent_name and agent not in capable_agents:
                             capable_agents.append(agent)

         # Fallback if no specific agent found
         if not capable_agents:
             print(f"No specific agent found for task type '{task_type}', assigning to general DFSNAgent.")
             # Find any generic DFSNAgent available
             general_agents = [agent for agent in self.agents if isinstance(agent, DFSNAgent) and not any(pattern in agent.name for patterns in self.domain_map.values() for pattern in patterns)]
             if general_agents:
                 capable_agents.append(random.choice(general_agents)) # Assign to a random general agent
             elif self.agents:
                  capable_agents.append(random.choice(self.agents)) # Assign to any agent if no general one exists

         # print(f"Found {len(capable_agents)} capable agents for task '{task_type}': {[a.name for a in capable_agents]}")
         return capable_agents


    def add_task(self, task: dict, priority: int):
        """Adds a task to the priority queue."""
        if 'type' not in task:
            print("Warning: Task added without 'type'. Assigning low priority.")
            task['type'] = 'unknown'
            priority = -10 # Use negative for min-heap

        print(f"Coordinator adding task: {task['type']} with priority {priority}")
        heappush(self.task_queue, (-priority, task)) # Use negative priority for max-heap behavior

    def assign_tasks(self, task: Dict) -> Dict:
         """Assigns a single task to the most suitable agent(s)."""
         task_type = task.get("type", "unknown")
         capable_agents = self.find_capable_agents(task_type)

         if not capable_agents:
             print(f"Error: No capable agent found for task type '{task_type}'.")
             return {"error": "No suitable agent found", "task_type": task_type}

         # Simple assignment: Assign to the least loaded capable agent
         # More complex: Consider engagement state, specialization score, etc.
         best_agent = min(capable_agents, key=lambda a: self.agent_states.get(a.name, {"load": 0.0})["load"])

         print(f"Assigning task '{task_type}' to agent: {best_agent.name} (Load: {self.agent_states.get(best_agent.name, {}).get('load', 0):.2f})")
         self.agent_states[best_agent.name]["load"] += task.get("complexity", 5.0) # Increment load estimate

         # Execute the task via the agent's _execute_single_task_iteration method
         result = best_agent._execute_single_task_iteration(task)

         # Decrement load after execution (simplified)
         self.agent_states[best_agent.name]["load"] -= task.get("complexity", 5.0)
         self.agent_states[best_agent.name]["load"] = max(0, self.agent_states[best_agent.name]["load"]) # Ensure load doesn't go negative

         # Synchronize states after task completion
         self.synchronize_flow_states({"success": "error" not in result})

         return result

    def execute_next_task(self) -> Optional[Dict]:
        """Retrieves and executes the highest priority task from the queue."""
        if not self.task_queue:
            # print("Coordinator task queue is empty.")
            return None

        neg_priority, task = heappop(self.task_queue)
        priority = -neg_priority
        print(f"Coordinator executing next task (Priority: {priority}): {task.get('type', 'Unknown')}")

        return self.assign_tasks(task) # Use the assign_tasks logic


    def synchronize_flow_states(self, task_feedback: Optional[Dict] = None):
        """Synchronizes agent states, potentially based on feedback or global state."""
        # print("Coordinator synchronizing flow states...")
        if not self.agents: return

        engagement_levels = [agent.get_engagement_state() for agent in self.agents]
        avg_engagement = np.mean(engagement_levels) if engagement_levels else 0
        # print(f"  Average engagement: {avg_engagement:.2f}")

        # Update internal tracking
        for agent in self.agents:
            self.agent_states[agent.name]["engagement"] = agent.get_engagement_state()
            # Optional: Adjust load based on engagement?

        # Basic synchronization: Agents adjust workload based on average (or peers)
        for agent in self.agents:
            # Pass average engagement, agent can decide how to react
             agent.adjust_workload(avg_engagement)

             # Optionally trigger DFSN adjustments based on global state
             # if hasattr(self.manager, 'flow_state_network'): # Requires link to manager
             #     self.manager.flow_state_network.adjust_flow_states(global_complexity_measure)
             pass

        # print("Flow state synchronization complete.")


class MLIW: # Machine Learning Iterative Workflow
    """Manages the episode and iteration state for the MLIW."""
    def __init__(self):
        self.current_episode: int = 0
        self.current_iteration: int = 0
        print("MLIW Controller initialized.")

    def start_episode(self, episode: Optional[int] = None, iteration: Optional[int] = None):
        """Starts or advances the episode/iteration counter."""
        if episode is not None:
            self.current_episode = episode
        else:
            self.current_episode += 1

        if iteration is not None:
            self.current_iteration = iteration
        else:
            self.current_iteration = 1 # Reset iteration for new episode

        print(f"MLIW starting Episode {self.current_episode}, Iteration {self.current_iteration}")

    def next_iteration(self):
        """Advances to the next iteration within the current episode."""
        self.current_iteration += 1
        print(f"MLIW advanced to Episode {self.current_episode}, Iteration {self.current_iteration}")

    def get_state(self) -> Tuple[int, int]:
        """Returns the current episode and iteration."""
        return self.current_episode, self.current_iteration


class ZSGTodo:
    """Represents a task item within the ZSG framework."""
    def __init__(self, task_id: str, description: str, status: str, priority: float, mliw_step: str, data_payload: Dict):
        self.task_id = task_id # Unique ID (e.g., "T001", "QuantumTask_abc")
        self.description = description # E.g., "Optimize PNS sampling", "Run Grover Search"
        self.status = status # "Pending", "In Progress", "Completed", "Failed"
        self.priority = priority # Numerical priority (higher is more important)
        self.mliw_step = mliw_step # Which MLIW phase (e.g., "Analyze", "Modulate", "Test", "Generate", "Validate")
        self.data_payload = data_payload # Input data, parameters, or results needed/produced (dict)

        # Additional fields for tracking, if needed
        self.creation_time = time.time()
        self.assigned_agent: Optional[str] = None
        self.completion_time: Optional[float] = None

    def to_json(self) -> Dict:
        """Serializes the ZSGTodo object into a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "mliw_step": self.mliw_step,
            "data": self.data_payload, # Keep payload under 'data' key for consistency
            "creation_time": self.creation_time,
            "assigned_agent": self.assigned_agent,
            "completion_time": self.completion_time
        }

    def update_status(self, new_status: str, agent_name: Optional[str] = None):
        """Updates the status of the TODO item."""
        self.status = new_status
        if agent_name:
            self.assigned_agent = agent_name
        if new_status in ["Completed", "Failed"]:
            self.completion_time = time.time()
        print(f"TODO {self.task_id} status updated to {new_status}" + (f" by {agent_name}" if agent_name else ""))

# --- Specialized Agent Implementations ---
# PhysicsAgent, OrganicChemistryAgent, MolecularBiologyAgent, CreativeAgent,
# HypothesisAgent, DataScienceAgent, AstrophysicsAgent, RoboticsAgent,
# EnvironmentalScienceAgent, MachineLearningAgent, ValidationAgent,
# FractalAgent, HopfieldAgent, MultiTaskAgent, ContextAwareAgent, MultiModalAgent
# (Keep definitions from previous correct versions, ensuring they override _execute_single_task_iteration)
# ... Example for CollaborativeAgent shown below ...

# --- Physics & Quantum Agents ---

class PhysicsAgent(DFSNAgent):
    """Domain-focused agent for physics simulations."""
    def __init__(self, name: str): # Changed agent_id to name for consistency
        super().__init__(name, math_module=PyZeroMathTorch())
        self.epsilon = self.math_module.epsilon_0 # Use epsilon from math module
        print(f"PhysicsAgent {self.name} initialized.")


    # Overriding the base execute method from DFSNAgent
    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles physics-specific tasks."""
        task_type = task.get("type")
        action = task.get("action")

        required_keys_map = {
            "fluid_dynamics": ["grid_size", "reynolds", "velocity", "pressure", "viscosity", "density"],
            "electromagnetism": ["electric_field", "magnetic_field", "charge_density"],
            "thermodynamics": ["temperature", "thermal_diffusivity", "time_range"]
        }

        if task_type != "physics_simulation":
             return super()._execute_single_task_iteration(task) # Delegate to base if not physics

        if not action:
            return {"error": "Missing 'action' key for physics_simulation task", "agent": self.name}

        keys_needed = required_keys_map.get(action)
        if keys_needed is None:
            return {"error": f"Unsupported physics action: {action}", "agent": self.name}

        error = self.check_task_requirements(task, keys_needed)
        if error:
            return error

        # Execute the specific physics action
        try:
            if action == "fluid_dynamics":
                v = self.f0z_nav_stokes(
                    task["grid_size"], task["reynolds"],
                    task["velocity"], task["pressure"], task["viscosity"], task["density"]
                )
                return {"result": {"velocity": v}, "agent": self.name}
            elif action == "electromagnetism":
                e, b = self.f0z_maxwell(
                    task["electric_field"], task["magnetic_field"], task["charge_density"]
                )
                return {"result": {"electric_field": e, "magnetic_field": b}, "agent": self.name}
            elif action == "thermodynamics":
                temp = self.f0z_heat_equation(
                    task["temperature"], task["thermal_diffusivity"], task["time_range"]
                )
                return {"result": {"temperature": temp}, "agent": self.name}
        except Exception as e:
             print(f"Error during physics execution ({action}) for {self.name}: {e}")
             return {"error": f"Exception during physics execution: {e}", "agent": self.name}

        # Fallback if action somehow wasn't handled
        return {"error": f"Physics action '{action}' not implemented after check.", "agent": self.name}


    def f0z_nav_stokes(self, grid_size, reynolds, velocity, pressure, viscosity, density, time_step=0.01, iterations=10): # Reduced iterations
        """Solve 1D Navier-Stokes using finite differences with F0Z stabilization."""
        dx = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
        v = torch.tensor(velocity, dtype=torch.float32) if isinstance(velocity, (list, np.ndarray)) else velocity * torch.ones(grid_size, dtype=torch.float32)
        p = torch.tensor(pressure, dtype=torch.float32) if isinstance(pressure, (list, np.ndarray)) else pressure * torch.ones(grid_size, dtype=torch.float32)

        for _ in range(iterations):
            dv_dx = torch.zeros_like(v)
            if grid_size > 1:
                dv_dx[1:] = (v[1:] - v[:-1]) / dx
            convective = -v * dv_dx

            dp_dx = torch.zeros_like(p)
            if grid_size > 1:
                dp_dx[1:] = (p[1:] - p[:-1]) / dx
            pressure_term = -dp_dx / (density + self.math_module.epsilon_0) # Stabilize density division

            d2v_dx2 = torch.zeros_like(v)
            if grid_size > 2:
                d2v_dx2[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / (dx ** 2 + self.math_module.epsilon_0) # Stabilize dx^2
            viscous = viscosity * d2v_dx2

            dv_dt = convective + pressure_term + viscous
            v = v + time_step * dv_dt
            v = self.math_module.f0z_stabilize(v, system_size=grid_size)

            # Simplified pressure update (placeholder for projection method)
            if grid_size > 1 :
                 p = p + time_step * density * dv_dx * 0.1 # Damped pseudo-correction

        return v.tolist() # Return as list


    def f0z_maxwell(self, electric_field, magnetic_field, charge_density, time_step=0.01, iterations=10): # Reduced iterations
        """Solve 1D Maxwell's equations using finite differences with F0Z."""
        grid_size = len(electric_field)
        dx = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
        e = torch.tensor(electric_field, dtype=torch.float32)
        b = torch.tensor(magnetic_field, dtype=torch.float32)
        rho = torch.tensor(charge_density, dtype=torch.float32) if isinstance(charge_density, (list, np.ndarray)) else charge_density * torch.ones(grid_size, dtype=torch.float32)

        c = 1.0 # Speed of light

        for _ in range(iterations):
            db_dx = torch.zeros_like(b)
            if grid_size > 1:
                db_dx[1:] = (b[1:] - b[:-1]) / dx
            # Ampere-Maxwell Law: dE/dt = c*curl(B) - J (simplified 1D: curl(B) -> dB/dx, J=rho*v -> approx rho)
            de_dt = c * db_dx - rho # Simplified current term J = rho

            de_dx = torch.zeros_like(e)
            if grid_size > 1:
                de_dx[1:] = (e[1:] - e[:-1]) / dx
            # Faraday's Law: dB/dt = -c*curl(E) (simplified 1D: curl(E) -> dE/dx)
            db_dt = -c * de_dx

            e = e + time_step * de_dt
            b = b + time_step * db_dt

            e = self.math_module.f0z_stabilize(e, system_size=grid_size)
            b = self.math_module.f0z_stabilize(b, system_size=grid_size)

        return e.tolist(), b.tolist()


    def f0z_heat_equation(self, temperature, thermal_diffusivity, time_range, iterations=10): # Reduced iterations
        """Solve 1D heat equation using finite differences with F0Z."""
        grid_size = len(temperature)
        dx = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
        alpha = thermal_diffusivity
        # Ensure stability: dt <= dx^2 / (2 * alpha)
        dt = min(time_range) / iterations if time_range else 0.01 # Simplified dt calculation
        stable_dt = (dx**2 / (2 * alpha + self.math_module.epsilon_0)) * 0.9
        dt = min(dt, stable_dt) if alpha > 0 else dt # Ensure dt respects stability
        if dt <= 0: dt = 0.001 # Fallback dt


        T = torch.tensor(temperature, dtype=torch.float32)

        steps = iterations # Use fixed iterations instead of time_range for simplicity here

        for _ in range(steps):
            d2T_dx2 = torch.zeros_like(T)
            if grid_size > 2:
                d2T_dx2[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / (dx ** 2 + self.math_module.epsilon_0)
            T = T + alpha * dt * d2T_dx2
            T = self.math_module.f0z_stabilize(T, system_size=grid_size)

        return T.tolist()

    def get_system_size(self):
        """Estimate system size for stabilization adjustments."""
        # Example: could be grid size in simulations
        return 100 # Placeholder

    def get_domain_specific_state(self) -> Optional[Dict]:
        # Example: Could share current simulation parameters or field states
        return {"epsilon": self.epsilon.item()} # Share current epsilon

# --- Specialized Agent Implementations (Continued) ---

class QuantumAgent(DFSNAgent):
    """Quantum agent for simulations and quantum algorithms."""
    def __init__(self, name: str): # Removed agent_id, domain - use name and task type
        super().__init__(name)
        self.epsilon = self.math_module.epsilon_0
        # Each QuantumAgent might have its own simulator or share one via the bridge
        # For simplicity, let's assume it uses a bridge provided by the manager
        self.quantum_bridge: Optional[ZSGQuantumBridge] = None # To be set by manager potentially
        print(f"QuantumAgent {self.name} initialized.")

    def set_bridge(self, bridge: ZSGQuantumBridge):
         """Allows the manager to inject the shared quantum bridge."""
         self.quantum_bridge = bridge
         print(f"{self.name} received quantum bridge.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles quantum-specific tasks."""
        if not self.quantum_bridge or not self.quantum_bridge.simulator:
             return {"error": f"{self.name} does not have a configured quantum bridge/simulator.", "agent": self.name}

        task_type = task.get("type")
        action = task.get("action")

        required_keys_map = {
            "quantum_field": ["amplitude", "momentum"],
            "grover_search": ["n_qubits", "target"],
            "shor_factor": ["number", "n_qubits"],
            "quantum_circuit": ["gates", "n_qubits"],
            "quantum_game": ["game_data"] # Example
        }

        # Check if task type suggests quantum work
        if not any(qt in task_type for qt in ["quantum", "grover", "shor", "circuit", "qft"]):
            return super()._execute_single_task_iteration(task) # Delegate if not a quantum task

        if not action:
             # Infer action from type if possible
             if "grover" in task_type: action = "grover_search"
             elif "shor" in task_type: action = "shor_factor"
             elif "circuit" in task_type: action = "quantum_circuit"
             elif "field" in task_type: action = "quantum_field"
             elif "game" in task_type: action = "quantum_game"
             else: return {"error": "Missing 'action' key for quantum task", "agent": self.name}
             task['action'] = action # Add inferred action back to task dict

        keys_needed = required_keys_map.get(action)
        if keys_needed is None:
            return {"error": f"Unsupported quantum action: {action}", "agent": self.name}

        error = self.check_task_requirements(task, keys_needed)
        if error:
            return error

        # Ensure simulator size matches task requirements if specified
        n_qubits_req = task.get("n_qubits")
        if n_qubits_req and n_qubits_req != self.quantum_bridge.simulator.n_qubits:
             # This ideally requires recreating the simulator or bridge, which is complex.
             # For now, return an error or warning.
             print(f"Warning: Task requires {n_qubits_req} qubits, but simulator has {self.quantum_bridge.simulator.n_qubits}. Results may be incorrect.")
             # Or: return {"error": f"Simulator qubit count mismatch", "agent": self.name}


        # Execute the specific quantum action
        try:
            if action == "quantum_field":
                # Note: f0z_quantum_field was defined inside QuantumAgent in the prompt, but doesn't use quantum gates directly.
                # It seems more like a classical simulation using F0Z. Let's keep it here but note its nature.
                state = self.f0z_quantum_field_simulation(task["amplitude"], task["momentum"])
                return {"result": {"state": state}, "agent": self.name, "task_type": "quantum_field_simulation"} # Clarify it's a simulation

            elif action == "grover_search":
                 if task["target"] >= 2**self.quantum_bridge.simulator.n_qubits:
                      return {"error": f"Grover target {task['target']} out of range for {self.quantum_bridge.simulator.n_qubits} qubits.", "agent": self.name}
                 result = self.f0z_grover_search(task["n_qubits"], task["target"]) # n_qubits check done above
                 return {"result": result, "agent": self.name}

            elif action == "shor_factor":
                 result = self.f0z_shor_factor(task["number"], task["n_qubits"])
                 return {"result": result, "agent": self.name}

            elif action == "quantum_circuit":
                 final_state = self.f0z_quantum_circuit(task["gates"], task["n_qubits"])
                 # Decode the final state for a meaningful result
                 decoded = self.quantum_bridge.decode(quantum_state=final_state.tolist())
                 return {"result": {"final_state": final_state.tolist(), "decoded_info": decoded}, "agent": self.name}

            elif action == "quantum_game":
                 # Example game logic: encode data, evolve state, decode result
                 initial_state = self.quantum_bridge.encode(task["game_data"])
                 # Apply some game-specific gates (e.g., random rotations)
                 for q in range(self.quantum_bridge.simulator.n_qubits):
                     self.quantum_bridge.simulator.ry_gate(random.random() * np.pi, q)
                 game_result_state = self.quantum_bridge.simulator.state.tolist()
                 decoded_outcome = self.quantum_bridge.decode(quantum_state=game_result_state)
                 return {"result": {"game_outcome": decoded_outcome}, "agent": self.name}

        except Exception as e:
             print(f"Error during quantum execution ({action}) for {self.name}: {e}")
             # traceback.print_exc() # Uncomment for detailed debugging
             return {"error": f"Exception during quantum execution: {e}", "agent": self.name}

        # Fallback
        return {"error": f"Quantum action '{action}' not implemented.", "agent": self.name}


    # --- Quantum Algorithm Implementations (using self.quantum_bridge.simulator) ---

    def f0z_quantum_field_simulation(self, field_amplitude, momentum, time_step=0.01, iterations=10):
        """Simulate 1D quantum field evolution classically using F0Z (moved from initial prompt)."""
        sim_state_vector = np.array([0.0]) # Placeholder, this needs a proper QFT simulation setup

        try:
            N = 64 # Discretized grid size
            dx = 1.0 / (N - 1) if N > 1 else 1.0
            mass = 1.0

            # Initial field configuration
            phi = np.array(field_amplitude) if isinstance(field_amplitude, (list, np.ndarray)) else field_amplitude * np.ones(N)
            pi = np.array(momentum) if isinstance(momentum, (list, np.ndarray)) else momentum * np.ones(N)

            # Represent state (simplified: using classical arrays, not qubit state vector)
            current_phi = phi
            current_pi = pi

            for _ in range(iterations):
                 # Simplified update using F0Z-stabilized derivatives (conceptual)
                 d2phi_dx2 = F0ZAlgebra.f0z_gradient(F0ZAlgebra.f0z_gradient(current_phi)) # Approx Laplacian
                 # Klein-Gordon equation: d^2phi/dt^2 = d^2phi/dx^2 - m^2*phi
                 dphi_dt = current_pi
                 dpi_dt = d2phi_dx2 - (mass**2) * current_phi

                 # Stabilize updates
                 dphi_dt_s = self.math_module.f0z_stabilize(torch.tensor(dphi_dt)).numpy()
                 dpi_dt_s = self.math_module.f0z_stabilize(torch.tensor(dpi_dt)).numpy()

                 current_phi = current_phi + time_step * dphi_dt_s
                 current_pi = current_pi + time_step * dpi_dt_s

                 # Stabilize state fields
                 current_phi = self.math_module.f0z_stabilize(torch.tensor(current_phi)).numpy()
                 current_pi = self.math_module.f0z_stabilize(torch.tensor(current_pi)).numpy()


            sim_state_vector = current_phi + 1j * current_pi # Combine into complex field value


        except Exception as e:
            print(f"Error in f0z_quantum_field_simulation: {e}")
            return {"error": str(e)}

        return {"phi": current_phi.tolist(), "pi": current_pi.tolist()} # Return field components


    def f0z_grover_search(self, n_qubits, target):
        """Grover's quantum search algorithm using the simulator."""
        sim = self.quantum_bridge.simulator
        if n_qubits != sim.n_qubits:
             print(f"Warning: Grover using simulator's {sim.n_qubits} qubits, not requested {n_qubits}.")
             # Ideally, resize simulator or throw error
        N = 2**sim.n_qubits

        # 1. Initial Superposition
        sim.state.fill(0)
        sim.state[0] = 1.0 # Start from |0>
        for q in range(sim.n_qubits):
             sim.h_gate(q)

        # Oracle function for FDO gate
        def oracle_func(index):
            return 1 if index == target else 0

        # Number of iterations optimal for Grover
        steps = int(np.pi * np.sqrt(N) / 4.0) if N > 0 else 0
        print(f"  Grover: Running {steps} iterations for {sim.n_qubits} qubits.")

        for _ in range(steps):
             # 2. Apply Oracle (using FDO)
             sim.fdo(oracle_func)
             # 3. Apply Diffusion Operator
             # Diffusion = H^{\otimes n} (2|0><0| - I) H^{\otimes n}
             # Simplified implementation via sim.diffusion() assumed correct
             for q in range(sim.n_qubits): sim.h_gate(q)
             sim.fdo(lambda i: 1 if i == 0 else 0) # 2|0><0| - I equivalent phase flip
             for q in range(sim.n_qubits): sim.h_gate(q)
             # Apply F0Z stabilization after each full step
             sim.state = self.math_module.f0z_stabilize(torch.tensor(sim.state, dtype=torch.complex64), system_size=N).numpy()


        # 4. Measure
        probs = np.abs(sim.state)**2
        probs = self.math_module.f0z_stabilize(torch.tensor(probs)).numpy()
        probs /= probs.sum()

        found_prob = probs[target] if target < len(probs) else 0
        print(f"  Grover: Probability of target state |{target}> = {found_prob:.4f}")
        return {"final_probs": probs.tolist(), "target": target, "target_probability": found_prob}


    def f0z_shor_factor(self, number, n_qubits):
        """Shor's algorithm period finding (simplified simulation)."""
        sim = self.quantum_bridge.simulator
        if n_qubits != sim.n_qubits:
             print(f"Warning: Shor using simulator's {sim.n_qubits} qubits, not requested {n_qubits}.")
             # Need enough qubits for the number N. Let 2^n_qubits >= number^2
             required_qubits = int(np.ceil(2 * np.log2(number)))
             if sim.n_qubits < required_qubits:
                  return {"error": f"Shor needs at least {required_qubits} qubits for N={number}, simulator has {sim.n_qubits}"}

        N = number # The number to factor
        # Classical part: Choose random 'a', check GCD
        a = random.randint(2, N - 1)
        g = np.gcd(a, N)
        if g != 1:
            print(f"  Shor: Lucky guess! Found factor classically: {g}")
            return {"factors": [g, N // g], "period": None, "base_a": a}

        print(f"  Shor: Trying to factor {N} with base a={a}")

        # Quantum Part: Period Finding
        # 1. Initialize registers (simulate using full state vector)
        # Requires two registers, but we simulate the combined state evolution.
        # Let simulator represent the second register (result of modular exponentiation).
        # First register (input) state is implicitly handled by QFT later.
        sim_size = sim.n_qubits
        M = 2**sim_size # Size of simulator state space

        # State preparation (|0> in first reg, |1> in second reg mapped implicitly)
        sim.state.fill(0)
        sim.state[1 % M] = 1.0 # Start second register in |1> (index 1)

        # 2. Apply Quantum Modular Exponentiation (simulated classically)
        # U|x>|y> = |x>|y * a^x mod N>
        # We simulate the effect on the second register for an implicit superposition in the first.
        # This is a major simplification. A real simulation applies QFT to the first register.
        # Let's simulate the QFT result directly.
        print("  Shor: Simulating Quantum Fourier Transform result...")
        # The QFT peaks at multiples of M/r, where r is the period.
        # Need to find the period 'r' classically first for this simulation.
        r = 1
        while pow(a, r, N) != 1:
            r += 1
        print(f"  Shor: Found period r={r} classically for simulation.")

        if r % 2 != 0:
             print("  Shor: Period r is odd. Algorithm fails for this 'a'.")
             return {"period": r, "factors": "Failed (odd period)", "base_a": a}

        # Simulate QFT measurement outcome probabilities peaking near k*M/r
        sim.state.fill(0)
        for k in range(r): # Superposition of peaks
             peak_index = int(round(k * M / r)) % M
             sim.state[peak_index] = 1.0 / np.sqrt(r) # Equal amplitude for peaks

        sim.state = self.math_module.f0z_stabilize(torch.tensor(sim.state, dtype=torch.complex64), system_size=M).numpy()
        probs = np.abs(sim.state)**2

        # 3. Measure (get a peak related to the period)
        measured_val = np.random.choice(M, p=(probs/probs.sum())) # Measure one outcome
        print(f"  Shor: Simulated measurement = {measured_val} (Expecting peak near k*M/r)")

        # 4. Classical Post-processing (Continued Fraction Algorithm - simplified)
        # Try to deduce r from measurement c = k*M/r
        # This part is complex, we'll use the known 'r' for now to find factors
        print(f"  Shor: Using known period r={r} for post-processing.")
        factor1 = np.gcd(pow(a, r//2, N) - 1, N)
        factor2 = np.gcd(pow(a, r//2, N) + 1, N)

        factors = []
        if factor1 != 1 and factor1 != N: factors.append(factor1)
        if factor2 != 1 and factor2 != N: factors.append(factor2)
        # Add the corresponding co-factors
        if factors:
             cofactor = N // factors[0]
             if cofactor != 1 and cofactor != N and cofactor not in factors:
                  factors.append(cofactor)
             factors = sorted(list(set(factors))) # Unique sorted factors

        if factors:
             print(f"  Shor: Found factors: {factors}")
             return {"period": r, "factors": factors, "base_a": a}
        else:
             print("  Shor: Failed to find non-trivial factors from this 'a' and 'r'.")
             return {"period": r, "factors": "Failed (no non-trivial factors found)", "base_a": a}


    def f0z_quantum_circuit(self, gates: List[Dict], n_qubits: int):
        """Executes a quantum circuit defined by a list of gates."""
        sim = self.quantum_bridge.simulator
        if n_qubits != sim.n_qubits:
            print(f"Warning: Circuit using simulator's {sim.n_qubits} qubits, not requested {n_qubits}.")

        # Initialize state to |0...0>
        sim.state.fill(0)
        sim.state[0] = 1.0

        print(f"  Executing quantum circuit with {len(gates)} gates on {sim.n_qubits} qubits...")
        for i, gate_info in enumerate(gates):
            gate_type = gate_info.get("type", "").upper()
            try:
                if gate_type == "H":
                    sim.h_gate(gate_info["qubits"][0])
                elif gate_type == "X":
                    sim.x_gate(gate_info["qubits"][0])
                elif gate_type == "Y":
                     sim.y_gate(gate_info["qubits"][0])
                elif gate_type == "Z":
                     sim.z_gate(gate_info["qubits"][0])
                elif gate_type == "RZ":
                    sim.rz_gate(gate_info["angle"], gate_info["qubit"])
                elif gate_type == "RY":
                    sim.ry_gate(gate_info["angle"], gate_info["qubit"])
                elif gate_type == "CNOT" or gate_type == "CX":
                    sim.cnot_gate(gate_info["control"], gate_info["target"])
                elif gate_type == "CZ":
                    sim.cz_gate(gate_info["control"], gate_info["target"])
                else:
                    print(f"    Warning: Skipping unknown gate type '{gate_type}' at step {i}.")
                    continue

                # Apply F0Z stabilization periodically or after each gate
                if (i + 1) % 5 == 0 or i == len(gates) - 1: # Stabilize every 5 gates or at the end
                    sim.state = self.math_module.f0z_stabilize(torch.tensor(sim.state, dtype=torch.complex64), system_size=2**sim.n_qubits).numpy()
                    # Optional: Normalize after stabilization if needed
                    norm = LA.norm(sim.state)
                    if norm > 1e-9 : sim.state /= norm

            except KeyError as e:
                 print(f"    Error: Missing key {e} in gate definition at step {i}: {gate_info}")
                 raise # Re-raise the error to stop execution
            except Exception as e:
                 print(f"    Error applying gate at step {i} ({gate_type}): {e}")
                 raise # Re-raise

        print("  Quantum circuit execution complete.")
        return sim.state # Return the final state vector


    def share_state(self, peer: Agent):
        """Share quantum state vector with a peer."""
        if not isinstance(peer, QuantumAgent) or not self.quantum_bridge:
            print(f"{self.name} cannot share quantum state with {peer.name}")
            return

        state_vector_list = self.quantum_bridge.simulator.state.tolist() if self.quantum_bridge.simulator else None
        state_to_share = {
            "performance_history": self.performance_history[-10:],
            "domain_data": {"quantum_state_vector": state_vector_list}
        }
        print(f"{self.name} sharing quantum state ({len(state_vector_list)} amplitudes) with {peer.name}")
        peer.receive_state(state_to_share)


    def receive_state(self, state: Dict):
        """Receive and potentially update quantum state from a peer."""
        super().receive_state(state) # Handle performance history etc.
        if self.quantum_bridge and "domain_data" in state and "quantum_state_vector" in state["domain_data"]:
            received_state = state["domain_data"]["quantum_state_vector"]
            if received_state is not None and len(received_state) == 2**self.quantum_bridge.simulator.n_qubits:
                # Option 1: Overwrite state
                # self.quantum_bridge.simulator.state = np.array(received_state, dtype=complex)
                # Option 2: Average states (careful, may not be physically meaningful)
                current_state = self.quantum_bridge.simulator.state
                avg_state = 0.5 * (current_state + np.array(received_state, dtype=complex))
                norm = LA.norm(avg_state)
                if norm > 1e-9 : self.quantum_bridge.simulator.state = avg_state / norm

                print(f"{self.name} updated quantum state based on peer (averaged).")
            else:
                 print(f"{self.name} received invalid quantum state from peer.")


# --- Multi-Task and Contextual Agents (Refined) ---

class MultiTaskAgent(DFSNAgent):
    """Agent capable of handling multiple predefined tasks based on state."""
    def __init__(self, name: str, task1: Any, task2: Any): # Tasks can be objects with a 'run' method
        super().__init__(name)
        self.task1 = task1
        self.task2 = task2
        print(f"MultiTaskAgent {self.name} initialized with tasks: {type(task1).__name__}, {type(task2).__name__}")

    # Override _execute_single_task_iteration to handle the multi-task logic
    def _execute_single_task_iteration(self, task: Optional[Dict]=None) -> Dict: # Task dict might be unused if tasks are predefined
        """Executes one of the predefined tasks based on engagement state."""
        print(f"{self.name} deciding task based on engagement state: {self.engagement_state}")
        # Use AIW to manage iterations if needed, but select task first
        if self.engagement_state > 3:
             print(f"  Engagement > 3, executing Task 1 ({type(self.task1).__name__})")
             if hasattr(self.task1, 'run'):
                 result = self.task1.run()
                 # AIW needs a task dict, we simulate one
                 simulated_task = {"type": "multitask_task1", "complexity": 6}
                 # We need to wrap the execution within the AIW framework if we want iterative refinement
                 # This structure is a bit awkward. Maybe AIW should take the function to run?
                 # Simplified: Just run the task directly and return result.
                 return {"result": result, "agent": self.name, "executed_task": "task1"}
             else: return {"error": "Task 1 has no run method", "agent": self.name}
        else:
             print(f"  Engagement <= 3, executing Task 2 ({type(self.task2).__name__})")
             if hasattr(self.task2, 'run'):
                 result = self.task2.run()
                 return {"result": result, "agent": self.name, "executed_task": "task2"}
             else: return {"error": "Task 2 has no run method", "agent": self.name}

class ContextAwareAgent(DFSNAgent):
    """Agent that processes sequences with context awareness."""
    def __init__(self, name: str, context_window: int = 5):
        super().__init__(name)
        self.context_window = context_window
        print(f"ContextAwareAgent {self.name} initialized with window size {self.context_window}.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles tasks requiring context-aware processing."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "contextual_processing":
            if action == "process_sequence":
                error = self.check_task_requirements(task, ["sequence"])
                if error: return error
                processed_tokens = self.process_tokens(task["sequence"])
                return {"result": {"processed_tokens_count": len(processed_tokens)}, "agent": self.name}

        return super()._execute_single_task_iteration(task) # Delegate


    def process_tokens(self, token_sequence: List[str]) -> List[Dict]:
        """Processes a sequence of tokens, considering context."""
        print(f"Processing token sequence (length {len(token_sequence)}) with context window {self.context_window}...")
        processed_results = []
        for i, token in enumerate(token_sequence):
            # Determine context window [start, end)
            start_index = max(0, i - self.context_window)
            # Context includes current token up to index i
            current_context = token_sequence[start_index : i+1]
            # Handle the token with its context
            processed_info = self.handle_token_with_context(token, current_context)
            processed_results.append({"token": token, "context": current_context, "info": processed_info})

        print("Token sequence processing complete.")
        # Adjust flow state based on sequence complexity (e.g., length, variability)
        complexity = len(token_sequence) / 10.0 # Simple complexity measure
        perf = len(processed_results) / len(token_sequence) if token_sequence else 0.0 # Performance = completion rate
        self.adjust_flow_state(complexity, perf)

        return processed_results

    def handle_token_with_context(self, token: str, context: List[str]) -> str:
        """Placeholder logic for handling a token given its context."""
        # Example: Identify if token is novel in context or follows a pattern
        context_str = " ".join(context[:-1]) # Context before current token
        # print(f"  Processing token: '{token}' with context: '[...{context_str[-30:]}]'")
        if token in context[:-1]:
             return "repeated_in_context"
        elif len(context) > 2 and token.isdigit() and context[-2].isdigit():
             return "continuation_of_numbers"
        else:
             return "standard_processing"


class MultiModalAgent(DFSNAgent):
    """Agent capable of processing inputs from multiple modalities."""
    def __init__(self, name: str, modalities: List[str]):
        super().__init__(name)
        self.modalities = modalities
        print(f"MultiModalAgent {self.name} initialized. Handles: {', '.join(modalities)}")
        # Placeholder for modality-specific models (e.g., image classifier, text processor)
        self.text_processor = lambda text: f"Processed text: {text[:30]}..."
        self.image_processor = lambda img: f"Processed image of shape: {img.shape}" if isinstance(img, np.ndarray) else "Processed non-numpy image"
        self.audio_processor = lambda aud: f"Processed audio data (length {len(aud)} samples)" if isinstance(aud, (list, np.ndarray)) else "Processed non-array audio"
        self.video_processor = lambda vid: f"Processed video data (frames: {len(vid)})" if isinstance(vid, (list, np.ndarray)) else "Processed non-array video"


    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles multimodal processing tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "multimodal_input":
             error = self.check_task_requirements(task, ["input_data"]) # Input data is expected to be a dict
             if error: return error
             if not isinstance(task["input_data"], dict):
                 return {"error": "input_data must be a dictionary for multimodal agent", "agent": self.name}

             results = self.process_input(task["input_data"])
             # Adjust flow based on complexity (e.g., number of modalities, data size)
             complexity = len(task["input_data"].keys()) # Simple: complexity = number of modalities present
             perf = 1.0 # Assume success if no errors
             self.adjust_flow_state(complexity, perf)
             return {"result": results, "agent": self.name}

        return super()._execute_single_task_iteration(task) # Delegate

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Processes input data based on supported modalities."""
        print(f"{self.name} processing multimodal input...")
        results = {}
        for modality in self.modalities:
            data = input_data.get(modality) # Get data for the modality
            if data is not None:
                print(f"  Processing modality: {modality}")
                handler = getattr(self, f"handle_{modality}", None)
                if handler and callable(handler):
                    try:
                         result_info = handler(data)
                         results[modality] = result_info
                    except Exception as e:
                         print(f"    Error processing {modality}: {e}")
                         results[modality] = f"Error: {e}"
                else:
                     print(f"    No handler found for modality: {modality}")
                     results[modality] = "No handler available"
            # else: print(f"  Modality '{modality}' not present in input.")
        print("Multimodal processing complete.")
        return results

    # --- Modality Handlers ---
    def handle_text(self, text_data: str) -> str:
        # print(f"  Handling text data...")
        return self.text_processor(text_data)

    def handle_image(self, image_data: Any) -> str:
        # print(f"  Handling image data...")
        # Add actual image processing logic here
        return self.image_processor(image_data)

    def handle_audio(self, audio_data: Any) -> str:
        # print(f"  Handling audio data...")
        return self.audio_processor(audio_data)

    def handle_video(self, video_data: Any) -> str:
        # print(f"  Handling video data...")
        return self.video_processor(video_data)


# --- Other Specialized Agents ---

class MemoryAgent(DFSNAgent):
    """Agent focused on managing and utilizing long-term memory."""
    def __init__(self, name: str):
        super().__init__(name)
        self.memory_system = MemorySystem() # Each agent could have its own or share one
        print(f"MemoryAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles memory-related tasks or uses memory for other tasks."""
        action = task.get("action")
        task_type = task.get("type")

        if task_type == "memory_task":
            if action == "store":
                 error = self.check_task_requirements(task, ["key", "data", "memory_type"])
                 if error: return error
                 self.memory_system.store_memory(task["data"], task["memory_type"], task["key"])
                 return {"status": "stored", "agent": self.name}
            elif action == "retrieve":
                 error = self.check_task_requirements(task, ["key", "memory_type"])
                 if error: return error
                 data = self.memory_system.retrieve_memory(task["memory_type"], task["key"])
                 return {"result": data, "status": "retrieved", "agent": self.name}
            elif action == "manage_long_horizon":
                 # Example from prompt: process steps using memory
                 error = self.check_task_requirements(task, ["steps"])
                 if error: return error
                 results = self.manage_long_horizon_task(task["steps"])
                 return {"result": results, "agent": self.name}

        # If not a memory task, use AIW (which calls base__execute_single_task_iteration if not overridden)
        return super()._execute_single_task_iteration(task)

    def base_execute_task(self, task: Dict) -> Dict:
         """Base execution for MemoryAgent - maybe retrieves relevant memory first."""
         print(f"{self.name} (MemoryAgent Base): Executing {task.get('type')}")
         # Example: Retrieve memory related to the task description
         relevant_mem = self.memory_system.retrieve_memory(
             memory_type='long',
             criteria=lambda item: isinstance(item, dict) and task.get('description', '') in item.get('description', '')
         )
         if relevant_mem:
             print(f"  Found relevant memory: {relevant_mem}")
             # Use memory in task execution (placeholder)
             task['context_from_memory'] = relevant_mem

         # Simulate work
         time.sleep(0.02 * task.get('complexity', 1.0))
         return {"result": f"Memory-enhanced execution for {task.get('type')}", "agent": self.name}


    def manage_long_horizon_task(self, steps: List[Dict]) -> List[Dict]:
        """Processes a sequence of steps, using memory when required."""
        step_results = []
        print(f"{self.name} managing long horizon task with {len(steps)} steps.")
        for i, step in enumerate(steps):
             print(f"  Step {i+1}: {step.getexecute_single_task_iteration}(task)") # Delegate


    def process_tokens(self, token_sequence: List[str]) -> List[Dict]:
        """Processes a sequence of tokens, considering context."""
        print(f"Processing token sequence (length {len(token_sequence)}) with context window {self.context_window}...")
        processed_results = []
        for i, token in enumerate(token_sequence):
            # Determine context window [start, end)
            start_index = max(0, i - self.context_window)
            # Context includes current token up to index i
            current_context = token_sequence[start_index : i+1]
            # Handle the token with its context
            processed_info = self.handle_token_with_context(token, current_context)
            processed_results.append({"token": token, "context": current_context, "info": processed_info})

        print("Token sequence processing complete.")
        # Adjust flow state based on sequence complexity (e.g., length, variability)
        complexity = len(token_sequence) / 10.0 # Simple complexity measure
        perf = len(processed_results) / len(token_sequence) if token_sequence else 0.0 # Performance = completion rate
        self.adjust_flow_state(complexity, perf)

        return processed_results

    def handle_token_with_context(self, token: str, context: List[str]) -> str:
        """Placeholder logic for handling a token given its context."""
        # Example: Identify if token is novel in context or follows a pattern
        context_str = " ".join(context[:-1]) # Context before current token
        # print(f"  Processing token: '{token}' with context: '[...{context_str[-30:]}]'")
        if token in context[:-1]:
             return "repeated_in_context"
        elif len(context) > 2 and token.isdigit() and context[-2].isdigit():
             return "continuation_of_numbers"
        else:
             return "standard_processing"


class MultiModalAgent(DFSNAgent):
    """Agent capable of processing inputs from multiple modalities."""
    def __init__(self, name: str, modalities: List[str]):
        super().__init__(name)
        self.modalities = modalities
        print(f"MultiModalAgent {self.name} initialized. Handles: {', '.join(modalities)}")
        # Placeholder for modality-specific models (e.g., image classifier, text processor)
        self.text_processor = lambda text: f"Processed text: {text[:30]}..."
        self.image_processor = lambda img: f"Processed image of shape: {img.shape}" if isinstance(img, np.ndarray) else "Processed non-numpy image"
        self.audio_processor = lambda aud: f"Processed audio data (length {len(aud)} samples)" if isinstance(aud, (list, np.ndarray)) else "Processed non-array audio"
        self.video_processor = lambda vid: f"Processed video data (frames: {len(vid)})" if isinstance(vid, (list, np.ndarray)) else "Processed non-array video"


    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles multimodal processing tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "multimodal_input":
             error = self.check_task_requirements(task, ["input_data"]) # Input data is expected to be a dict
             if error: return error
             if not isinstance(task["input_data"], dict):
                 return {"error": "input_data must be a dictionary for multimodal agent", "agent": self.name}

             results = self.process_input(task["input_data"])
             # Adjust flow based on complexity (e.g., number of modalities, data size)
             complexity = len(task["input_data"].keys()) # Simple: complexity = number of modalities present
             perf = 1.0 # Assume success if no errors
             self.adjust_flow_state(complexity, perf)
             return {"result": results, "agent": self.name}

        return super()._execute_single_task_iteration(task) # Delegate

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Processes input data based on supported modalities."""
        print(f"{self.name} processing multimodal input...")
        results = {}
        for modality in self.modalities:
            data = input_data.get(modality) # Get data for the modality
            if data is not None:
                print(f"  Processing modality: {modality}")
                handler = getattr(self, f"handle_{modality}", None)
                if handler and callable(handler):
                    try:
                         result_info = handler(data)
                         results[modality] = result_info
                    except Exception as e:
                         print(f"    Error processing {modality}: {e}")
                         results[modality] = f"Error: {e}"
                else:
                     print(f"    No handler found for modality: {modality}")
                     results[modality] = "No handler available"
            # else: print(f"  Modality '{modality}' not present in input.")
        print("Multimodal processing complete.")
        return results

    # --- Modality Handlers ---
    def handle_text(self, text_data: str) -> str:
        # print(f"  Handling text data...")
        return self.text_processor(text_data)

    def handle_image(self, image_data: Any) -> str:
        # print(f"  Handling image data...")
        # Add actual image processing logic here
        return self.image_processor(image_data)

    def handle_audio(self, audio_data: Any) -> str:
        # print(f"  Handling audio data...")
        return self.audio_processor(audio_data)

    def handle_video(self, video_data: Any) -> str:
        # print(f"  Handling video data...")
        return self.video_processor(video_data)


# --- Other Specialized Agents ---

class MemoryAgent(DFSNAgent):
    """Agent focused on managing and utilizing long-term memory."""
    def __init__(self, name: str):
        super().__init__(name)
        self.memory_system = MemorySystem() # Each agent could have its own or share one
        print(f"MemoryAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles memory-related tasks or uses memory for other tasks."""
        action = task.get("action")
        task_type = task.get("type")

        if task_type == "memory_task":
            if action == "store":
                 error = self.check_task_requirements(task, ["key", "data", "memory_type"])
                 if error: return error
                 self.memory_system.store_memory(task["data"], task["memory_type"], task["key"])
                 return {"status": "stored", "agent": self.name}
            elif action == "retrieve":
                 error = self.check_task_requirements(task, ["key", "memory_type"])
                 if error: return error
                 data = self.memory_system.retrieve_memory(task["memory_type"], task["key"])
                 return {"result": data, "status": "retrieved", "agent": self.name}
            elif action == "manage_long_horizon":
                 # Example from prompt: process steps using memory
                 error = self.check_task_requirements(task, ["steps"])
                 if error: return error
                 results = self.manage_long_horizon_task(task["steps"])
                 return {"result": results, "agent": self.name}

        # If not a memory task, use AIW (which calls base__execute_single_task_iteration if not overridden)
        return super()._execute_single_task_iteration(task)

    def base_execute_task(self, task: Dict) -> Dict:
         """Base execution for MemoryAgent - maybe retrieves relevant memory first."""
         print(f"{self.name} (MemoryAgent Base): Executing {task.get('type')}")
         # Example: Retrieve memory related to the task description
         relevant_mem = self.memory_system.retrieve_memory(
             memory_type='long',
             criteria=lambda item: isinstance(item, dict) and task.get('description', '') in item.get('description', '')
         )
         if relevant_mem:
             print(f"  Found relevant memory: {relevant_mem}")
             # Use memory in task execution (placeholder)
             task['context_from_memory'] = relevant_mem

         # Simulate work
         time.sleep(0.02 * task.get('complexity', 1.0))
         return {"result": f"Memory-enhanced execution for {task.get('type')}", "agent": self.name}


    def manage_long_horizon_task(self, steps: List[Dict]) -> List[Dict]:
        """Processes a sequence of steps, using memory when required."""
        step_results = []
        print(f"{self.name} managing long horizon task with {len(steps)} steps.")
        for i, step in enumerate(steps):
             print(f"  Step {i+1}: {step.get('description', 'No description')}")
             step_result = {}
             if step.get('requires_memory', False):
                 # Retrieve most relevant recent memory (e.g., based on description)
                 memory_key = step.get("memory_key")
                 retrieved_data = self.memory_system.retrieve_memory(memory_type='short', key=memory_key)
                 if retrieved_data:
                     print(f"    Retrieved memory for key '{memory_key}'. Applying to task.")
                     # Apply memory to task (e.g., add as context)
                     step_result = self.process_task_step(step, memory_context=retrieved_data)
                 else:
                     print(f"    Memory required but key '{memory_key}' not found. Proceeding without.")
                     step_result = self.process_task_step(step)
             else:
                 step_result = self.process_task_step(step)

             # Store result of step in short-term memory if needed for next steps
             if step.get("output_key"):
                  self.memory_system.store_memory(step_result.get("result"), memory_type='short', key=step.get("output_key"))

             step_results.append(step_result)
        return step_results

    def process_task_step(self, step_data: Dict, memory_context: Optional[Any] = None) -> Dict:
        """Simulates processing a single step of a long-horizon task."""
        # Placeholder: actual processing depends on step description/type
        print(f"    Processing step: {step_data.get('description', '')}" + (f" with memory context." if memory_context else ""))
        result_value = f"Processed '{step_data.get('description', '')[:20]}...'"
        if memory_context:
             result_value += " (used memory)"
        time.sleep(0.01) # Simulate work
        return {"result": result_value, "status": "completed"}

    def get_domain_specific_state(self) -> Optional[Dict]:
        # Share recent memory keys or stats
        return {"recent_short_term_keys": list(self.memory_system.short_term_memory.keys())[-5:],
                "long_term_size": len(self.memory_system.long_term_memory)}

    def process_domain_specific_state(self, domain_data: Dict):
        # Could potentially pre-fetch memory based on peer's recent keys
        pass


class OrganicChemistryAgent(DFSNAgent):
    """Domain-focused agent for organic chemistry simulations."""
    def __init__(self, name: str):
        super().__init__(name)
        self.epsilon = self.math_module.epsilon_0
        self.reaction_rates = {} # Store known reaction rates
        print(f"OrganicChemistryAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles organic chemistry tasks like reaction kinetics."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type != "organic_chemistry":
            return super()._execute_single_task_iteration(task)

        required_keys_map = {
            "reaction_kinetics": ["reactants", "rate_constant", "temperature"],
            "bond_energy": ["bond_strength", "distance"]
        }

        if not action:
            return {"error": "Missing 'action' key for organic_chemistry task", "agent": self.name}

        keys_needed = required_keys_map.get(action)
        if keys_needed is None:
            return {"error": f"Unsupported organic chemistry action: {action}", "agent": self.name}

        error = self.check_task_requirements(task, keys_needed)
        if error:
            return error

        try:
            if action == "reaction_kinetics":
                # Assume 'reactants' is a concentration value or dict
                conc = task["reactants"] if isinstance(task["reactants"], (int, float)) else np.mean(list(task["reactants"].values())) if isinstance(task["reactants"], dict) else 1.0
                rate = self.f0z_reaction_kinetics(conc, task["rate_constant"], task["temperature"])
                return {"result": {"reaction_rate": rate}, "agent": self.name}
            elif action == "bond_energy":
                energy = self.f0z_bond_energy(task["bond_strength"], task["distance"])
                return {"result": {"bond_energy": energy}, "agent": self.name}
        except Exception as e:
             print(f"Error during organic chemistry execution ({action}) for {self.name}: {e}")
             return {"error": f"Exception during organic chem execution: {e}", "agent": self.name}

        return {"error": f"Organic chemistry action '{action}' not implemented.", "agent": self.name}

    def f0z_reaction_kinetics(self, concentration: float, rate_constant: float, temperature: float) -> float:
        """Calculates reaction rate using Arrhenius equation (simplified) with F0Z."""
        # Simplified: Ea/R = 1.0 for example
        activation_energy_term = 1.0
        temp_k = temperature + 273.15 # Assume input temp is Celsius
        if temp_k <= 0: temp_k = self.epsilon # Avoid zero or negative Kelvin temp

        # Rate = k * [A]^n * exp(-Ea/RT) - simplified to Rate = k * [A] * exp(-1/T)
        rate = rate_constant * concentration * np.exp(-activation_energy_term / temp_k)
        stabilized_rate = self.math_module.f0z_stabilize(torch.tensor(rate)).item()
        # print(f"  Reaction Rate: k={rate_constant}, [A]={concentration:.2f}, T={temp_k:.1f}K => Rate={stabilized_rate:.4e}")
        return stabilized_rate

    def f0z_bond_energy(self, bond_strength: float, distance: float) -> float:
        """Calculates simplified bond energy with F0Z stabilization."""
        # Simple inverse relationship: Energy = Strength / Distance
        if distance <= 0: distance = self.epsilon # Avoid zero or negative distance

        energy = bond_strength / distance
        stabilized_energy = self.math_module.f0z_stabilize(torch.tensor(energy)).item()
        # Clipping to prevent extreme values from division by small distance
        clipped_energy = np.clip(stabilized_energy, -1e6, 1e6)
        # print(f"  Bond Energy: Strength={bond_strength}, Dist={distance:.3f} => Energy={clipped_energy:.4e}")
        return clipped_energy

    def get_domain_specific_state(self) -> Optional[Dict]:
        # Share known reaction rates
        return {"known_rates_count": len(self.reaction_rates)}


class MolecularBiologyAgent(DFSNAgent):
    """Domain-focused agent for molecular biology simulations."""
    def __init__(self, name: str):
        super().__init__(name)
        self.epsilon = self.math_module.epsilon_0
        self.sequences = {} # Store known sequences
        print(f"MolecularBiologyAgent {self.name} initialized.")


    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles molecular biology tasks like DNA replication and protein folding."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type != "molecular_biology":
            return super()._execute_single_task_iteration(task)

        required_keys_map = {
            "dna_replication": ["sequence", "polymerase_rate"],
            "protein_folding": ["amino_acids", "folding_energy"],
            "analyze_sequence": ["sequence"]
        }

        if not action:
            return {"error": "Missing 'action' key for molecular_biology task", "agent": self.name}

        keys_needed = required_keys_map.get(action)
        if keys_needed is None:
            return {"error": f"Unsupported molecular biology action: {action}", "agent": self.name}

        error = self.check_task_requirements(task, keys_needed)
        if error:
            return error

        try:
            if action == "dna_replication":
                result = self.f0z_dna_replication(task["sequence"], task["polymerase_rate"])
                return {"result": result, "agent": self.name}
            elif action == "protein_folding":
                # Assume amino_acids is a sequence or list of properties
                aa_data = task["amino_acids"]
                if isinstance(aa_data, str): # If it's a string sequence, use length as simple measure
                     aa_measure = len(aa_data)
                elif isinstance(aa_data, (list, np.ndarray)): # If numeric properties, use sum
                     aa_measure = np.sum(np.abs(aa_data))
                else: aa_measure = 1.0 # Fallback

                energy = self.f0z_protein_folding(aa_measure, task["folding_energy"])
                return {"result": {"folded_energy": energy}, "agent": self.name}
            elif action == "analyze_sequence":
                 gc_content = self.analyze_sequence(task["sequence"])
                 return {"result": {"gc_content": gc_content}, "agent": self.name}

        except Exception as e:
             print(f"Error during molecular biology execution ({action}) for {self.name}: {e}")
             return {"error": f"Exception during mol bio execution: {e}", "agent": self.name}

        return {"error": f"Molecular biology action '{action}' not implemented.", "agent": self.name}


    def f0z_dna_replication(self, sequence: str, polymerase_rate: float) -> Dict:
        """Simulates DNA replication rate and creates complementary strand."""
        if not sequence: return {"new_strand": "", "rate": 0.0}
        replication_rate = polymerase_rate * len(sequence)
        stabilized_rate = self.math_module.f0z_stabilize(torch.tensor(replication_rate)).item()

        # Generate complementary strand (simple A-T, G-C pairing)
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        try:
            new_strand = "".join(complement_map.get(base.upper(), 'N') for base in reversed(sequence))
        except Exception as e:
             print(f"Error generating complementary strand: {e}")
             new_strand = "Error"

        # print(f"  DNA Replication: Rate={stabilized_rate:.3f}, New Strand Generated.")
        return {"new_strand": new_strand, "rate": stabilized_rate}

    def f0z_protein_folding(self, amino_acid_measure: float, folding_energy: float) -> float:
        """Calculates simplified protein folding energy."""
        # Simple inverse relationship: Energy = FoldingEnergy / AminoAcidMeasure
        denominator = amino_acid_measure + self.epsilon # Stabilize denominator
        if denominator == 0: denominator = self.epsilon

        energy = folding_energy / denominator
        stabilized_energy = self.math_module.f0z_stabilize(torch.tensor(energy)).item()
        # Clip to prevent extreme values
        clipped_energy = np.clip(stabilized_energy, -1e6, 1e6)
        # print(f"  Protein Folding: Energy={folding_energy}, AA Measure={amino_acid_measure:.2f} => Folded Energy={clipped_energy:.4e}")
        return clipped_energy

    def analyze_sequence(self, sequence: str) -> float:
         """Calculates GC content of a sequence."""
         if not sequence: return 0.0
         sequence = sequence.upper()
         gc_count = sequence.count('G') + sequence.count('C')
         total_len = len(sequence)
         if total_len == 0: return 0.0
         gc_content = gc_count / total_len
         stabilized_gc = self.math_module.f0z_stabilize(torch.tensor(gc_content)).item()
         return stabilized_gc

    def get_domain_specific_state(self) -> Optional[Dict]:
        # Share count of stored sequences
        return {"known_sequences_count": len(self.sequences)}


class CreativeAgent(DFSNAgent):
    """Agent focused on generative and creative tasks."""
    def __init__(self, name: str):
        super().__init__(name)
        self.generated_outputs = [] # History of generated content
        # Could potentially integrate a simple generative model or LLM call here
        try:
            # Use a smaller model suitable for creative text generation snippets
            self.generator_pipe = pipeline("text-generation", model="gpt2", device=-1) # Use CPU for lighter model
            print(f"CreativeAgent {self.name} initialized with gpt2 pipeline.")
        except Exception as e:
            print(f"Warning: Failed to load gpt2 pipeline for CreativeAgent: {e}. Using placeholder.")
            self.generator_pipe = None


    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles creative tasks like text generation."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "creative":
            if action == "generate_text":
                error = self.check_task_requirements(task, ["prompt"])
                if error: return error
                feedback = task.get("feedback", "neutral") # Use feedback if provided
                complexity = task.get("complexity", 5)
                output = self.generate_creative_output(task["prompt"], complexity, feedback)
                return {"result": {"generated_text": output}, "agent": self.name}
            elif action == "generate_idea": # From collaborative example
                 idea = self.generate_idea()
                 return {"result": {"idea": idea}, "agent": self.name}


        return super()._execute_single_task_iteration(task) # Delegate non-creative tasks

    def generate_creative_output(self, prompt: str, complexity: float, feedback: str) -> str:
        """Generates creative text output based on prompt, complexity, and feedback."""
        print(f"{self.name} generating creative content. Prompt: '{prompt[:30]}...', Complexity: {complexity}, Feedback: {feedback}")

        # Adjust generation parameters based on complexity and feedback
        max_len = int(max(30, min(150, 30 + complexity * 10))) # Longer output for higher complexity
        num_seq = 1
        temperature = 1.0 # Default creativity
        if feedback == 'positive':
            temperature = 0.8 # Be slightly more focused if feedback is positive
        elif feedback == 'negative':
            temperature = 1.2 # Be more random/exploratory if feedback is negative

        # Use the generator pipeline if available
        if self.generator_pipe:
            try:
                # Ensure prompt is not overly long for the model
                truncated_prompt = prompt[:500]
                generated = self.generator_pipe(
                    truncated_prompt,
                    max_length=max_len,
                    num_return_sequences=num_seq,
                    temperature=temperature,
                    do_sample=True # Ensure sampling is enabled for temperature to have effect
                )
                output_text = generated[0]['generated_text']
                # Clean up output, remove prompt potentially
                if output_text.startswith(truncated_prompt):
                    output_text = output_text[len(truncated_prompt):].strip()

            except Exception as e:
                print(f"  Error during text generation: {e}")
                output_text = f"Error generating: {e}"
        else:
            # Placeholder generation
            output_text = f"Simulated creative response to '{prompt[:20]}...' based on complexity {complexity} and feedback {feedback}. [T={temperature:.1f}, L={max_len}]"

        self.generated_outputs.append(output_text)
        self.performance_history.append(len(output_text) / 100.0) # Simple performance metric: length
        print(f"  Generated output length: {len(output_text)}")
        return output_text

    def generate_idea(self) -> str:
         """Generates a simple creative idea."""
         ideas = ["Quantum poetry", "Fractal music", "AI-driven chemistry", "Entangled decision trees", "Zero-bug code"]
         idea = random.choice(ideas) + f" (variant {random.randint(1,100)})"
         print(f"{self.name} generated idea: {idea}")
         return idea

    def integrate_idea(self, shared_idea: str):
         """Placeholder for integrating an idea from another agent."""
         print(f"{self.name} received shared idea: '{shared_idea}'. Integrating into creative process.")
         # Could add to prompts or influence internal state
         self.generate_creative_output(f"Inspired by the idea: {shared_idea}", complexity=6, feedback="neutral")


    def get_domain_specific_state(self) -> Optional[Dict]:
         return {"recent_outputs_count": len(self.generated_outputs)}


class InformationTheoryAgent(DFSNAgent):
    """Agent focused on information theory concepts like entropy and mutual information."""
    def __init__(self, name: str):
        super().__init__(name)
        self.epsilon = self.math_module.epsilon_0
        self.mode = "classical" # Added from LLaDA example
        self.todo_queue = [] # Added from LLaDA example
        self.consolidated_todos = [] # Added from LLaDA example
        print(f"InformationTheoryAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles information theory tasks."""
        task_type = task.get("type")
        action = task.get("action")

        # Infer action from type if not specified
        if not action:
            if "entropy" in task_type: action = "compute_entropy"
            elif "mutual_info" in task_type: action = "compute_mutual_info"
            elif "channel_capacity" in task_type: action = "optimize_channel_capacity"
            elif "validate_hypothesis" in task_type: action = "validate_hypothesis" # From collab example
            task['action'] = action

        if action == "compute_entropy":
            error = self.check_task_requirements(task, ["probabilities"])
            if error: return error
            entropy = self.compute_entropy(task["probabilities"])
            return {"result": {"entropy": entropy}, "agent": self.name}
        elif action == "compute_mutual_info":
            error = self.check_task_requirements(task, ["X", "Y"])
            if error: return error
            mi = self.compute_mutual_info(task["X"], task["Y"])
            return {"result": {"mutual_information": mi}, "agent": self.name}
        elif action == "optimize_channel_capacity":
            error = self.check_task_requirements(task, ["signal", "noise", "bandwidth"])
            if error: return error
            capacity = self.optimize_channel_capacity(task["signal"], task["noise"], task["bandwidth"])
            return {"result": {"channel_capacity": capacity}, "agent": self.name}
        elif action == "validate_hypothesis": # From ZSG Collab Manager example
             error = self.check_task_requirements(task, ["hypothesis"])
             if error: return error
             is_valid = self.validate_hypothesis(task["hypothesis"])
             return {"result": {"is_valid": is_valid}, "agent": self.name}


        # If task involves TODO management (from LLaDA example)
        elif action == "add_todo":
             error = self.check_task_requirements(task, ["todo_data"])
             if error: return error
             self.add_todo(task["todo_data"])
             return {"status": "todo_added", "agent": self.name}
        elif action == "collect_results":
             error = self.check_task_requirements(task, ["source_agent_name"])
             if error: return error
             # Needs access to other agents - requires manager/registry
             print(f"Warning: collect_results needs agent registry access (not implemented here).")
             # result = self.collect_results(task["source_agent"]) # Conceptual
             return {"status": "collect_results (simulated)", "agent": self.name}
        elif action == "consolidate":
             # Use internal queue for consolidation
             self.consolidate_todos(self.todo_queue) # Consolidate pending todos
             return {"status": "consolidation_attempted", "consolidated_count": len(self.consolidated_todos), "agent": self.name}


        return super()._execute_single_task_iteration(task) # Delegate other tasks


    def compute_entropy(self, probabilities: Any) -> float:
        """Computes Shannon entropy with F0Z stabilization."""
        try:
            probs_tensor = torch.tensor(probabilities, dtype=torch.float32)
            # Ensure probabilities are non-negative and sum approximately to 1
            probs_tensor = torch.clamp(probs_tensor, min=0.0)
            prob_sum = torch.sum(probs_tensor)
            if prob_sum <= 0: return 0.0 # Avoid division by zero or log(0) if sum is zero
            probs_tensor = probs_tensor / prob_sum # Normalize

            # Stabilize probabilities before log calculation
            stabilized_probs = self.math_module.f0z_stabilize(probs_tensor, system_size=len(probabilities))

            # Add epsilon inside log2 for numerical stability near zero probability
            log_probs = torch.log2(stabilized_probs + float(self.epsilon))
            entropy = -torch.sum(stabilized_probs * log_probs)
            # print(f"  Entropy calculated: {entropy.item():.4f}")
            return entropy.item()
        except Exception as e:
             print(f"Error computing entropy: {e}")
             return 0.0

    def compute_mutual_info(self, X: Any, Y: Any, bins: int = 10) -> float:
        """Computes Mutual Information between two variables X and Y with F0Z."""
        try:
            X_np = np.array(X).flatten()
            Y_np = np.array(Y).flatten()
            if len(X_np) != len(Y_np):
                 raise ValueError("X and Y must have the same length.")
            if len(X_np) == 0: return 0.0

            # Calculate 2D histogram to get joint probability distribution
            joint_hist, _, _ = np.histogram2d(X_np, Y_np, bins=bins, density=True)
            joint_prob = torch.tensor(joint_hist, dtype=torch.float32)
            # Normalize joint probability
            joint_prob /= torch.sum(joint_prob)

            # Stabilize joint probability
            joint_prob_s = self.math_module.f0z_stabilize(joint_prob, system_size=joint_prob.numel())

            # Calculate marginal probabilities
            marginal_x = torch.sum(joint_prob_s, axis=1)
            marginal_y = torch.sum(joint_prob_s, axis=0)

            # Stabilize marginals (important!)
            marginal_x_s = self.math_module.f0z_stabilize(marginal_x, system_size=len(marginal_x))
            marginal_y_s = self.math_module.f0z_stabilize(marginal_y, system_size=len(marginal_y))

            # Compute MI using stabilized probabilities
            # MI = sum_{x,y} P(x,y) * log2( P(x,y) / (P(x)P(y)) )
            # Add epsilon for stability in logs and division
            term_xy = marginal_x_s[:, None] * marginal_y_s[None, :] + float(self.epsilon)
            log_term = torch.log2(joint_prob_s + float(self.epsilon)) - torch.log2(term_xy)

            mutual_info = torch.sum(joint_prob_s * log_term)

            # Ensure MI is non-negative (theoretically should be, but numerical issues)
            mi_val = max(0.0, mutual_info.item())
            # print(f"  Mutual Information calculated: {mi_val:.4f}")
            return mi_val
        except Exception as e:
             print(f"Error computing mutual information: {e}")
             return 0.0


    def optimize_channel_capacity(self, signal_power: float, noise_power: float, bandwidth: float) -> float:
        """Calculates channel capacity using Shannon-Hartley theorem with F0Z."""
        # C = B * log2(1 + S/N)
        signal = torch.tensor(signal_power, dtype=torch.float32)
        noise = torch.tensor(noise_power, dtype=torch.float32)
        bw = torch.tensor(bandwidth, dtype=torch.float32)

        # Stabilize noise power before division
        noise_s = self.math_module.f0z_stabilize(noise, system_size=1)
        if noise_s.item() <= 0: # If noise is stabilized to zero or negative, capacity is effectively infinite or undefined
             print("  Warning: Noise power stabilized to zero or less. Returning large capacity value.")
             return 1e9 # Return a large number

        snr = signal / noise_s
        # Stabilize SNR >= 0 before adding 1
        snr = torch.clamp(snr, min=0.0)

        capacity = bw * torch.log2(1.0 + snr)
        stabilized_capacity = self.math_module.f0z_stabilize(capacity, system_size=1)
        # print(f"  Channel Capacity: S={signal_power:.2f}, N={noise_power:.2f}, B={bandwidth:.2f} => C={stabilized_capacity.item():.4f}")
        return stabilized_capacity.item()

    def validate_hypothesis(self, hypothesis: Dict) -> bool:
        """Validates a hypothesis based on information-theoretic principles (placeholder)."""
        # Example: Check if hypothesis reduces uncertainty (increases information)
        # Requires quantifying information content of hypothesis and data
        print(f"Validating hypothesis (ID: {hypothesis.get('id', 'N/A')}) using InfoTheory (placeholder).")
        # Placeholder logic: Check if expected info gain is positive
        expected_info_gain = hypothesis.get("variables", {}).get("expected_information_gain", 0.1)
        is_valid = expected_info_gain > 0 and random.random() < 0.9 # Simple probabilistic validation
        print(f"  Hypothesis valid: {is_valid}")
        return is_valid

    # --- Methods from LLaDA Consolidation Example ---
    def transition_mode(self, bridge: Optional[ZSGQuantumBridge] = None):
        """Transition between classical and quantum modes based on entropy."""
        if not bridge or not bridge.simulator:
             print(f"{self.name}: Cannot transition mode without quantum bridge.")
             self.mode = "classical"
             return

        # Encode a dummy high-priority task to check entanglement/entropy
        bridge.encode({"priority": 10}, entangle=True)
        entropy, _ = bridge.entropy_bot.monitor_state(bridge.simulator.state)
        # Threshold could be adaptive
        entropy_threshold = bridge.simulator.n_qubits * np.log2(2) * 0.6 # 60% of max entropy
        new_mode = "quantum" if entropy > entropy_threshold else "classical"
        if new_mode != self.mode:
            self.mode = new_mode
            print(f"{self.name}: Transitioned to {self.mode} mode (Entropy: {entropy:.3f} / Threshold: {entropy_threshold:.3f}).")

    def add_todo(self, todo_data: Dict):
        """Adds a ZSGTodo item to the internal queue."""
        # Create a ZSGTodo object - Ensure necessary fields are present
        task_id = todo_data.get("task_id", f"todo_{len(self.todo_queue)}_{int(time.time())}")
        description = todo_data.get("description", "No description")
        status = todo_data.get("status", "Pending")
        priority = todo_data.get("priority", 5.0) # Default priority
        mliw_step = todo_data.get("mliw_step", "Process")
        payload = todo_data.get("data", {})

        # Ensure priority is float
        try:
            priority = float(priority)
        except (ValueError, TypeError):
            priority = 5.0 # Default if conversion fails

        todo = ZSGTodo(task_id, description, status, priority, mliw_step, payload)
        self.todo_queue.append(todo)
        print(f"{self.name}: Added TODO {task_id} to internal queue (Queue size: {len(self.todo_queue)}).")
        # Optionally store in MemorySystem as well
        # self.memory_system.store(f"todo_{task_id}", todo.to_json())

    def collect_results(self, agent: Agent, bridge: Optional[ZSGQuantumBridge] = None):
        """Collects results (TODOs) from another agent, potentially syncing quantum state."""
        if not hasattr(agent, 'state'):
            print(f"Warning: Agent {agent.name} does not have a 'state' attribute to collect results from.")
            return None

        agent_state = agent.state # Assuming agent stores its state here

        # Quantum state synchronization if in quantum mode
        if self.mode == "quantum" and bridge and bridge.simulator:
            peer_quantum_state = agent_state.get("quantum_state_vector")
            if peer_quantum_state and len(peer_quantum_state) == 2**bridge.simulator.n_qubits:
                # Option 1: Update bridge state
                bridge.simulator.state = np.array(peer_quantum_state, dtype=complex)
                print(f"{self.name}: Synchronized quantum state from {agent.name}.")
            else:
                print(f"{self.name}: Failed to synchronize quantum state from {agent.name} (invalid state).")

        # Collect completed TODO from agent state
        todo_in_agent_state = agent_state.get("current_todo")
        is_completed = agent_state.get("task_completed", False)

        if isinstance(todo_in_agent_state, ZSGTodo) and is_completed:
             print(f"{self.name}: Collected completed TODO {todo_in_agent_state.task_id} from {agent.name}.")
             # Remove from internal queue if it exists there
             self.todo_queue = [t for t in self.todo_queue if t.task_id != todo_in_agent_state.task_id]
             # Store completion in memory system (optional)
             # self.memory_system.store(f"completed_{agent.name}_{todo.task_id}", todo.to_json())
             return todo_in_agent_state # Return the completed TODO object
        elif isinstance(todo_in_agent_state, dict) and todo_in_agent_state.get('status') == 'Completed':
              # Handle if agent stores basic dict instead of ZSGTodo object
              print(f"{self.name}: Collected completed TODO (dict) {todo_in_agent_state.get('task_id', 'N/A')} from {agent.name}.")
              self.todo_queue = [t for t in self.todo_queue if t.task_id != todo_in_agent_state.get('task_id')]
              return todo_in_agent_state

        # print(f"{self.name}: No completed TODO found in {agent.name}'s state.")
        return None

    def consolidate_todos(self, todos_to_consolidate: List[ZSGTodo]):
        """Consolidates a list of TODOs if they meet balance criteria."""
        if not todos_to_consolidate or len(todos_to_consolidate) < 2:
            # print(f"{self.name}: Not enough TODOs ({len(todos_to_consolidate)}) to consolidate.")
            return

        try:
            priorities = [float(todo.priority) for todo in todos_to_consolidate]
            # Use variance from payload if available, else default
            variances = [float(todo.data_payload.get("variance", 0.0)) for todo in todos_to_consolidate]

            # Balance score: lower is more balanced (less variance in priority/data variance)
            balance_score = np.std(priorities) + np.std(variances)
            balance_threshold = 1.5 # Tunable threshold

            print(f"{self.name}: Checking consolidation for {len(todos_to_consolidate)} TODOs. Balance score: {balance_score:.3f} (Threshold: {balance_threshold})")

            if balance_score < balance_threshold:
                consolidated_id = f"consolidated_{len(self.consolidated_todos)}_{int(time.time())}"
                consolidated = {
                    "consolidated_id": consolidated_id,
                    "task_ids": [todo.task_id for todo in todos_to_consolidate],
                    "total_priority": sum(priorities),
                    "average_priority": np.mean(priorities),
                    "total_variance": sum(variances),
                    "average_variance": np.mean(variances),
                    "mliw_step": "Consolidated", # Mark step
                    "status": "Consolidated",
                    "description": f"Consolidated {len(todos_to_consolidate)} tasks."
                    # Combine payloads intelligently if needed (e.g., merge dicts)
                    # "combined_payload": self._merge_payloads([todo.data_payload for todo in todos_to_consolidate])
                }
                self.consolidated_todos.append(consolidated)
                # Remove consolidated todos from the main queue
                consolidated_ids = set(consolidated["task_ids"])
                self.todo_queue = [todo for todo in self.todo_queue if todo.task_id not in consolidated_ids]

                print(f"  Consolidated {len(todos_to_consolidate)} TODOs into document {consolidated_id}. Queue size: {len(self.todo_queue)}")
                # Store consolidated doc in memory (optional)
                # self.memory_system.store(consolidated_id, consolidated, memory_type='long')
            else:
                print(f"  TODOs not balanced enough for consolidation (Score {balance_score:.3f} >= {balance_threshold}).")
                # Optional: Redistribute or re-queue logic could go here
                # self.redistribute(todos_to_consolidate) # Example redistribution call

        except Exception as e:
            print(f"Error during TODO consolidation: {e}")

    def redistribute(self, todos: List[ZSGTodo]):
        """Handles redistribution of TODOs if consolidation fails or is undone."""
        print(f"{self.name}: Redistributing {len(todos)} TODOs back to AIW queue.")
        # Simply add them back to the internal queue for now
        # More complex logic could involve re-prioritizing or assigning to specific agents
        for todo in todos:
            if not any(t.task_id == todo.task_id for t in self.todo_queue): # Avoid duplicates
                 todo.status = "Pending" # Reset status
                 self.todo_queue.append(todo)

    def _merge_payloads(self, payloads: List[Dict]) -> Dict:
         """Helper to merge data payloads from multiple TODOs."""
         merged = {}
         all_keys = set(k for p in payloads for k in p.keys())
         for key in all_keys:
             values = [p.get(key) for p in payloads if p.get(key) is not None]
             if not values: continue
             # Simple merge: store values in a list, or average if numeric
             if all(isinstance(v, (int, float)) for v in values):
                 merged[key] = np.mean(values)
             else:
                 merged[key] = values # Store as list if non-numeric or mixed
         return merged

    def get_domain_specific_state(self) -> Optional[Dict]:
         return {"mode": self.mode, "queue_size": len(self.todo_queue), "consolidated_count": len(self.consolidated_todos)}


class HypothesisAgent(DFSNAgent):
    """Agent responsible for generating and proposing hypotheses."""
    def __init__(self, name: str):
        super().__init__(name)
        self.hypotheses_proposed = []
        print(f"HypothesisAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles hypothesis generation or related tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if action == "propose_hypothesis":
            # Generate hypothesis based on input data or task context
            input_data = task.get("data", {})
            hypothesis = self.propose_hypothesis(input_data)
            if hypothesis:
                self.hypotheses_proposed.append(hypothesis)
                return {"result": hypothesis, "agent": self.name}
            else:
                return {"error": "Could not generate hypothesis", "agent": self.name}
        elif action == "suggest_improvement": # From info theory agent example interaction
             input_data = task.get("data", {})
             suggestion = self.suggest_improvement(input_data)
             return {"result": {"suggestion": suggestion}, "agent": self.name}


        return super()._execute_single_task_iteration(task) # Delegate other tasks

    def propose_hypothesis(self, context_data: Dict) -> Optional[Dict]:
        """Generates a new hypothesis based on context."""
        # Simple rule-based or template-based generation for now
        hypothesis_id = f"hyp_{len(self.hypotheses_proposed)}_{int(time.time())}"
        description = "Generated Hypothesis: "
        variables = {}

        # Example: If context involves quantum simulation results
        if context_data.get("task_type") == "quantum_circuit" and "decoded_info" in context_data.get("result", {}):
            entropy = context_data["result"]["decoded_info"].get("entropy", 0)
            if entropy > 1.0:
                description += "High entanglement observed correlates with task success."
                variables = {"correlation_metric": "entanglement_entropy", "threshold": 1.0, "expected_outcome": "improved_performance"}
            else:
                description += "Low entanglement suggests simpler states dominate."
                variables = {"correlation_metric": "entanglement_entropy", "threshold": 1.0, "expected_outcome": "simpler_dynamics"}
        elif context_data.get("task_type") == "physics_simulation":
             # Look for chaotic behavior in results (placeholder)
             if context_data.get("result", {}).get("chaos_detected", False):
                 description += "System exhibits chaotic behavior under current parameters."
                 variables = {"phenomenon": "chaos", "trigger_params": context_data.get("params"), "prediction": "sensitivity_to_initial_conditions"}
             else:
                 description += "System appears stable or periodic."
                 variables = {"phenomenon": "stability", "conditions": context_data.get("params")}
        else:
            # Generic hypothesis
            description += f"Observed pattern in data payload suggests a power-law relationship."
            variables = {"relationship": "power-law", "confidence": 0.6}

        print(f"{self.name} proposing hypothesis: {description}")
        hypothesis = {
             "id": hypothesis_id,
             "description": description,
             "variables": variables,
             "status": "proposed",
             "validation_results": {}
        }
        return hypothesis


    def suggest_improvement(self, task_result_data: Dict) -> Optional[str]:
        """Suggests improvements or next steps based on results and hypotheses."""
        # Example: If a hypothesis was recently proposed, suggest testing it
        if self.hypotheses_proposed:
            last_hypothesis = self.hypotheses_proposed[-1]
            if last_hypothesis["status"] == "proposed":
                 suggestion = f"Suggest testing hypothesis '{last_hypothesis['id']}'. Design experiment targeting variables: {list(last_hypothesis['variables'].keys())}"
                 print(f"{self.name} suggesting improvement: {suggestion}")
                 return suggestion

        # Example: If results show high error, suggest parameter tuning
        if task_result_data.get("error", 0) > 0.5:
             suggestion = "High error observed. Suggest parameter sweep or trying alternative model/algorithm."
             print(f"{self.name} suggesting improvement: {suggestion}")
             return suggestion

        return None # No specific suggestion

    def get_domain_specific_state(self) -> Optional[Dict]:
        return {"proposed_hypotheses_count": len(self.hypotheses_proposed)}


class DataScienceAgent(DFSNAgent):
    """Agent for data analysis and modeling tasks."""
    def __init__(self, name: str):
        super().__init__(name)
        self.models = {} # Store trained models or analysis results
        print(f"DataScienceAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles data science tasks like regression."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "data_science":
            if action == "linear_regression":
                error = self.check_task_requirements(task, ["X", "Y"])
                if error: return error
                try:
                    slope, intercept = self.linear_regression(task["X"], task["Y"])
                    return {"result": {"slope": slope, "intercept": intercept}, "agent": self.name}
                except Exception as e:
                    return {"error": f"Regression failed: {e}", "agent": self.name}
            elif action == "analyze_data":
                 error = self.check_task_requirements(task, ["data"])
                 if error: return error
                 stats = self.analyze_data(task["data"])
                 return {"result": {"statistics": stats}, "agent": self.name}

        return super()._execute_single_task_iteration(task) # Delegate other tasks


    def linear_regression(self, X: Any, Y: Any) -> Tuple[float, float]:
        """Performs simple linear regression using F0Z-stabilized calculations."""
        try:
            X_t = torch.tensor(X, dtype=torch.float32).squeeze() # Ensure 1D
            Y_t = torch.tensor(Y, dtype=torch.float32).squeeze() # Ensure 1D
            if X_t.dim() != 1 or Y_t.dim() != 1 or len(X_t) != len(Y_t) or len(X_t) < 2:
                 raise ValueError("Inputs must be 1D arrays of the same length (>= 2).")

            # Using F0Z variance and mean
            mean_x = torch.mean(X_t)
            mean_y = torch.mean(Y_t)
            # Stabilized variance of X
            var_x_val = torch.mean((X_t - mean_x)**2)
            var_x_s = self.math_module.f0z_stabilize(var_x_val, system_size=len(X_t))

            if var_x_s.item() == 0: # Avoid division by zero if variance is stabilized to zero
                 print("  Warning: Stabilized variance of X is zero. Cannot compute slope.")
                 return 0.0, mean_y.item() # Return slope 0, intercept as mean Y

            # Covariance
            cov_xy = torch.mean((X_t - mean_x) * (Y_t - mean_y))
            cov_xy_s = self.math_module.f0z_stabilize(cov_xy, system_size=len(X_t))

            # Slope = Cov(X, Y) / Var(X)
            slope = cov_xy_s / var_x_s
            slope_s = self.math_module.f0z_stabilize(slope, system_size=len(X_t))

            # Intercept = mean(Y) - slope * mean(X)
            intercept = mean_y - slope_s * mean_x
            intercept_s = self.math_module.f0z_stabilize(intercept, system_size=len(X_t))

            print(f"  Linear Regression: Slope={slope_s.item():.4f}, Intercept={intercept_s.item():.4f}")
            return slope_s.item(), intercept_s.item()

        except Exception as e:
            print(f"Error during linear regression: {e}")
            raise # Re-raise exception


    def analyze_data(self, data: Any) -> Dict:
         """Performs basic statistical analysis on data."""
         stats = {}
         try:
             data_np = np.array(data).flatten()
             numeric_data = data_np[np.isfinite(data_np) & ~np.isnan(data_np)] # Filter out NaN/inf
             if numeric_data.size == 0: return {"error": "No valid numeric data found"}

             stats['count'] = len(numeric_data)
             stats['mean'] = self.math_module.f0z_stabilize(torch.tensor(np.mean(numeric_data))).item()
             stats['median'] = self.math_module.f0z_stabilize(torch.tensor(np.median(numeric_data))).item()
             stats['std_dev'] = np.sqrt(F0ZAlgebra.f0z_variance(numeric_data))
             stats['min'] = self.math_module.f0z_stabilize(torch.tensor(np.min(numeric_data))).item()
             stats['max'] = self.math_module.f0z_stabilize(torch.tensor(np.max(numeric_data))).item()
             print(f"  Data Analysis: Mean={stats['mean']:.3f}, StdDev={stats['std_dev']:.3f}, Count={stats['count']}")
         except Exception as e:
             print(f"Error during data analysis: {e}")
             return {"error": str(e)}
         return stats


    def get_domain_specific_state(self) -> Optional[Dict]:
        return {"model_count": len(self.models)}


class AstrophysicsAgent(DFSNAgent):
    """Agent for astrophysics calculations and simulations."""
    def __init__(self, name: str):
        super().__init__(name)
        self.orbital_data = {} # Store orbital parameters
        self.G = 6.67430e-11 # Gravitational constant
        print(f"AstrophysicsAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles astrophysics tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "astrophysics":
            if action == "compute_orbital_velocity":
                error = self.check_task_requirements(task, ["central_mass", "distance"])
                if error: return error
                velocity = self.compute_orbital_velocity(task["central_mass"], task["distance"])
                return {"result": {"orbital_velocity_mps": velocity}, "agent": self.name}
            elif action == "simulate_orbit":
                 error = self.check_task_requirements(task, ["central_mass", "body_mass", "initial_pos", "initial_vel", "time_steps"])
                 if error: return error
                 trajectory = self.simulate_orbit(task["central_mass"], task["body_mass"], task["initial_pos"], task["initial_vel"], task["time_steps"])
                 return {"result": {"trajectory": trajectory}, "agent": self.name}


        return super()._execute_single_task_iteration(task) # Delegate other tasks

    def compute_orbital_velocity(self, central_mass: float, distance: float) -> float:
        """Computes circular orbital velocity using F0Z."""
        mass_t = torch.tensor(central_mass, dtype=torch.float64) # Use float64 for precision
        dist_t = torch.tensor(distance, dtype=torch.float64)

        dist_s = self.math_module.f0z_stabilize(dist_t, system_size=1)
        if dist_s.item() <= 0:
            print("  Warning: Distance is zero or negative. Returning zero velocity.")
            return 0.0

        # v = sqrt(G * M / r)
        velocity_sq = (self.G * mass_t) / dist_s
        velocity_sq = torch.clamp(velocity_sq, min=0.0) # Ensure non-negative before sqrt
        velocity = torch.sqrt(velocity_sq)

        stabilized_velocity = self.math_module.f0z_stabilize(velocity, system_size=1)
        print(f"  Orbital Velocity: M={central_mass:.2e}, r={distance:.2e} => v={stabilized_velocity.item():.3f} m/s")
        return stabilized_velocity.item()

    def simulate_orbit(self, central_mass: float, body_mass: float, initial_pos: List[float], initial_vel: List[float], time_steps: int, dt: float = 86400.0): # dt default = 1 day
        """Simulates a simple 2D orbit using F0Z-stabilized Euler method."""
        pos = torch.tensor(initial_pos, dtype=torch.float64)
        vel = torch.tensor(initial_vel, dtype=torch.float64)
        m1 = torch.tensor(central_mass, dtype=torch.float64)
        # m2 = torch.tensor(body_mass, dtype=torch.float64) # Body mass often negligible

        trajectory = [pos.tolist()]

        for _ in range(time_steps):
             # Calculate distance and direction vector
             dist_vec = -pos # Vector from body to central mass at origin
             distance = torch.norm(dist_vec)
             dist_s = self.math_module.f0z_stabilize(distance, system_size=1)

             if dist_s.item() <= 0:
                 print("  Warning: Orbital distance became zero. Stopping simulation.")
                 break

             # Calculate gravitational force F = G * m1 * m2 / r^2 (direction is dist_vec/distance)
             # Acceleration a = F / m2 = G * m1 / r^2 * (dist_vec/distance)
             accel_mag = (self.G * m1) / (dist_s**2)
             accel_vec = accel_mag * (dist_vec / dist_s)
             accel_s = self.math_module.f0z_stabilize(accel_vec, system_size=2)

             # Update velocity and position using Euler method
             vel = vel + accel_s * dt
             pos = pos + vel * dt

             # Stabilize position and velocity vectors
             vel = self.math_module.f0z_stabilize(vel, system_size=2)
             pos = self.math_module.f0z_stabilize(pos, system_size=2)

             trajectory.append(pos.tolist())

        print(f"  Simulated orbit for {time_steps} steps.")
        return trajectory

    def get_domain_specific_state(self) -> Optional[Dict]:
        return {"orbital_data_count": len(self.orbital_data)}


class RoboticsAgent(DFSNAgent):
    """Agent for robotics tasks like path planning."""
    def __init__(self, name: str):
        super().__init__(name)
        self.path_data = [] # Store planned paths
        print(f"RoboticsAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles robotics tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "robotics":
            if action == "plan_path":
                error = self.check_task_requirements(task, ["start_pose", "goal_pose", "environment"])
                if error: return error
                path = self.plan_path(task["start_pose"], task["goal_pose"], task["environment"])
                if path:
                     self.path_data.append(path)
                     return {"result": {"path": path}, "agent": self.name}
                else:
                     return {"error": "Path planning failed", "agent": self.name}
            elif action == "execute_path":
                 error = self.check_task_requirements(task, ["path"])
                 if error: return error
                 success = self.execute_path(task["path"])
                 return {"result": {"execution_success": success}, "agent": self.name}

        return super()._execute_single_task_iteration(task) # Delegate other tasks


    def plan_path(self, start_pose: Dict, goal_pose: Dict, environment: Dict) -> Optional[List[Dict]]:
        """Plans a path from start to goal (simplified A* placeholder)."""
        print(f"Planning path from {start_pose} to {goal_pose}...")
        # --- Placeholder A* or RRT* Implementation ---
        # 1. Represent environment (e.g., grid map, obstacles)
        # 2. Initialize open set (priority queue) with start node
        # 3. Initialize closed set
        # 4. Loop while open set is not empty:
        #    a. Get node with lowest f-cost (g+h) from open set
        #    b. If current is goal, reconstruct path and return
        #    c. Add current to closed set
        #    d. For each neighbor:
        #       i. If neighbor is obstacle or in closed set, skip
        #       ii. Calculate tentative g-cost
        #       iii. If new path to neighbor is shorter OR neighbor not in open set:
        #            - Update neighbor's g, h, f costs and parent
        #            - Add neighbor to open set
        # --- End Placeholder ---

        # Simplified straight-line path for now
        path = [start_pose, goal_pose] # Direct path
        print(f"  Planned simple path with {len(path)} points.")

        # Apply F0Z to waypoints if needed (e.g., stabilize coordinates)
        stabilized_path = []
        for pose in path:
            # Assuming pose is like {'x': float, 'y': float, 'theta': float}
            stable_pose = pose.copy()
            for key, val in pose.items():
                 if isinstance(val, (int, float)):
                      stable_pose[key] = self.math_module.f0z_stabilize(torch.tensor(val)).item()
            stabilized_path.append(stable_pose)


        return stabilized_path

    def execute_path(self, path: List[Dict]) -> bool:
         """Simulates executing a planned path."""
         if not path: return False
         print(f"Executing path with {len(path)} waypoints...")
         # Simulate movement, check for collisions etc.
         time.sleep(0.05 * len(path)) # Simulate time taken
         success = random.random() > 0.1 # 90% success rate
         print(f"  Path execution {'succeeded' if success else 'failed'}.")
         return success

    def get_domain_specific_state(self) -> Optional[Dict]:
        return {"paths_planned": len(self.path_data)}


class EnvironmentalScienceAgent(DFSNAgent):
    """Agent for environmental modeling and simulation."""
    def __init__(self, name: str):
        super().__init__(name)
        self.eco_data = {} # Store ecological parameters or simulation results
        print(f"EnvironmentalScienceAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles environmental science tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "environmental_science":
            if action == "simulate_population":
                error = self.check_task_requirements(task, ["initial_population", "growth_rate", "carrying_capacity", "time_steps"])
                if error: return error
                population_trend = self.simulate_population_logistic(
                    task["initial_population"], task["growth_rate"], task["carrying_capacity"], task["time_steps"]
                )
                return {"result": {"population_trend": population_trend}, "agent": self.name}
            elif action == "model_pollution_spread":
                 error = self.check_task_requirements(task, ["source_location", "diffusion_rate", "grid_size", "time_steps"])
                 if error: return error
                 concentration_grid = self.model_pollution_spread(
                      task["source_location"], task["diffusion_rate"], task["grid_size"], task["time_steps"]
                 )
                 return {"result": {"concentration_grid": concentration_grid}, "agent": self.name}


        return super()._execute_single_task_iteration(task) # Delegate other tasks

    def simulate_population_logistic(self, initial_pop: float, growth_rate: float, capacity: float, time_steps: int) -> List[float]:
        """Simulates population growth using logistic model with F0Z."""
        pop = torch.tensor(initial_pop, dtype=torch.float32)
        r = torch.tensor(growth_rate, dtype=torch.float32)
        K = torch.tensor(capacity, dtype=torch.float32)
        K_s = self.math_module.f0z_stabilize(K, system_size=1) # Stabilize carrying capacity
        if K_s.item() <= 0: K_s = torch.tensor(1e6) # Ensure positive capacity

        pop_trend = [pop.item()]

        for _ in range(time_steps):
             # Logistic growth: dP/dt = r * P * (1 - P/K)
             dP_dt = r * pop * (1.0 - pop / K_s)
             dP_dt_s = self.math_module.f0z_stabilize(dP_dt, system_size=1)
             # Simple Euler integration
             pop = pop + dP_dt_s # Assuming dt=1 step
             pop = torch.clamp(pop, min=0.0) # Population cannot be negative
             pop_s = self.math_module.f0z_stabilize(pop, system_size=1) # Stabilize pop each step
             pop = pop_s
             pop_trend.append(pop.item())

        print(f"  Simulated logistic population growth for {time_steps} steps. Final pop: {pop_trend[-1]:.2f}")
        return pop_trend


    def model_pollution_spread(self, source_loc: Tuple[int, int], diffusion_rate: float, grid_size: Tuple[int, int], time_steps: int, dt: float = 0.1):
        """Simulates 2D pollution spread using diffusion equation (like heat eq)."""
        rows, cols = grid_size
        grid = torch.zeros(rows, cols, dtype=torch.float32)
        alpha = torch.tensor(diffusion_rate, dtype=torch.float32)

        # Add initial pollution source
        sx, sy = source_loc
        if 0 <= sx < rows and 0 <= sy < cols:
            grid[sx, sy] = 100.0 # Initial concentration amount

        # Spatial steps
        dx, dy = 1.0, 1.0 # Assume unit spacing

        print(f"Simulating pollution spread on {rows}x{cols} grid for {time_steps} steps...")
        for _ in range(time_steps):
            laplacian = torch.zeros_like(grid)
            # Calculate Laplacian using finite differences (5-point stencil)
            if rows > 2 and cols > 2:
                 laplacian[1:-1, 1:-1] = (grid[:-2, 1:-1] + grid[2:, 1:-1] +
                                          grid[1:-1, :-2] + grid[1:-1, 2:] -
                                          4 * grid[1:-1, 1:-1]) / (dx**2) # Assume dx=dy

            # Update grid: dC/dt = alpha * Laplacian(C)
            grid = grid + alpha * dt * laplacian
            # Apply F0Z stabilization
            grid = self.math_module.f0z_stabilize(grid, system_size=grid.numel())
            # Optional: Add boundary conditions (e.g., reflecting, absorbing)

        print("  Pollution spread simulation complete.")
        return grid.tolist() # Return concentration grid as list of lists

    def get_domain_specific_state(self) -> Optional[Dict]:
        return {"eco_data_entries": len(self.eco_data)}


class MachineLearningAgent(DFSNAgent):
    """Agent specializing in machine learning tasks like training and prediction."""
    def __init__(self, name: str, model_params: Optional[Dict] = None):
        super().__init__(name)
        default_params = {"input_features": 2, "output_features": 1}
        params = model_params if model_params else default_params

        # --- CHANGE HERE: Use nn.Parameter ---
        # Initialize weights and bias data first
        weights_data = torch.randn(params["input_features"], params["output_features"], dtype=torch.float32) * 0.1
        bias_data = torch.randn(params["output_features"], dtype=torch.float32) * 0.1

        # Wrap the data in nn.Parameter to register them as trainable parameters
        self.learning_rate = 0.01 # Learning rate for optimizer
        self.weights = nn.Parameter(weights_data)
        self.bias = nn.Parameter(bias_data)

        # --- CORRECTED OPTIMIZER INITIALIZATION ---
        self.learning_rate = 0.01 # Define learning rate as attribute if needed elsewhere
        # Pass only the parameters to be optimized in the list, lr is a keyword argument
        self.optimizer = torch.optim.SGD([self.weights, self.bias], lr=self.learning_rate)
        # --- END CORRECTION -

        # requires_grad is True by default for nn.Parameter

        # --- Optimizer creation should now work ---
        #self.optimizer = torch.optim.SGD([self.weights, self.bias], lr=0.01)
        self.loss_history = []
        print(f"MachineLearningAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles ML tasks like training and prediction."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "machine_learning":
            if action == "train":
                error = self.check_task_requirements(task, ["X", "Y"])
                if error: return error
                try:
                    X_data = task["X"]
                    Y_data = task["Y"]
                    loss = self.train_model(X_data, Y_data)
                    self.loss_history.append(loss)
                    return {"result": {"final_loss": loss}, "agent": self.name}
                except Exception as e:
                     # traceback.print_exc() # Debugging
                     return {"error": f"Training failed: {e}", "agent": self.name}
            elif action == "predict":
                error = self.check_task_requirements(task, ["X"])
                if error: return error
                try:
                     X_data = task["X"]
                     predictions = self.predict(X_data)
                     return {"result": {"predictions": predictions}, "agent": self.name}
                except Exception as e:
                     # traceback.print_exc() # Debugging
                     return {"error": f"Prediction failed: {e}", "agent": self.name}


        # Delegate other tasks via AIW. AIW will call base__execute_single_task_iteration if needed.
        # This agent doesn't need a specific base__execute_single_task_iteration override if it only handles ML tasks.
        return super()._execute_single_task_iteration(task)


    def train_model(self, X: Any, Y: Any, epochs: int = 5) -> float:
        """Trains the simple linear model using F0Z stabilization."""
        try:
            X_t = torch.tensor(X, dtype=torch.float32)
            Y_t = torch.tensor(Y, dtype=torch.float32)

            # Reshape Y if necessary to match output dimensions
            if Y_t.dim() == 1 and self.weights.shape[1] == 1:
                 Y_t = Y_t.unsqueeze(1)
            elif Y_t.dim() == 0 and self.weights.shape[1] == 1:
                 Y_t = Y_t.unsqueeze(0).unsqueeze(0)

            # Basic input validation
            if X_t.shape[0] != Y_t.shape[0]: raise ValueError("Number of samples in X and Y must match.")
            if X_t.shape[1] != self.weights.shape[0]: raise ValueError(f"Input features mismatch: X has {X_t.shape[1]}, model expects {self.weights.shape[0]}.")
            if Y_t.shape[1] != self.weights.shape[1]: raise ValueError(f"Output features mismatch: Y has {Y_t.shape[1]}, model expects {self.weights.shape[1]}.")

        except Exception as e:
             print(f"Error processing training data: {e}")
             raise # Re-raise error

        print(f"  Training model on {X_t.shape[0]} samples for {epochs} epochs...")
        last_loss = float('inf')
        for epoch in range(epochs):
            # Forward pass: y_pred = X @ W + b
            logits = X_t @ self.weights + self.bias
            # Apply F0Z stabilization to model output (logits)
            y_pred = self.math_module.f0z_stabilize(logits, system_size=X_t.shape[0])

            # Loss calculation (MSE Loss)
            loss = torch.mean((y_pred - Y_t)**2)
            loss_s = self.math_module.f0z_stabilize(loss, system_size=1) # Stabilize loss value

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss_s.backward() # Backpropagate the stabilized loss

            # Optional: Stabilize gradients before optimizer step
            # for param in [self.weights, self.bias]:
            #     if param.grad is not None:
            #         param.grad = self.math_module.f0z_stabilize(param.grad, system_size=param.numel())

            self.optimizer.step()

            last_loss = loss_s.item()
            if (epoch + 1) % (epochs // 2 + 1) == 0: # Print progress occasionally
                 print(f"    Epoch {epoch+1}/{epochs}, Loss: {last_loss:.6f}")

        print(f"  Training complete. Final Loss: {last_loss:.6f}")
        return last_loss


    def predict(self, X: Any) -> List:
        """Makes predictions using the trained linear model."""
        try:
             X_t = torch.tensor(X, dtype=torch.float32)
             if X_t.dim() == 1 and self.weights.shape[0] == 1: # Handle single feature input
                 X_t = X_t.unsqueeze(1)
             elif X_t.dim() == 1 and self.weights.shape[0] != 1:
                  raise ValueError(f"Input has 1 feature, model expects {self.weights.shape[0]}.")
             elif X_t.dim() > 1 and X_t.shape[1] != self.weights.shape[0]:
                  raise ValueError(f"Input features mismatch: X has {X_t.shape[1]}, model expects {self.weights.shape[0]}.")

        except Exception as e:
             print(f"Error processing prediction data: {e}")
             raise # Re-raise error

        print(f"  Predicting for {X_t.shape[0]} samples...")
        with torch.no_grad(): # No need to track gradients during prediction
            logits = X_t @ self.weights + self.bias
            # Stabilize final prediction
            y_pred = self.math_module.f0z_stabilize(logits, system_size=X_t.shape[0])
        print("  Prediction complete.")
        return y_pred.tolist()


    def get_domain_specific_state(self) -> Optional[Dict]:
        # Share model parameters (or hash/summary) and recent loss
        return {
            "model_weights_norm": torch.norm(self.weights).item(),
            "model_bias_norm": torch.norm(self.bias).item(),
            "recent_loss": self.loss_history[-1] if self.loss_history else None
        }

    def receive_state(self, state: Dict):
        super().receive_state(state)
        # Could implement model averaging or parameter updates based on peer state
        domain_data = state.get("domain_data", {})
        peer_loss = domain_data.get("recent_loss")
        my_loss = self.loss_history[-1] if self.loss_history else None
        # Example: If peer loss is significantly better, maybe slightly adjust weights towards peer? (Complex)
        if peer_loss is not None and my_loss is not None and peer_loss < my_loss * 0.8:
             print(f"{self.name} observed better loss from peer ({peer_loss:.4f} vs {my_loss:.4f}). Considering adaptation (placeholder).")


class ValidationAgent(DFSNAgent):
    """Agent responsible for validating results produced by other agents."""
    def __init__(self, name: str):
        super().__init__(name)
        self.validation_history = [] # Store True/False results
        print(f"ValidationAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles validation tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "validation":
            if action == "verify_result":
                error = self.check_task_requirements(task, ["original_result", "task_data"])
                if error: return error
                is_valid = self.verify_result(task["original_result"], task["task_data"])
                self.validation_history.append(is_valid)
                return {"result": {"is_valid": is_valid}, "agent": self.name}

        return super()._execute_single_task_iteration(task) # Delegate other tasks

    def verify_result(self, original_result: Any, task_data: Dict) -> bool:
        """Verifies a result by re-running or checking constraints (simplified)."""
        task_type = task_data.get("type")
        action = task_data.get("action")
        print(f"Validating result for task type '{task_type}', action '{action}'...")

        # --- Verification Logic (Task-Specific) ---
        is_valid = True # Default to valid unless checks fail
        tolerance = 1e-3 # Tolerance for numeric comparisons

        try:
            if task_type == "machine_learning" and action == "predict":
                # Re-run prediction (needs access to model state - complex)
                # Simplified check: Ensure output format/type is correct
                if not isinstance(original_result, dict) or "predictions" not in original_result:
                     is_valid = False; print("  Fail: ML prediction result format incorrect.")
                elif not isinstance(original_result["predictions"], list):
                     is_valid = False; print("  Fail: ML predictions are not a list.")
                # Could add range checks if expected output range is known

            elif task_type == "physics_simulation" and action == "fluid_dynamics":
                 # Check if velocity list has expected length/numeric values
                 res_data = original_result.get("result", {})
                 vel = res_data.get("velocity")
                 grid_size = task_data.get("grid_size")
                 if not isinstance(vel, list): is_valid = False; print("  Fail: Velocity is not a list.")
                 elif grid_size and len(vel) != grid_size: is_valid = False; print(f"  Fail: Velocity length mismatch ({len(vel)} vs {grid_size}).")
                 elif not all(isinstance(v, (int, float)) for v in vel): is_valid = False; print("  Fail: Velocity contains non-numeric values.")
                 # Could add physics constraints checks (e.g., max velocity)

            elif task_type == "temporal_forecast":
                 res_data = original_result.get("result", {})
                 forecast = res_data.get("forecast")
                 if not isinstance(forecast, (int, float)): is_valid = False; print("  Fail: Forecast is not numeric.")
                 # Could check if forecast is within reasonable bounds based on input data

            # Add more verification rules for other task types...

            else:
                print(f"  No specific verification rules for task type '{task_type}/{action}'. Assuming valid.")

        except Exception as e:
            print(f"  Error during verification: {e}. Marking as invalid.")
            is_valid = False

        print(f"  Verification result: {'Valid' if is_valid else 'Invalid'}")
        return is_valid


    def get_domain_specific_state(self) -> Optional[Dict]:
        # Share validation success rate
        rate = sum(self.validation_history) / len(self.validation_history) if self.validation_history else 0.0
        return {"validation_success_rate": rate, "validations_performed": len(self.validation_history)}


class FractalAgent(DFSNAgent):
    """Agent for generating and analyzing fractals using F0Z."""
    def __init__(self, name: str, config: Optional[Dict]=None, primes: Optional[List[int]]=None): # Matches TemporalPrimeAgent structure
        super().__init__(name)
        default_config = {"prime_limit": 100}
        self.config = config if config else default_config
        self.primes = primes if primes else self._generate_primes(self.config["prime_limit"])
        print(f"FractalAgent {self.name} initialized.")

    def _generate_primes(self, n): # Duplicated from TemporalPrimeAgent - should be in a utility module
        primes = []
        is_prime = [True] * (n + 1)
        if n >= 0: is_prime[0] = False
        if n >= 1: is_prime[1] = False
        for p in range(2, int(np.sqrt(n)) + 1):
            if is_prime[p]:
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
        for p in range(2, n + 1):
            if is_prime[p]:
                primes.append(p)
        return primes if primes else [2, 3, 5]

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles fractal generation tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "fractal_generate": # Assuming type implies action here
             # Keys based on RealTimeForecastingManager example usage
             required_keys = ["initial_c", "iterations", "embeddings", "gamma_factor",
                              "hnn_state", "x_t", "delta", "gamma", "h_t", "use_dl"]
             error = self.check_task_requirements(task, required_keys)
             if error: return error

             try:
                 fractal_data, final_z = self.fractal_generator(
                     task["initial_c"], task["iterations"],
                     task["embeddings"], task["gamma_factor"], task["hnn_state"],
                     task["x_t"], task["delta"], task["gamma"], task["h_t"], task["use_dl"]
                 )
                 # Compute fractal dimension (placeholder)
                 dimension = self.detect_fractals({"trajectory": fractal_data})
                 return {"result": {"fractal_sequence": fractal_data, "final_z": final_z, "fractal_dimension": dimension}, "agent": self.name}
             except Exception as e:
                  return {"error": f"Fractal generation failed: {e}", "agent": self.name}

        elif action == "detect_fractals": # Task to analyze existing data
             error = self.check_task_requirements(task, ["data"])
             if error: return error
             dimension = self.detect_fractals({"trajectory": task["data"]}) # Pass data structure expected by detect_fractals
             return {"result": {"fractal_dimension": dimension}, "agent": self.name}


        return super()._execute_single_task_iteration(task) # Delegate other tasks

    def fractal_generator(self, initial_c, iterations, embeddings, gamma_factor, hnn_state, x_t, delta, gamma, h_t, use_dl):
        """
        Generates a fractal-like sequence based on complex dynamics influenced by multiple inputs,
        using F0Z stabilization. Adapted from RealTimeForecastingManager example.
        """
        if not self.primes: return [], 0.0 # Need primes

        # Ensure inputs are usable numpy arrays or scalars
        embeddings = np.array(embeddings) if isinstance(embeddings, list) else embeddings
        hnn_state = np.array(hnn_state) if isinstance(hnn_state, list) else hnn_state
        x_t = np.array(x_t) if isinstance(x_t, list) else x_t
        h_t = np.array(h_t) if isinstance(h_t, list) else h_t

        # Calculate mean values, handle potential empty arrays
        mu_p = np.mean(self.primes)
        mu_e = np.mean(embeddings) if embeddings.size > 0 else 0.0
        mean_hnn = np.mean(hnn_state) if hnn_state.size > 0 else 0.0
        mean_xt = np.mean(x_t) if x_t.size > 0 else 0.0
        mean_ht = np.mean(h_t) if h_t.size > 0 else 0.0

        # Use median for robust error, handle potential NaN from mean_xt
        pred = mu_e # Simple prediction based on embeddings mean
        robust_error = np.median([mean_xt - pred, 0.1]) if not np.isnan(mean_xt) else 0.1

        # Initial complex value z
        z = torch.tensor(initial_c, dtype=torch.complex64) # Start with initial_c as complex
        fractal_sequence = []

        print(f"  Generating fractal sequence ({iterations} iterations)...")
        for t in range(iterations):
            # Calculate the complex parameter 'c' dynamically
            # Combine influences: primes, embeddings, iteration, hnn, input data, error, hidden state, dl flag
            c_real = (mu_p + mu_e +
                      t * gamma_factor + # Iteration influence
                      mean_hnn + # Hopfield state influence
                      0.2 * mean_xt + # Recent input data influence
                      delta * robust_error + # Error feedback
                      gamma * mean_ht * use_dl) # Influence from temporal DL state

            # Introduce imaginary part based on variance or other metric (example)
            c_imag = 0.1 * np.std(x_t) if x_t.size > 1 else 0.01
            c_imag = np.nan_to_num(c_imag) # Handle NaN if std dev fails

            c = torch.tensor(c_real + 1j * c_imag, dtype=torch.complex64)
            c_s = self.math_module.f0z_stabilize(c, system_size=iterations) # Stabilize c

            # Apply the fractal iteration: z = z^2 + c
            z = z**2 + c_s
            z = self.math_module.f0z_stabilize(z, system_size=iterations) # Stabilize z

            # Optional: Bounding z to prevent explosion (though F0Z helps)
            # z = z / (1 + torch.abs(z)) # Normalize/bound

            fractal_sequence.append(z.item()) # Store complex value

        # Add variance of the sequence (real part) to performance history
        real_parts = [v.real for v in fractal_sequence]
        variance = F0ZAlgebra.f0z_variance(real_parts) if real_parts else 0.0
        self.performance_history.append(variance)

        final_z = fractal_sequence[-1] if fractal_sequence else initial_c + 0j
        print(f"  Fractal generation complete. Final z = {final_z:.3f}")
        return fractal_sequence, final_z # Return sequence and final complex value


    def detect_fractals(self, data: Dict) -> float:
        """Estimates fractal dimension (placeholder)."""
        trajectory = data.get("trajectory")
        if trajectory is None or len(trajectory) < 10:
            return 0.0 # Not enough data

        # Example: Box-counting dimension (simplified)
        # This requires a proper implementation, treating trajectory as points in space
        # Placeholder: Return a value based on variance or complexity
        try:
             # Use variance of real part as proxy for complexity/fractal nature
             if isinstance(trajectory[0], complex):
                 real_parts = [v.real for v in trajectory]
                 variance = np.var(real_parts)
             elif isinstance(trajectory[0], (list, np.ndarray)): # Handle simulation trajectories
                 flat_traj = np.array(trajectory).flatten()
                 numeric_traj = flat_traj[np.isfinite(flat_traj)]
                 variance = np.var(numeric_traj) if numeric_traj.size > 0 else 0.0
             else: # Assume list of numbers
                 numeric_traj = np.array(trajectory)[np.isfinite(trajectory)]
                 variance = np.var(numeric_traj) if numeric_traj.size > 0 else 0.0

             # Map variance to a dimension-like value (0 to 2)
             dimension = 1.0 + np.tanh(variance / 10.0) # Scale variance effect
             dimension = np.clip(dimension, 1.0, 2.0) # Typical range for simple fractals
             print(f"  Estimated fractal dimension (proxy): {dimension:.3f} (based on variance {variance:.3f})")
             return dimension
        except Exception as e:
            print(f"Error calculating fractal dimension proxy: {e}")
            return 0.0


    def get_domain_specific_state(self) -> Optional[Dict]:
        # Share prime numbers used
        return {"primes_used_sample": self.primes[:10]}


class HopfieldAgent(DFSNAgent):
    """Agent for Hopfield Neural Network operations."""
    def __init__(self, name: str, config: Optional[Dict]=None, primes: Optional[List[int]]=None):
        super().__init__(name)
        default_config = {"hnn_size": 100, "sparsity": 0.7, "prime_limit": 100}
        self.config = config if config else default_config
        self.size = self.config.get("hnn_size", 100)
        self.primes = primes if primes else self._generate_primes(self.config["prime_limit"])
        # Initialize weights with F0Z in mind (potentially small random values)
        self.weights = torch.randn(self.size, self.size, dtype=torch.float32) * 0.01
        self.weights.fill_diagonal_(0) # No self-connections
        self.sparsity = self.config.get("sparsity", 0.7)
        self._apply_sparsity()
        print(f"HopfieldAgent {self.name} initialized (Size: {self.size}, Sparsity: {self.sparsity}).")

    def _generate_primes(self, n): # Duplicated code - refactor needed
        primes = []
        is_prime = [True] * (n + 1)
        if n >= 0: is_prime[0] = False
        if n >= 1: is_prime[1] = False
        for p in range(2, int(np.sqrt(n)) + 1):
            if is_prime[p]:
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
        for p in range(2, n + 1):
            if is_prime[p]:
                primes.append(p)
        return primes if primes else [2, 3, 5]

    def _apply_sparsity(self):
         """Applies sparsity mask to the weights."""
         mask = torch.rand(self.weights.size()) > self.sparsity
         self.weights.data *= mask.float()
         print(f"  Applied sparsity ({self.sparsity}) to Hopfield weights.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles Hopfield network tasks like state updates."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "hnn_update": # Assuming type implies action
            required_keys = ["state", "context_vector", "zero_equilibrium", "alpha", "t",
                             "x_t", "delta", "gamma", "h_t", "use_dl"]
            error = self.check_task_requirements(task, required_keys)
            if error: return error

            try:
                 new_state = self.update_state(
                     task["state"], task["context_vector"], task["zero_equilibrium"], task["alpha"], task["t"],
                     task["x_t"], task["delta"], task["gamma"], task["h_t"], task["use_dl"]
                 )
                 return {"result": {"hnn_state": new_state.tolist()}, "agent": self.name}
            except Exception as e:
                 return {"error": f"HNN update failed: {e}", "agent": self.name}
        elif task_type == "hnn_train":
             error = self.check_task_requirements(task, ["patterns"])
             if error: return error
             self.train(task["patterns"])
             return {"status": "HNN trained", "agent": self.name}

        return super()._execute_single_task_iteration(task) # Delegate other tasks


    def train(self, patterns: List[List[int]]):
        """Trains the Hopfield network using Hebbian rule with F0Z and primes."""
        if not patterns or not isinstance(patterns[0], (list, np.ndarray)):
             print("Error: HNN training requires a list of patterns (lists/arrays).")
             return

        num_patterns = len(patterns)
        pattern_len = len(patterns[0])
        if pattern_len != self.size:
             print(f"Error: Pattern length ({pattern_len}) does not match network size ({self.size}).")
             return

        print(f"Training Hopfield network with {num_patterns} patterns...")
        # Hebbian rule: W_ij = sum_p (pattern_p_i * pattern_p_j) / N
        new_weights = torch.zeros_like(self.weights)
        patterns_t = torch.tensor(patterns, dtype=torch.float32) # Convert patterns to tensor [-1, 1]

        # Incorporate primes into training (example: weighted sum)
        prime_weights = torch.tensor(self.primes[:num_patterns] / np.sum(self.primes[:num_patterns]), dtype=torch.float32) if self.primes else torch.ones(num_patterns) / num_patterns
        if len(prime_weights) < num_patterns: # Repeat primes if not enough
            prime_weights = prime_weights.repeat(num_patterns // len(prime_weights) + 1)[:num_patterns]

        for p_idx in range(num_patterns):
            pattern = patterns_t[p_idx]
            # Outer product: pattern.view(-1, 1) * pattern.view(1, -1)
            weight_update = torch.outer(pattern, pattern) * prime_weights[p_idx]
            new_weights += weight_update

        # Normalize weights (optional) and apply F0Z
        new_weights /= num_patterns # Average contribution
        new_weights.fill_diagonal_(0) # Ensure no self-connections
        stabilized_weights = self.math_module.f0z_stabilize(new_weights, system_size=self.size*self.size)

        # Combine with existing weights (e.g., incremental learning) or replace
        self.weights = 0.8 * self.weights + 0.2 * stabilized_weights # Moving average update
        self._apply_sparsity() # Re-apply sparsity after training
        print("Hopfield network training complete.")


    def update_state(self, state, context_vector, zero_equilibrium, alpha, t, x_t, delta, gamma, h_t, use_dl):
        """Updates the network state asynchronously with F0Z and external influences."""
        state_t = torch.tensor(state, dtype=torch.float32)
        context_t = torch.tensor(context_vector, dtype=torch.float32) if context_vector is not None else torch.zeros(self.size)
        xt_t = torch.tensor(x_t[-self.size:], dtype=torch.float32) if x_t else torch.zeros(self.size) # Use recent part of x_t
        ht_t = torch.tensor(h_t, dtype=torch.float32) if h_t is not None else torch.zeros(self.size) # Use DL hidden state

        if len(xt_t) < self.size: xt_t = torch.cat((torch.zeros(self.size - len(xt_t)), xt_t)) # Pad if needed
        if len(ht_t) < self.size: ht_t = torch.cat((torch.zeros(self.size - len(ht_t)), ht_t)) # Pad if needed


        # Calculate input signal: h_i = sum_j (W_ij * s_j)
        signal = self.weights @ state_t

        # Calculate inhibition term based on context and equilibrium
        inhibition_base = zero_equilibrium + torch.dot(context_t[:self.size], state_t) # Ensure context matches size
        flow_factor = 2.5 if self.flow_state == "flow" else 1.0
        inhibition = inhibition_base * flow_factor

        # Calculate prediction and robust error (using stabilized mean)
        pred = self.math_module.f0z_stabilize(torch.mean(state_t)).item()
        mean_xt = self.math_module.f0z_stabilize(torch.mean(xt_t)).item()
        robust_error = np.median([mean_xt - pred, 0.1])

        # Prime influence for this step
        prime_influence = alpha * self.primes[t % len(self.primes)] if self.primes else 0.0

        # Combine all influences
        combined_input = (signal - inhibition +
                          prime_influence +
                          0.2 * xt_t + # Direct influence from recent input data
                          delta * robust_error + # Error feedback term
                          gamma * ht_t * use_dl) # Influence from temporal DL state

        # Apply F0Z stabilization to the combined input before activation
        stabilized_input = self.math_module.f0z_stabilize(combined_input, system_size=self.size)

        # Activation function (sign function for Hopfield)
        new_state = torch.sign(stabilized_input)
        # Ensure output is binary (-1 or 1), handle zero case
        new_state = torch.where(new_state == 0, torch.tensor(1.0), new_state)

        # Update performance history (e.g., stability or distance to target pattern)
        # Using distance to the input x_t as a simple measure
        if xt_t.shape == new_state.shape:
             dist = torch.mean(torch.abs(new_state - torch.sign(xt_t))) # Compare signs
             self.performance_history.append(1.0 - dist.item()) # Performance = 1 - distance
        else:
             self.performance_history.append(0.5) # Default performance if shapes mismatch

        # print(f"  Hopfield state updated. Change magnitude: {torch.norm(new_state - state_t):.3f}")
        return new_state.numpy() # Return as numpy array


    def get_domain_specific_state(self) -> Optional[Dict]:
        # Share weight matrix norm or sparsity info
        return {"weight_norm": torch.norm(self.weights).item(), "sparsity": self.sparsity}

    def receive_state(self, state: Dict):
        super().receive_state(state)
        # Could average weights with peer if compatible size/sparsity
        domain_data = state.get("domain_data", {})
        if "weights" in domain_data and isinstance(domain_data["weights"], list):
             try:
                 peer_weights = torch.tensor(domain_data["weights"])
                 if peer_weights.shape == self.weights.shape:
                     # Average weights
                     self.weights = 0.7 * self.weights + 0.3 * peer_weights
                     self.weights.fill_diagonal_(0) # Ensure no self-connections
                     self._apply_sparsity() # Re-apply sparsity
                     print(f"{self.name} averaged weights with peer.")
                 else:
                      print(f"{self.name} received incompatible weights from peer.")
             except Exception as e:
                  print(f"{self.name} failed to process peer weights: {e}")



class CollaborativeAgent(DFSNAgent):
    """Agent focused on collaboration, state sharing, and synchronization."""
    def __init__(self, name: str):
        super().__init__(name); self.peers: List[str] = []; self.shared_data: Dict[str, Any] = {}
        self.agent_registry: Optional[Dict[str, Agent]] = None; print(f"CollaborativeAgent {self.name} initialized.")
    def set_agent_registry(self, registry: Dict[str, Agent]): self.agent_registry = registry
    def add_peer(self, peer_name: str):
        if peer_name not in self.peers: self.peers.append(peer_name); print(f"{self.name} added peer: {peer_name}")
    def find_peer_object(self, peer_name: str) -> Optional[Agent]:
         if not self.agent_registry: print(f"Error: Agent registry not set for {self.name}."); return None
         peer = self.agent_registry.get(peer_name); # if not peer: print(f"Error: Peer '{peer_name}' not found.") # Reduce noise
         return peer
    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        action = task.get("action"); task_type = task.get("type")
        print(f"DEBUG: CollabAgent received task: type='{task_type}', action='{action}'") # <-- ADD DEBUG

        # Handle science fair experiment first (as corrected before)
        if task_type == "science_fair_experiment":
            exp_desc = task.get('data', {}).get('prompt', task.get('description', 'N/A'))
            print(f"{self.name} running science fair experiment: {exp_desc}")
            time.sleep(0.1 * task.get('complexity', 1.0)); result_value = random.uniform(0.5, 1.0)
            # Ensure the key matches what ZSGBatchModeScienceFair expects
            return {"result": {"experiment_outcome": result_value}, "agent": self.name}

        # ... (collaborate, synchronize_with_peers, receive_state, get_domain_specific_state methods) ...
        # THEN handle collaboration tasks
        elif task_type == "collaboration":
            print(f"DEBUG: CollabAgent handling collaboration action: '{action}'") # <-- ADD DEBUG

            if action == "collaborate_on_task":
                print(f"DEBUG: CollabAgent executing COLLAB ON TASK action.") # <-- ADD DEBUG
                error = self.check_task_requirements(task, ["data", "peer_names"]) # Check task['data']
                if error: return error
                # Pass task['data']['task_data'] if it's nested like that, otherwise just task['data']
                task_content = task.get("data", {}).get("task_data", task.get("data"))
                peer_names = task.get("data", {}).get("peer_names", [])
                result = self.collaborate(task_content, peer_names)
                return {"result": result, "agent": self.name} # Return result directly

            elif action == "synchronize":
                self.synchronize_with_peers(); return {"status": "sync complete", "agent": self.name}
                print(f"DEBUG: CollabAgent executing SYNCHRONIZE action.") # <-- ADD DEBUG

            elif action == "combine": # From Science Fair example
                print(f"DEBUG: CollabAgent executing COMBINE action.") # <-- ADD DEBUG
                error = self.check_task_requirements(task, ["data"]) # Checks task['data'] exists
                if error: return error
                # --- Combine Logic ---
                combined_data = {}
                # task["data"] should be {"batch_results": [...], "forecast": ...}
                combine_input = task.get("data", {})
                for key, value in combine_input.items():
                     combined_data[f"{key}_combined"] = value # Add suffix
                print(f"{self.name}: Combined data for keys: {list(combine_input.keys())}")
                return {"result": combined_data, "agent": self.name} # Return combined data
            else:
                 # Unrecognized collaboration action
                 return {"error": f"Unsupported collaboration action: {action}", "agent": self.name}

        # Delegate tasks not handled above
        return super()._execute_single_task_iteration(task)

    # Need to update collaborate method slightly based on the change above
    def collaborate(self, task_content: Any, peer_names: List[str]) -> Dict:
        print(f"{self.name} initiating collaboration on task with peers: {peer_names}")
        if not self.agent_registry: return {"error": "Agent registry not available."}
        results = {}
        for peer_name in peer_names:
            peer = self.find_peer_object(peer_name)
            if peer and isinstance(peer, DFSNAgent):
                print(f"  Sharing state and task with {peer_name}")
                self.share_state(peer)
                # Create the sub-task for the peer
                peer_task = {
                    "type": "collaborative_subtask", # Give it a type
                    "action": "process_collab_data", # Define an action for the peer
                    "data": task_content, # Pass the actual task content
                    "collaborator": self.name,
                    "complexity": 5.0 # Assign complexity
                }
                peer_result = peer.execute_task(peer_task) # Use execute_task -> AIW
                results[peer_name] = peer_result
            else: results[peer_name] = {"error": "Peer unavailable"}
        final_result = {"collaboration_summary": f"Results from {len(results)} peers.", "details": results}
        return final_result

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        action = task.get("action"); task_type = task.get("type")
        # --- Logic from previous correction ---
        if task.get("type") == "science_fair_experiment":
            exp_desc = task.get('data', {}).get('prompt', task.get('description', 'N/A')) # Corrected description access
            print(f"{self.name} running science fair experiment: {exp_desc}")
            time.sleep(0.1 * task.get('complexity', 1.0)); result_value = random.uniform(0.5, 1.0)
            return {"result": {"experiment_outcome": result_value}, "agent": self.name}
        # --- End Logic from previous correction ---
        elif task_type == "collaboration":
            if action == "collaborate_on_task": error = self.check_task_requirements(task, ["task_data", "peer_names"]); # ... (rest of collab logic) ...
            elif action == "synchronize": self.synchronize_with_peers(); return {"status": "sync complete", "agent": self.name}
            elif action == "combine": error = self.check_task_requirements(task, ["data"]); # ... (rest of combine logic) ...
        return super()._execute_single_task_iteration(task)

# --- Qiskit Integration ---

# --- ZSGQuantumSimulator Modification ---
class ZSGQuantumSimulator:
    """Simulates quantum circuits using Qiskit AerSimulator."""
    def __init__(self, n_qubits: int, noise_level: float = 0.01, use_noise: bool = True):
        if QuantumCircuit is None: raise ImportError("Qiskit not found.")
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)
        self.math_module = PyZeroMathTorch()
        self.noise_level = noise_level
        self.simulator = AerSimulator() # Default ideal simulator
        if use_noise and depolarizing_error:
            print(f"Setting up Qiskit noise model (Depolarizing p={self.noise_level})...")
            self.noise_model = NoiseModel()
            error_1q = depolarizing_error(self.noise_level, 1)
            error_2q = depolarizing_error(self.noise_level * 5, 2)
            self.noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z', 'rz', 'ry']) # Added Y, Z
            self.noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
            self.simulator = AerSimulator(noise_model=self.noise_model)
            print("Qiskit AerSimulator configured with noise model.")
        else: self.noise_model = None; print("Qiskit AerSimulator configured without noise.")

    def _reset_circuit(self): self.circuit = QuantumCircuit(self.n_qubits)
    # --- Redefined gate methods ---
    def h_gate(self, qubit: int): self.circuit.h(qubit)
    def x_gate(self, qubit: int): self.circuit.x(qubit)
    def y_gate(self, qubit: int): self.circuit.y(qubit)
    def z_gate(self, qubit: int): self.circuit.z(qubit)
    def rz_gate(self, angle: float, qubit: int): self.circuit.rz(angle, qubit)
    def ry_gate(self, angle: float, qubit: int): self.circuit.ry(angle, qubit)
    def cz_gate(self, control_qubit: int, target_qubit: int): self.circuit.cz(control_qubit, target_qubit)
    def cnot_gate(self, control_qubit: int, target_qubit: int): self.circuit.cx(control_qubit, target_qubit)

    def measure(self, qubit_indices: Optional[List[int]] = None, shots: int = 1024) -> Dict:
        if qubit_indices is None: qubit_indices = list(range(self.n_qubits))
        meas_circuit = self.circuit.copy(); num_classical_bits = len(qubit_indices)
        if num_classical_bits > 0:
            cr_name = 'c' # Default classical register name
            if not meas_circuit.cregs or cr_name not in [reg.name for reg in meas_circuit.cregs]:
                 cr = ClassicalRegister(num_classical_bits, name=cr_name)
                 meas_circuit.add_register(cr)
            else:
                 # Ensure existing register is large enough
                 existing_cr = next(reg for reg in meas_circuit.cregs if reg.name == cr_name)
                 if existing_cr.size < num_classical_bits:
                      # This is tricky - might need to remove/add register or handle error
                      print(f"Warning: Existing classical register '{cr_name}' is too small.")
                      # Attempt to use available bits
                      qubit_indices = qubit_indices[:existing_cr.size]
                      num_classical_bits = len(qubit_indices)
                      if num_classical_bits == 0: return {"counts": {}, "notes": "Classical register size mismatch"}

            # Map qubits to classical bits
            meas_map = [(qubit_indices[i], i) for i in range(num_classical_bits)]
            meas_circuit.measure([m[0] for m in meas_map], [m[1] for m in meas_map]) # Measure specified qubits to first N classical bits

            # transpiled_circuit = transpile(meas_circuit, self.simulator) # Often needed
            try:
                 job = self.simulator.run(meas_circuit, shots=shots); result = job.result(); counts = result.get_counts(0)
                 total_shots = sum(counts.values())
                 probabilities = {state: count / total_shots for state, count in counts.items()}
                 return {"counts": counts, "probabilities": probabilities, "shots": shots}
            except Exception as e:
                 print(f"Qiskit simulation error: {e}"); return {"error": str(e)}
        else: return {"counts": {}, "probabilities": {}, "shots": shots, "notes": "No qubits measured."}

    def get_statevector(self) -> np.ndarray:
         if self.noise_model: print("Warning: Cannot get exact statevector with noise."); return np.array([])
         sv_sim = AerSimulator(method='statevector')
         sv_circuit = self.circuit.copy(); sv_circuit.save_statevector()
         try:
              job = sv_sim.run(sv_circuit); result = job.result(); statevector = result.get_statevector(0).data
              stabilized_sv = self.math_module.f0z_stabilize(torch.tensor(statevector, dtype=torch.complex64)).numpy()
              return stabilized_sv
         except Exception as e: print(f"Qiskit statevector error: {e}"); return np.array([])

# --- QRL Policy Network (Required by Qiskit-based QRL Agent) ---

class TemporalPrimeAgent(DFSNAgent):
    """Agent focused on time-series analysis and forecasting using prime number properties."""
    def __init__(self, name: str, config: Optional[Dict]=None, primes: Optional[List[int]]=None):
        super().__init__(name)
        # Default config if none provided
        default_config = {"prime_limit": 100, "memory_horizon": 20, "tau": 5.0, "use_deep_learning": False, "hidden_size": 10, "sparsity": 0.7}
        self.config = config if config else default_config

        self.primes = primes if primes else self._generate_primes(self.config["prime_limit"])
        self.time_scale = 1.0
        self.forecast_history = []
        self.memory_horizon = self.config["memory_horizon"]
        self.tau = self.config.get("tau", 5.0)
        self.state_history: List[float] = [] # History of internal state z_t
        self.use_dl = self.config.get("use_deep_learning", False)
        self.lstm = None
        self.lstm_optimizer = None
        if self.use_dl:
            self.lstm = SparseLSTMModel(input_size=1, hidden_size=self.config["hidden_size"], sparsity=self.config.get("sparsity", 0.7))
            self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.01)
            print(f"TemporalPrimeAgent {self.name} initialized with LSTM.")
        else:
             print(f"TemporalPrimeAgent {self.name} initialized (No DL). Primes up to {self.config['prime_limit']}.")


    def _generate_primes(self, n):
        """Generates prime numbers up to n using Sieve of Eratosthenes."""
        primes = []
        is_prime = [True] * (n + 1)
        if n >= 0: is_prime[0] = False
        if n >= 1: is_prime[1] = False
        for p in range(2, int(np.sqrt(n)) + 1):
            if is_prime[p]:
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
        for p in range(2, n + 1):
            if is_prime[p]:
                primes.append(p)
        return primes if primes else [2, 3, 5] # Fallback primes

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Handles temporal forecasting and scaling tasks."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "temporal_forecast":
             error = self.check_task_requirements(task, ["data", "horizon"])
             if error: return error
             forecast_result = self.forecast(task["data"], task["horizon"])
             self.forecast_history.append(forecast_result)
             return {"result": {"forecast": forecast_result}, "agent": self.name}
        elif task_type == "temporal_scaling": # From RealTimeForecastingManager example
             error = self.check_task_requirements(task, ["beta", "eta", "lambda", "delta", "gamma", "t", "x_t"])
             if error: return error
             z_t, pred, error_val = self.temporal_scaling(
                 task["beta"], task["eta"], task["lambda"], task["delta"], task["gamma"], task["t"], task["x_t"]
             )
             return {"result": {"z_t": z_t, "pred": pred, "error": error_val}, "agent": self.name}

        # Delegate other tasks to AIW/base
        return super()._execute_single_task_iteration(task)

    def forecast(self, data: Any, horizon: int) -> float:
        """Simple forecasting based on recent stabilized average."""
        # Extract recent numeric values from various data types
        if isinstance(data, dict):
             # Try to find numeric values in the dict, prioritize keys like 'value', 'result'
             values = [v for k, v in data.items() if isinstance(v, (int, float))]
             if not values: values = list(data.values()) # Fallback to all values
        elif isinstance(data, (list, np.ndarray, torch.Tensor)):
             values = np.array(data).flatten()
        elif isinstance(data, (int, float)):
             values = [data]
        else:
             values = []

        numeric_values = [v for v in values if isinstance(v, (int, float))]

        if not numeric_values:
            print(f"{self.name} forecast: No numeric data found.")
            return 0.0

        recent_values = numeric_values[-5:] # Use last 5 numeric points
        if not recent_values: return 0.0

        avg_val = np.mean(recent_values)
        stabilized_avg = self.math_module.f0z_stabilize(torch.tensor(avg_val)).item()

        # Simple projection: scale average by time scale and horizon
        forecast_value = stabilized_avg * self.time_scale * horizon
        # Stabilize the final forecast value as well
        final_forecast = self.math_module.f0z_stabilize(torch.tensor(forecast_value)).item()
        print(f"{self.name} forecast: Avg={stabilized_avg:.3f}, Horizon={horizon}, Forecast={final_forecast:.3f}")
        return final_forecast

    def temporal_scaling(self, beta, eta, lambda_, delta, gamma, t, x_t, noise_threshold=0.1):
        """Applies the Temporal Prime Scaling update rule."""
        if not self.primes: # Ensure primes are available
             return 0.0, 0.0, x_t # Return default values if no primes

        # Get previous internal state z_{t-1}
        z_prev = self.state_history[-1] if self.state_history else 0.0

        # Calculate memory term based on past states
        memory_term = 0.0
        flow_factor = 2.5 if self.flow_state == "flow" else 1.0 # Influence of flow state
        horizon = min(self.memory_horizon, len(self.state_history))
        for k in range(1, horizon + 1):
            weight = np.exp(-k / self.tau) * (1 + 0.15 * flow_factor) # Weight decays exponentially, boosted by flow
            memory_term += weight * self.state_history[-k]

        # Handle deep learning component if enabled
        h_t = torch.zeros(self.config["hidden_size"]) # Default hidden state
        if self.use_dl and self.lstm:
             try:
                 # Prepare input for LSTM (batch size 1, seq length 1)
                 lstm_input = torch.tensor([[x_t]], dtype=torch.float32)
                 lstm_output = self.lstm(lstm_input) # Get output from LSTM
                 h_t = lstm_output.detach().squeeze().numpy() # Convert to numpy array
                 # Simple LSTM training step (online learning)
                 # We need a target to train. Let's predict the *next* x_t based on current h_t
                 # This requires storing next_x_t or modifying the structure.
                 # Simplified: Train based on reconstructing current x_t from h_t (autoencoder style)
                 # output_layer = nn.Linear(self.config["hidden_size"], 1) # Assume a simple output layer
                 # predicted_x = output_layer(lstm_output)
                 # loss = nn.MSELoss()(predicted_x, torch.tensor([[x_t]]))
                 # self.lstm_optimizer.zero_grad()
                 # loss.backward()
                 # self.lstm_optimizer.step()
             except Exception as e:
                 print(f"Error during LSTM processing/training: {e}")
                 h_t = np.zeros(self.config["hidden_size"]) # Fallback


        # Predict next value based on memory and DL state
        # Prediction combines memory term and last element of hidden state
        prediction = memory_term + 0.5 * h_t[-1] * self.use_dl
        pred_s = self.math_module.f0z_stabilize(torch.tensor(prediction)).item()

        # Calculate robust error (median absolute deviation from prediction, clipped by noise threshold)
        error = x_t - pred_s
        robust_error = np.sign(error) * min(abs(error), noise_threshold + abs(np.median([error, 0.0])))
        # Alternative robust error: median of [error, threshold_or_zero]
        # robust_error = np.median([error, noise_threshold * np.sign(error)])

        # Calculate new internal state z_t using the TPS formula
        prime_term = self.primes[t % len(self.primes)] # Cyclical prime influence
        dl_term = h_t[-1] * self.use_dl # Contribution from deep learning state

        # Combine terms for the update
        update_val = (z_prev +
                      beta * prime_term +
                      eta * memory_term +
                      lambda_ * x_t + # Direct influence of current observation
                      delta * robust_error + # Influence of prediction error
                      gamma * dl_term) # Influence of DL component

        # Apply tanh activation and F0Z stabilization
        z_t_tensor = torch.tanh(self.math_module.f0z_stabilize(torch.tensor(update_val)))
        z_t = z_t_tensor.item()

        # Update state history and performance history
        self.state_history.append(z_t)
        if len(self.state_history) > self.memory_horizon * 2: # Limit history size
             self.state_history.pop(0)
        # Use absolute error for performance tracking
        self.performance_history.append(abs(robust_error))

        # print(f"  TPS (t={t}): x_t={x_t:.3f}, Mem={memory_term:.3f}, Pred={pred_s:.3f}, Err={robust_error:.3f}, z_t={z_t:.3f}")
        return z_t, pred_s, robust_error


    def get_domain_specific_state(self) -> Optional[Dict]:
        # Share recent internal state and forecast history
        return {"state_history_tail": self.state_history[-5:],
                "forecast_history_tail": self.forecast_history[-5:]}

    def receive_state(self, state: Dict):
        super().receive_state(state)
        # Could potentially align internal state or forecasts based on peer
        if "state_history_tail" in state.get("domain_data", {}):
             # Simple merge/ignore for now
             pass



class QuantumPolicyNetwork(nn.Module):
    """Classical network to output action probabilities based on state embedding."""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_size)
        self.softmax = nn.Softmax(dim=-1) # Output probabilities

    def forward(self, state_embedding):
        x = self.relu(self.fc1(state_embedding))
        action_logits = self.fc2(x)
        action_probs = self.softmax(action_logits)
        return action_probs

# --- LLM Integration Agent ---

# Remove inheritance from DFSNAgent for F0ZAgent if it's just an interface
class F0ZAgent: # NOT inheriting DFSNAgent
    """Agent integrating a local LLM for inference, simulating F0Z concepts."""
    def __init__(self, name="F0ZAgent_LLM"): # Give it a name
        # Removed super().__init__(name)
        self.name = name # Set name directly
        self.llm = None
        self.math_sim = PyZeroMathTorch()
        self.k=1.0; self.Ea=0.1; self.kB=8.617e-5; self.T=300
        try:
            device = -1 # Default to CPU for the small embedded model
            # Use CPU for gpt2 if Ollama agent handles the main LLM tasks
            self.llm = pipeline("text-generation", model="gpt2", device=device)
            print(f"F0ZAgent LLM interface initialized with: gpt2 on CPU")
        except Exception as e:
            print(f"LLM loading failed for F0ZAgent: {e}. Inference disabled.")
            self.llm = None


    def simulate_f0z_reaction(self, D_Pd_ratio: float, defect_density: float = 0.7, f0z_symmetry: float = 0.95) -> float:
        """Simulates a hypothetical F0Z-influenced reaction rate based on formula in prompt."""
        # R_F0Z = k * (D/Pd)^3.2 * exp(-Ea/kBT) * (1 + 0.14*defect) * (1 + 0.10*symmetry)
        if D_Pd_ratio < 0: D_Pd_ratio = 0 # Ensure base is non-negative

        term1 = self.k * (D_Pd_ratio ** 3.2)
        term2 = np.exp(-self.Ea / (self.kB * self.T))
        term3 = (1.0 + 0.14 * defect_density)
        term4 = (1.0 + 0.10 * f0z_symmetry)

        rate = term1 * term2 * term3 * term4

        # Apply F0Z stabilization to the final rate
        stabilized_rate = self.math_sim.f0z_stabilize(torch.tensor(rate)).item()
        print(f"  Simulated F0Z Reaction Rate: {stabilized_rate:.4e}")
        return stabilized_rate

    def infer(self, query: str, max_length: int = 100) -> str:
        """Uses the integrated LLM to answer queries related to F0Z or other topics."""
        if not self.llm:
            return "LLM not available in F0ZAgent."
        try:
            # Add context to the query for the LLM
            full_prompt = f"Regarding the Formula for Zero (F0Z) concept, {query}"
            # Ensure max_length is reasonable
            max_len = max(20, min(max_length, 200)) # Clamp length
            result = self.llm(full_prompt, max_length=max_len, num_return_sequences=1, do_sample=True)
            generated_text = result[0]['generated_text']
            # Clean up the response (e.g., remove the prompt if model includes it)
            if generated_text.startswith(full_prompt):
                 cleaned_text = generated_text[len(full_prompt):].strip()
            else:
                 cleaned_text = generated_text
            print(f"  LLM Inference: Q='{query[:30]}...' -> A='{cleaned_text[:50]}...'")
            return cleaned_text
        except Exception as e:
            print(f"Error during LLM inference: {e}")
            return f"Error during inference: {e}"

    # Make F0ZAgent behave like a DFSNAgent for integration
    # Remove execute_task if not needed, or simplify it if called directly
    def execute_simulation_task(self, task: Dict) -> Dict:
         """Handles ONLY f0z_simulation tasks directly."""
         if task.get("type") == "f0z_simulation" or task.get("action") == "simulate_f0z_reaction":
              # Basic check, assumes 'data' payload contains keys if needed
              required = task.get("data", {}).get("required_keys", ["D_Pd_ratio"])
              missing = [k for k in required if k not in task.get("data", {})]
              if missing: return {"error": f"F0Z Sim Missing keys: {missing}", "agent": self.name}
              rate = self.simulate_f0z_reaction(
                  task["data"]["D_Pd_ratio"],
                  task.get("data", {}).get("defect_density", 0.7),
                  task.get("data", {}).get("f0z_symmetry", 0.95)
              )
              return {"result": {"reaction_rate": rate}, "agent": self.name}
         return {"error": f"Unsupported task type for F0ZAgent simulation: {task.get('type')}", "agent": self.name}

    def execute_inference_task(self, task: Dict) -> Dict:
         """Handles ONLY llm_inference tasks directly."""
         if task.get("type") == "llm_inference" or task.get("action") == "infer":
              query = task.get("data", {}).get("query", task.get("query")) # Get query from data or root
              if not query: return {"error": "Missing query for inference", "agent": self.name}
              response = self.infer(query, task.get("data", {}).get("max_length", 100))
              return {"result": {"llm_response": response}, "agent": self.name}
         return {"error": f"Unsupported task type for F0ZAgent inference: {task.get('type')}", "agent": self.name}

    # Keep check_task_requirements if used by execute_* methods above

    # Add dummy methods required by DFSNAgent structure if needed for coordination
    #def __init__(self, name="F0ZAgent_LLM"): # Give it a name
    def __init__(self, name="F0ZAgent_LLM"): # Give it a name
        super().__init__() # Call parent init if inheriting (needs DFSNAgent inheritance) - OR just be standalone
        self.llm = None # Re-init LLM here
        self.math_sim = PyZeroMathTorch()
        self.k=1.0; self.Ea=0.1; self.kB=8.617e-5; self.T=300
        try:
            device = 0 if torch.cuda.is_available() else -1
            # Ensure model name is valid or handle error
            valid_models = ["gpt2", "distilgpt2"] # Example small models
            model_to_load = "gpt2" # Default to gpt2
            # If a specific model was intended: model_to_load = "meta-llama/Llama-3.1" # Requires access/download
            self.llm = pipeline("text-generation", model=model_to_load, device=device)
            print(f"F0ZAgent LLM interface initialized with: {model_to_load}")
        except Exception as e:
            print(f"LLM loading failed for F0ZAgent: {e}. Inference disabled.")
            self.llm = None

    def get_engagement_state(self): return 0 # LLM agent might not use DFSN states
    def set_engagement_state(self, state): pass
    def adjust_workload(self, level): pass
    def enter_flow_state(self): pass
    def exit_flow_state(self): pass
    def adjust_flow_state(self, c, p): pass
    def check_task_requirements(self, task, keys):
         missing = [k for k in keys if k not in task]
         return {"error": f"Missing keys: {missing}"} if missing else None
    def share_state(self, peer): pass # LLM state isn't typically shared this way
    def receive_state(self, state): pass
    def compute_stability(self): return 0.0
    def get_domain_specific_state(self): return None
    def process_domain_specific_state(self, data): pass


# --- Task-Specific Classes (Examples) ---

class ImageClassificationTask:
    """Represents an image classification task."""
    def __init__(self, dataset: str):
        self.dataset = dataset
        print(f"ImageClassificationTask created for dataset: {self.dataset}")

    def run(self):
        """Simulates running the image classification task."""
        print(f"--- Running Image Classification on {self.dataset} ---")
        # Simulate loading data, model prediction, evaluation
        time.sleep(0.2)
        accuracy = random.uniform(0.85, 0.98)
        print(f"--- Task Complete. Accuracy: {accuracy:.4f} ---")
        return {"accuracy": accuracy}

class TextSummarizationTask:
    """Represents a text summarization task."""
    def __init__(self, dataset: str):
        self.dataset = dataset
        print(f"TextSummarizationTask created for dataset: {self.dataset}")

    def run(self):
        """Simulates running the text summarization task."""
        print(f"--- Running Text Summarization on {self.dataset} ---")
        # Simulate loading text, model generation, evaluation (e.g., ROUGE score)
        time.sleep(0.3)
        rouge_score = random.uniform(0.4, 0.6)
        print(f"--- Task Complete. ROUGE Score: {rouge_score:.4f} ---")
        return {"rouge_score": rouge_score}




# --- CuriosityQRLAgent Modification ---
class CuriosityQRLAgent(DFSNAgent):
    """QRL agent using classical policy network and Qiskit backend for env simulation."""
    def __init__(self, name: str, bridge: ZSGQuantumBridge, action_size: int, state_size: int = 4): # Default state_size = 4
        super().__init__(name)
        if not bridge or not bridge.simulator: raise ValueError("CuriosityQRLAgent requires Qiskit Bridge.")
        self.bridge = bridge
        self.policy_network = QuantumPolicyNetwork(state_size, action_size)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.005)
        self.action_size = action_size; self.state_size = state_size
        self.gamma = 0.9; self.epsilon = 0.1
        self.state_predictor = nn.Linear(state_size, state_size)
        self.predictor_optimizer = optim.Adam(self.state_predictor.parameters(), lr=0.005)
        self.memory = []; print(f"CuriosityQRLAgent {self.name} initialized (Qiskit). Policy State Size: {state_size}, Actions: {action_size}")

    def _get_state_embedding(self, state: Any) -> torch.Tensor:
         # Adapt this based on actual state representation!
         state_size = self.state_size
         if isinstance(state, (list, np.ndarray)):
             state_np = np.array(state).flatten()
             if len(state_np) >= state_size: return torch.tensor(state_np[:state_size], dtype=torch.float32)
             else: padded = np.pad(state_np, (0, state_size - len(state_np))); return torch.tensor(padded, dtype=torch.float32)
         else: # Fallback hash
             state_str = str(state); state_hash = hashlib.sha256(state_str.encode()).digest()
             embedding = [b / 255.0 for b in state_hash[:state_size]]
             if len(embedding) < state_size: embedding.extend([0.0] * (state_size - len(embedding)))
             return torch.tensor(embedding, dtype=torch.float32)
    def select_action(self, state_embedding: torch.Tensor) -> Tuple[int, torch.Tensor]:
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            with torch.no_grad(): action_probs = self.policy_network(state_embedding)
        else:
            action_probs = self.policy_network(state_embedding)
            m = torch.distributions.Categorical(probs=action_probs)
            action = m.sample().item()
        return action, action_probs
    def compute_curiosity_reward(self, state: Any, action: int, next_state: Any) -> float:
        state_emb = self._get_state_embedding(state); next_state_emb = self._get_state_embedding(next_state)
        predicted_next_emb = self.state_predictor(state_emb); loss = nn.MSELoss()(predicted_next_emb, next_state_emb)
        self.predictor_optimizer.zero_grad(); loss.backward(); self.predictor_optimizer.step()
        curiosity_reward = loss.item(); curiosity_reward = self.math_module.f0z_stabilize(torch.tensor(curiosity_reward)).item()
        self.memory.append((state_emb.detach().numpy(), action, next_state_emb.detach().numpy()));
        if len(self.memory) > 500: self.memory.pop(0)
        return curiosity_reward * 0.1 # Scaled reward
    def standard_reward(self, fidelity, noise_level): return fidelity - 0.5 * noise_level
    def ffz_reward(self, fidelity, noise_level, action_entropy):
        fidelity_t = torch.tensor(fidelity, dtype=torch.float32); noise_t = torch.tensor(noise_level, dtype=torch.float32); action_entropy_t = torch.tensor(action_entropy, dtype=torch.float32)
        balance_factor = 1.0 - torch.abs(fidelity_t - noise_t)
        reward = fidelity_t - 0.5 * noise_t + 0.1 * action_entropy_t * balance_factor
        return reward.item()
    def update(self, state: Any, action: int, action_probs: torch.Tensor, extrinsic_reward: float, next_state: Any, fidelity: float, noise_level: float, is_ffz: bool = False):
        state_emb = self._get_state_embedding(state) # Need state emb for curiosity
        curiosity_reward = self.compute_curiosity_reward(state, action, next_state)
        action_entropy = torch.distributions.Categorical(probs=action_probs).entropy().item()
        if is_ffz: total_reward = self.ffz_reward(fidelity, noise_level, action_entropy) + curiosity_reward
        else: total_reward = self.standard_reward(fidelity, noise_level) + curiosity_reward
        m = torch.distributions.Categorical(probs=action_probs); log_prob = m.log_prob(torch.tensor(action))
        loss = -log_prob * total_reward # REINFORCE
        self.policy_optimizer.zero_grad(); loss.backward(); self.policy_optimizer.step()
        self.performance_history.append(total_reward)

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        action = task.get("action")
        if action == "qrl_step_qiskit":
             error = self.check_task_requirements(task, ["state", "qiskit_circuit_func", "qiskit_shots"]) # Changed to circuit_func
             if error: return error
             current_state = task["state"]; circuit_func = task["qiskit_circuit_func"]; shots = task["qiskit_shots"]
             use_ffz_reward = task.get("use_ffz_reward", False)
             state_emb = self._get_state_embedding(current_state)
             chosen_action, action_probs = self.select_action(state_emb)
             # Get base circuit from function and apply action
             sim_circuit = circuit_func() # Call the function to get a fresh circuit
             if not isinstance(sim_circuit, QuantumCircuit) or sim_circuit.num_qubits != self.bridge.n_qubits: return {"error": "Invalid circuit from qiskit_circuit_func"}
             qubit_to_act = 0 # Example action mapping
             if chosen_action == 0: sim_circuit.h(qubit_to_act)
             elif chosen_action == 1: sim_circuit.x(qubit_to_act)
             elif chosen_action == 2: sim_circuit.cx(qubit_to_act, (qubit_to_act + 1) % self.bridge.n_qubits)
             elif chosen_action == 3: pass # Identity
             sim_results = self.bridge.run_circuit(sim_circuit, shots=shots)
             if "error" in sim_results: return sim_results
             # Determine reward/next_state (Environment Logic)
             counts = sim_results.get("counts", {}); fidelity = (counts.get('00', 0) + counts.get('11', 0)) / shots if shots > 0 else 0.0 # Example for Bell state
             noise_level = max(0.0, 1.0 - fidelity); extrinsic_reward = fidelity
             next_state = [fidelity, 1.0 - fidelity, noise_level, 1.0 - noise_level] # Example next state
             self.update(current_state, chosen_action, action_probs, extrinsic_reward, next_state, fidelity, noise_level, is_ffz=use_ffz_reward)
             return {"result": {"action_taken": chosen_action, "fidelity": fidelity, "noise_level": noise_level, "extrinsic_reward": extrinsic_reward, "next_state": next_state, "qiskit_counts": counts}, "agent": self.name}
        return super()._execute_single_task_iteration(task)


# --- Replace or Augment F0ZAgent/Create New LLMAgent ---
# Option: Create a new LLMAgent inheriting from DFSNAgent
class LLMAgent(DFSNAgent):
    def __init__(self, name: str, model_name: str = "llama3", ollama_base_url: str = "http://localhost:11434"):
        super().__init__(name)
        self.model_name = model_name
        self.base_url = ollama_base_url
        self.api_endpoint = f"{self.base_url}/api/generate"
        print(f"LLMAgent {self.name} initialized to use Ollama model '{self.model_name}' at {self.base_url}")
        # Verify connection (optional)
        self._check_connection()

    def _check_connection(self):
        try:
            response = requests.get(self.base_url) # Check if base URL is reachable
            response.raise_for_status() # Raise an exception for bad status codes
            print(f"  Ollama connection successful to {self.base_url}")
            # Check if model is available (optional, makes init slower)
            # models_res = requests.get(f"{self.base_url}/api/tags")
            # available_models = [m['name'] for m in models_res.json().get('models', [])]
            # if self.model_name not in [m.split(':')[0] for m in available_models]: # Check base name
            #     print(f"  Warning: Model '{self.model_name}' not listed in Ollama tags. Ensure it's pulled.")
        except requests.exceptions.RequestException as e:
            print(f"  Warning: Could not connect to Ollama at {self.base_url}. Error: {e}")
            print("  Ensure the Ollama server is running.")
        except Exception as e:
             print(f" Error checking Ollama connection: {e}")


    # Implement the single iteration execution logic
    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        action = task.get("action", "generate") # Default action
        data = task.get("data", {})
        prompt = data.get("prompt", task.get("query", "Explain the ZSG Framework briefly.")) # Get prompt/query
        # Parameters for the Ollama API
        ollama_params = data.get("ollama_params", {
            "stream": False, # Get full response at once
            "options": { # Model-specific options
                "temperature": 0.7,
                "num_predict": 150 # Max tokens to generate (~max_new_tokens)
            }
        })

        if action in ["generate", "infer", "explain", "generate_text", "plan", "summarize"]: # Actions this agent handles
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                **ollama_params # Include stream and options
            }

            print(f"  {self.name}: Sending prompt to Ollama model '{self.model_name}': '{prompt[:70]}...'")

            try:
                response = requests.post(self.api_endpoint, json=payload, timeout=120) # 120s timeout
                response.raise_for_status() # Check for HTTP errors

                response_data = response.json()
                llm_output = response_data.get("response", "Error: No response field in Ollama output.")
                # print(f"  {self.name}: Received response: {llm_output[:100]}...") # Debug
                return {"result": {"llm_response": llm_output.strip()}, "agent": self.name}

            except requests.exceptions.Timeout:
                print(f"  Error: Ollama request timed out for {self.name}.")
                return {"error": "Ollama request timed out", "agent": self.name}
            except requests.exceptions.RequestException as e:
                print(f"  Error: Ollama request failed for {self.name}: {e}")
                return {"error": f"Ollama request failed: {e}", "agent": self.name}
            except Exception as e:
                 print(f"  Error processing Ollama response for {self.name}: {e}")
                 return {"error": f"Failed to process Ollama response: {e}", "agent": self.name}

        else:
            # If action isn't for LLM, delegate to base DFSNAgent logic
            return super()._execute_single_task_iteration(task)

    # Optional: Keep specific overrides if needed
    def get_engagement_state(self): return 1 # LLM agent might have a base level activity
    def share_state(self, peer): pass
    def receive_state(self, state): pass


# --- ZSGManager Modifications --

# --- Core ZSG Systems ---
class MemorySystem:
    """Manages short-term and long-term memory for ZSG agents."""
    def __init__(self):
        # Simple dict-based short-term memory (e.g., last result per task type)
        self.short_term_memory: Dict[str, Any] = {}
        # List-based long-term memory (e.g., history of significant events or results)
        self.long_term_memory: List[Any] = []
        # Task-specific storage (from later iterations)
        self.task_store: Dict[str, List[Dict]] = {} # Key: episode_iteration string
        self.episode_memory: Dict[Tuple[int, int], Any] = {} # Key: (episode, iteration) tuple
        self.max_long_term_size = 1000 # Configurable limit
        print("MemorySystem initialized.")


    def store_memory(self, memory_data: Any, memory_type: str = 'short', key: Optional[str] = None):
        """Store data in the specified memory type."""
        if memory_type == 'short':
            if key is None: key = f"generic_{time.time()}" # Generate a key if none provided
            self.short_term_memory[key] = memory_data
            # print(f"Stored in short-term memory with key '{key}'.")
        elif memory_type == 'long':
            self.long_term_memory.append(memory_data)
            # Enforce size limit
            if len(self.long_term_memory) > self.max_long_term_size:
                self.long_term_memory.pop(0) # Remove the oldest item
            # print(f"Stored in long-term memory (size: {len(self.long_term_memory)}).")
        else:
            print(f"Warning: Unknown memory type '{memory_type}'. Data not stored.")

    def retrieve_memory(self, memory_type: str = 'short', key: Optional[str] = None, criteria: Optional[callable] = None) -> Any:
        """
        Retrieve data from the specified memory type.
        For long-term, criteria can be a function that takes a memory item and returns True if it matches.
        """
        if memory_type == 'short':
            if key:
                return self.short_term_memory.get(key)
            else:
                # Return the most recent item or None if empty
                return list(self.short_term_memory.values())[-1] if self.short_term_memory else None
        elif memory_type == 'long':
            if criteria:
                # Search long-term memory backwards (most recent first) based on criteria
                for item in reversed(self.long_term_memory):
                    try:
                        if criteria(item):
                            return item
                    except Exception as e:
                         # Catch errors in criteria function evaluation
                        print(f"Error applying criteria function in retrieve_memory: {e}")
                        continue # Skip this item
                return None # No item matched the criteria
            else:
                # Return the most recent long-term memory item or None if empty
                return self.long_term_memory[-1] if self.long_term_memory else None
        else:
            print(f"Warning: Unknown memory type '{memory_type}'. Cannot retrieve.")
            return None

    def store_task(self, episode: int, iteration: int, todo): # Assuming todo has a .to_json() method
        """Stores a specific task identified by episode and iteration."""
        key = f"{episode}_{iteration}"
        if key not in self.task_store:
            self.task_store[key] = []
        # Ensure todo is serializable or get its JSON representation
        try:
            todo_data = todo.to_json() if hasattr(todo, 'to_json') else todo
            if not isinstance(todo_data, dict):
                 raise ValueError("Stored task data must be a dictionary or have a to_json method.")
            self.task_store[key].append(todo_data)
            # print(f"Stored task for {key}. Total tasks for key: {len(self.task_store[key])}")
        except Exception as e:
             print(f"Error storing task for {key}: {e}. Data: {todo}")


    def store_episode(self, episode: int, iteration: int, results: Any):
        """Stores the results associated with a specific episode and iteration."""
        self.episode_memory[(episode, iteration)] = results
        # print(f"Stored results for episode {episode}, iteration {iteration}.")

    def retrieve_task(self, episode: int, iteration: int) -> Optional[List[Dict]]:
         """Retrieves tasks for a given episode and iteration."""
         key = f"{episode}_{iteration}"
         return self.task_store.get(key)

    def retrieve_episode(self, episode: int, iteration: int) -> Optional[Any]:
         """Retrieves results for a given episode and iteration."""
         return self.episode_memory.get((episode, iteration))




# --- Batch Processing & Science Fair Mode ---

class DynamicBatchModeProcessor:
    """Handles dynamic batching of tasks for ZSG."""
    def __init__(self, manager: 'ZSGManager'): # Forward reference ZSGManager
        self.manager = manager
        self.batch: List[ZSGTodo] = []
        print("DynamicBatchModeProcessor initialized.")

    def add_to_batch(self, prompt: str, complexity: float, task_type="generic_batch_task"):
        """Adds a task described by a prompt to the current batch."""
        # Use hash for a simple task ID
        task_id = f"batch_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
        # Create ZSGTodo object
        todo = ZSGTodo(
            task_id=task_id,
            description=prompt,
            status="Pending",
            priority=complexity, # Use complexity as priority
            mliw_step="BatchProcess",
            data_payload={"prompt": prompt, "task_type": task_type} # Include original prompt and type
        )
        self.batch.append(todo)
        print(f"Added task {task_id} to batch. Batch size: {len(self.batch)}")

    def process_batch(self) -> List[Dict]:
        """Processes all tasks currently in the batch using the ZSGManager."""
        print(f"Processing batch of {len(self.batch)} tasks...")
        results = []
        if not self.batch:
             return results

        # Get current MLIW state from manager
        episode, iteration = self.manager.mliw.get_state()

        # Use ThreadPoolExecutor for parallel processing (optional)
        # Adjust max_workers based on system resources
        # max_workers = min(4, os.cpu_count())
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #      futures = [executor.submit(self._process_single_todo, todo, episode, iteration) for todo in self.batch]
        #      results = [future.result() for future in futures]

        # Sequential processing:
        for todo in self.batch:
             result_package = self._process_single_todo(todo, episode, iteration)
             results.append(result_package)


        # Clear the batch after processing
        processed_count = len(self.batch)
        self.batch = []
        print(f"Batch processing complete. Processed {processed_count} tasks.")
        return results # List of results from each task


    def _process_single_todo(self, todo: ZSGTodo, episode: int, iteration: int) -> Dict:
         """Helper to process a single TODO item within the batch."""
         print(f"  Processing batch task: {todo.task_id} ({todo.description[:30]}...)")
         todo.update_status("InProgress", agent_name="BatchProcessor")

         # Construct the task dictionary for the manager
         task_dict = {
             "type": todo.data_payload.get("task_type", "generic_batch_task"),
             "action": todo.mliw_step, # Or derive action from description/type
             "complexity": todo.priority,
             "data": todo.data_payload, # Pass entire payload
             "task_id": todo.task_id # Include ID for tracking
         }

         # Process task using the manager's main processing function
         result = self.manager.process_task_with_zsg(task_dict) # Pass the constructed task

         # Update TODO status based on result
         if "error" in result:
             todo.update_status("Failed")
             print(f"    Task {todo.task_id} failed: {result['error']}")
         else:
             todo.update_status("Completed")
             # Store the result back into the todo's payload (optional)
             todo.data_payload["result"] = result.get("result")

         # Store the completed/failed TODO using the MemorySystem
         self.manager.memory_system.store_task(episode, iteration, todo)

         # Return a package containing original task info and the result
         return {"task": todo.to_json(), "result_package": result}



class ZSGBatchModeScienceFair:
    """Orchestrates a 'Science Fair' simulation using batch processing and forecasting."""
    def __init__(self, manager: 'ZSGManager'):
        self.manager = manager
        self.processor = DynamicBatchModeProcessor(manager)
        # Find required agents within the manager's list
        self.temporal_agent = self._find_agent(TemporalPrimeAgent)
        self.collab_agent = self._find_agent(CollaborativeAgent)
        if not self.temporal_agent or not self.collab_agent:
             raise RuntimeError("Required agents (TemporalPrimeAgent, CollaborativeAgent) not found in manager.")
        print("ZSGBatchModeScienceFair initialized.")

    def _find_agent(self, agent_class: type) -> Optional[DFSNAgent]:
         """Utility to find the first agent of a specific class."""
         for agent in self.manager.agents:
             if isinstance(agent, agent_class):
                 return agent
         return None

    def run_science_fair(self, experiments: List[str], forecast_horizon: int = 5) -> Dict:
        """Runs a series of experiments, processes them in batch, forecasts, and combines."""
        print(f"\n--- Running ZSG Science Fair with {len(experiments)} experiments ---")

        # 1. Add experiments to batch processor
        print("Adding experiments to batch...")
        for i, exp_description in enumerate(experiments):
             # Assign complexity based on experiment name length or index (example)
             complexity = max(3.0, min(10.0, len(exp_description) / 5.0))
             self.processor.add_to_batch(f"Experiment {i}: {exp_description}", complexity, task_type="science_fair_experiment")

        # 2. Process the batch of experiments
        batch_results_packages = self.processor.process_batch()
        # Extract just the results relevant for forecasting
        batch_results_for_forecast = [pkg['result_package'].get('result', {}).get('experiment_outcome', 0) # Extract specific numeric outcome
                                      for pkg in batch_results_packages if 'error' not in pkg['result_package']]

        # 3. Forecast based on batch results using TemporalPrimeAgent
        print(f"\nForecasting future trends (Horizon: {forecast_horizon})...")
        forecast_input_data = [res if isinstance(res, (int, float)) else (res.get('value', 0) if isinstance(res, dict) else 0) for res in batch_results_for_forecast] # Extract numeric values
        forecast_task = {
            "type": "temporal_forecast",
            "action": "forecast", # Explicit action
            "data": forecast_input_data,
            "horizon": forecast_horizon,
            "complexity": 6.0 # Assign complexity to the forecast task itself
        }
        forecast_result_package = self.temporal_agent._execute_single_task_iteration(forecast_task)
        forecast_value = forecast_result_package.get("result", {}).get("forecast", "Forecast Error")
        print(f"Forecast Result: {forecast_value}")

        # 4. Combine results and forecast using CollaborativeAgent
        print("\nCombining batch results and forecast...")
        collab_task = {
             "type": "collaboration",
             "action": "combine", # Use the 'combine' action defined in CollaborativeAgent
             "data": {"batch_results": batch_results_for_forecast, "forecast": forecast_value},
             "complexity": 4.0
        }
        combined_result_package = self.collab_agent._execute_single_task_iteration(collab_task)
        final_combined_data = combined_result_package.get("result", {"summary": "Combination Error"})
        print("Combination complete.")

        # 5. Return comprehensive results
        print("--- ZSG Science Fair Run Complete ---")
        return {
            "batch_run_details": batch_results_packages, # Full details of each batch task
            "forecast_package": forecast_result_package,
            "combination_package": combined_result_package,
            "final_summary": final_combined_data # The data combined by the collab agent
        }

class ZSGTodo:
    """Represents a task item within the ZSG framework."""
    def __init__(self, task_id: str, description: str, status: str, priority: float, mliw_step: str, data_payload: Dict):
        self.task_id = task_id # Unique ID (e.g., "T001", "QuantumTask_abc")
        self.description = description # E.g., "Optimize PNS sampling", "Run Grover Search"
        self.status = status # "Pending", "In Progress", "Completed", "Failed"
        self.priority = priority # Numerical priority (higher is more important)
        self.mliw_step = mliw_step # Which MLIW phase (e.g., "Analyze", "Modulate", "Test", "Generate", "Validate")
        self.data_payload = data_payload # Input data, parameters, or results needed/produced (dict)

        # Additional fields for tracking, if needed
        self.creation_time = time.time()
        self.assigned_agent: Optional[str] = None
        self.completion_time: Optional[float] = None

    def to_json(self) -> Dict:
        """Serializes the ZSGTodo object into a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "mliw_step": self.mliw_step,
            "data": self.data_payload, # Keep payload under 'data' key for consistency
            "creation_time": self.creation_time,
            "assigned_agent": self.assigned_agent,
            "completion_time": self.completion_time
        }

    def update_status(self, new_status: str, agent_name: Optional[str] = None):
        """Updates the status of the TODO item."""
        self.status = new_status
        if agent_name:
            self.assigned_agent = agent_name
        if new_status in ["Completed", "Failed"]:
            self.completion_time = time.time()
        print(f"TODO {self.task_id} status updated to {new_status}" + (f" by {agent_name}" if agent_name else ""))



class ResourceMonitor:
    """Monitors system resources and can adjust parameters like batch size."""
    def __init__(self, batch_size_init=32):
        self.batch_size = batch_size_init
        self.base_cpu = 10 # Base allocation % per agent
        self.base_memory = 10 # Base allocation % per agent
        self.total_cpu = psutil.cpu_count() * 100 # Theoretical max %
        self.total_memory = 100 # Percentage based
        self.processes = {} # Track resource usage per process/agent if needed
        print(f"ResourceMonitor initialized. Initial batch size: {self.batch_size}")


    def check_usage(self) -> Tuple[float, float]:
        """Checks current overall CPU and virtual memory usage."""
        cpu_usage = psutil.cpu_percent(interval=0.1) # Non-blocking short interval
        mem_usage = psutil.virtual_memory().percent
        # print(f"Resource Check: CPU={cpu_usage}%, Memory={mem_usage}%")
        return cpu_usage, mem_usage

    def adjust_batch_size(self) -> int:
        """Adjusts batch size based on resource usage."""
        cpu, mem = self.check_usage()
        if cpu > 85.0 or mem > 85.0:
            print(f"High resource usage (CPU: {cpu:.1f}%, Mem: {mem:.1f}%). Reducing batch size.")
            self.batch_size = max(8, self.batch_size // 2) # Halve batch size, min 8
        elif cpu < 50.0 and mem < 50.0:
             # Gradually increase batch size if resources are low
             self.batch_size = min(128, self.batch_size + 4) # Increase slowly, max 128
        print(f"Adjusted batch size to: {self.batch_size}")
        return self.batch_size

    def pre_allocate(self, agents: List[DFSNAgent], task_complexity: float) -> Dict[str, float]:
        """Estimate and pre-allocate resources based on agent count and task complexity."""
        num_agents = len(agents)
        if num_agents == 0: return {"cpu_total": 0, "memory_total": 0}

        # Estimate needed resources - simple scaling
        cpu_needed_per_agent = min(self.base_cpu + 5 * task_complexity, 100 / num_agents) # Limit by available share
        mem_needed_per_agent = min(self.base_memory + 4 * task_complexity, 100 / num_agents)

        total_cpu_allocated = 0
        total_memory_allocated = 0
        for agent in agents:
            agent.cpu_allocation = cpu_needed_per_agent
            agent.memory_allocation = mem_needed_per_agent
            total_cpu_allocated += cpu_needed_per_agent
            total_memory_allocated += mem_needed_per_agent

        print(f"Pre-allocated resources: ~{cpu_needed_per_agent:.1f}% CPU, ~{mem_needed_per_agent:.1f}% Mem per agent. Total: {total_cpu_allocated:.1f}% CPU, {total_memory_allocated:.1f}% Mem")
        return {"cpu_total": total_cpu_allocated, "memory_total": total_memory_allocated}


    def update_allocations(self, agents: List[DFSNAgent]):
        """Dynamically update resource allocations based on agent engagement and performance."""
        num_agents = len(agents)
        if num_agents == 0: return

        active_agents = [agent for agent in agents if agent.engagement_state > 0]
        num_active = len(active_agents)

        if num_active == 0: # No active agents, allocate base resources
            for agent in agents:
                agent.cpu_allocation = self.base_cpu
                agent.memory_allocation = self.base_memory
            # print("No active agents. Allocating base resources.")
            return

        # Calculate available resources beyond base allocation for all agents
        available_extra_cpu = self.total_cpu - (self.base_cpu * num_agents)
        available_extra_memory = self.total_memory - (self.base_memory * num_agents)

        # Distribute extra resources among active agents based on performance (simple weighting)
        total_performance_score = sum(agent.performance_history[-1] if agent.performance_history else 0.1 for agent in active_agents) + 1e-8 # Avoid zero division
        if total_performance_score <= 0: total_performance_score = 1e-8 # Ensure positive

        for agent in agents:
            if agent in active_agents:
                perf_score = agent.performance_history[-1] if agent.performance_history else 0.1
                perf_weight = (perf_score / total_performance_score) if total_performance_score > 0 else (1/num_active)

                # Allocate proportional share of extra resources + base
                agent.cpu_allocation = self.base_cpu + available_extra_cpu * perf_weight
                agent.memory_allocation = self.base_memory + available_extra_memory * perf_weight
            else:
                # Inactive agents get base allocation
                agent.cpu_allocation = self.base_cpu
                agent.memory_allocation = self.base_memory

        # Clamp allocations to reasonable bounds (e.g., max 80% per agent)
        for agent in agents:
             agent.cpu_allocation = max(5.0, min(agent.cpu_allocation, 80.0))
             agent.memory_allocation = max(5.0, min(agent.memory_allocation, 80.0))
             # print(f"  {agent.name}: CPU={agent.cpu_allocation:.1f}%, Mem={agent.memory_allocation:.1f}%")

        # print("Updated resource allocations for active agents based on performance.")

    def start(self):
         """Start monitoring (if running in a separate thread/process)."""
         print("Resource monitoring started (simulated).")

    def stop(self):
         """Stop monitoring."""
         print("Resource monitoring stopped (simulated).")

class DynamicFlowStateNetwork:
    """Manages the flow states of a collection of agents based on task complexity."""
    def __init__(self, agents: List[DFSNAgent], task_complexity_threshold: float = 5.0, max_agents: int = 15):
        self.agents = agents # This should be a reference, updated by ZSGManager
        self.task_complexity_threshold = task_complexity_threshold
        self.agent_states: Dict[str, str] = {agent.name: agent.flow_state for agent in agents}
        self.max_agents = max_agents
        self.is_dynamic_enabled = False
        print(f"DFSN initialized. Threshold: {self.task_complexity_threshold}, Max Agents: {self.max_agents}")


    def adjust_flow_states(self, current_task_complexity: float, batch_info: Optional[List] = None):
        """Adjust agent flow states based on complexity and potentially batch info."""
        if not self.is_dynamic_enabled:
            # print("DFSN is disabled. No flow state adjustments.")
            return

        # Determine complexity measure (use batch avg if available)
        complexity_measure = current_task_complexity
        if batch_info and len(batch_info) > 0:
            # Assuming batch_info is a list of ZSGTodo objects or similar dicts with 'priority'
            try:
                batch_avg_complexity = sum(item.priority if hasattr(item, 'priority') else item.get('priority', 5.0) for item in batch_info) / len(batch_info)
                complexity_measure = (current_task_complexity + batch_avg_complexity) / 2 # Average task and batch complexity
                print(f"DFSN using combined complexity: {complexity_measure:.2f} (Task: {current_task_complexity:.2f}, Batch Avg: {batch_avg_complexity:.2f})")
            except Exception as e:
                print(f"DFSN Warning: Could not calculate batch average complexity: {e}")


        # print(f"DFSN adjusting flow states based on complexity measure: {complexity_measure:.2f} (Threshold: {self.task_complexity_threshold})")
        num_flow = 0
        num_idle = 0
        for agent in self.agents:
            # Agents decide their own state via adjust_flow_state called within AIW/_execute_single_task_iteration
            # DFSN can provide global context or override based on system-wide needs
            performance = np.mean(agent.performance_history[-5:]) if agent.performance_history else 0.0
            # The agent's internal optimizer will handle the transition logic
            agent.adjust_flow_state(complexity_measure, performance)
            self.agent_states[agent.name] = agent.flow_state # Update tracked state

            if agent.flow_state == 'flow':
                num_flow += 1
            else:
                num_idle += 1

        print(f"DFSN status: {num_flow} agents in flow, {num_idle} agents idle.")
        self.scale_agents(complexity_measure) # Scale agents after adjustments

    def enable_dynamic_states(self):
        """Enable dynamic adjustments."""
        self.is_dynamic_enabled = True
        print("DFSN enabled for dynamic state adjustments.")

    def disable_dynamic_states(self):
        """Disable dynamic adjustments and reset agents to idle."""
        self.is_dynamic_enabled = False
        for agent in self.agents:
             if agent.flow_state == 'flow':
                 agent.exit_flow_state() # Explicitly exit flow
             self.agent_states[agent.name] = agent.flow_state
        print("DFSN disabled. Agents reset towards idle states.")

    def scale_agents(self, task_complexity: float):
        """Dynamically scale the number of active agents (placeholder)."""
        # Example scaling logic: more complex tasks require more agents active/instantiated
        target_active_agents = max(2, min(self.max_agents, int(task_complexity / 2.0) + 1))
        num_current_agents = len(self.agents)
        num_currently_active = sum(1 for agent in self.agents if agent.engagement_state > 0)

        print(f"DFSN Scaling Check: Complexity={task_complexity:.2f}, TargetActive={target_active_agents}, CurrentTotal={num_current_agents}, CurrentActive={num_currently_active}")

        # Add new agents if needed and below max limit
        if num_current_agents < target_active_agents and num_current_agents < self.max_agents:
            num_to_add = min(target_active_agents - num_current_agents, self.max_agents - num_current_agents)
            print(f"  Scaling up: Adding {num_to_add} new DFSNAgent(s).")
            for i in range(num_to_add):
                 new_agent_name = f"DFSNAgent_{num_current_agents + i}"
                 # This needs interaction with the ZSGManager to actually add the agent
                 # self.manager.add_agent(DFSNAgent, new_agent_name) # Conceptual
                 print(f"    (Conceptual) Added {new_agent_name}")
            # Note: Need a reference to the manager or a callback to add agents properly.

        # Deactivate surplus agents if complexity is low (or handle via engagement states)
        elif num_currently_active > target_active_agents and task_complexity < self.task_complexity_threshold * 0.8:
             num_to_deactivate = num_currently_active - target_active_agents
             print(f"  Scaling down: Deactivating {num_to_deactivate} agent(s) (setting to low engagement).")
             # Find agents to deactivate (e.g., lowest performance or idle)
             agents_to_consider = sorted(self.agents, key=lambda a: (a.engagement_state, np.mean(a.performance_history[-5:]) if a.performance_history else 0))
             for i in range(num_to_deactivate):
                 if i < len(agents_to_consider):
                     agents_to_consider[i].exit_flow_state() # Force lower engagement
                     print(f"    Deactivated {agents_to_consider[i].name}")

    def handle_chaos(self, chaos_metrics: Dict):
         """Adjust DFSN parameters based on chaos metrics."""
         # --- CORRECTED KEY ---
         lyapunov_exp = chaos_metrics.get("lyapunov_estimate", 0.0)
         # --- END CORRECTION ---
         print(f"DFSN received chaos metrics: Lyapunov={lyapunov_exp:.3f}")
         # Rest of the logic using lyapunov_exp remains the same...
         if lyapunov_exp > 0.5: # System becoming more chaotic
             print("  High chaos detected. Increasing stability preference (reducing threshold).")
             self.task_complexity_threshold *= 0.95
             for agent in self.agents:
                 # Check if optimizer exists before modifying lr
                 if hasattr(agent, 'optimizer') and hasattr(agent.optimizer, 'learning_rate'):
                     agent.optimizer.learning_rate = min(0.2, agent.optimizer.learning_rate * 1.1)
         elif lyapunov_exp < 0.1: # System very stable
             print("  Low chaos detected. Increasing complexity tolerance (increasing threshold).")
             self.task_complexity_threshold *= 1.05
             for agent in self.agents:
                  if hasattr(agent, 'optimizer') and hasattr(agent.optimizer, 'learning_rate'):
                     agent.optimizer.learning_rate = max(0.05, agent.optimizer.learning_rate * 0.95)


class MultiAgentCoordinator:
    """Coordinates task assignment and synchronization among multiple agents."""
    def __init__(self, agents: List[Agent]):
        self.agents = agents # Reference to the list of agents managed by ZSGManager
        self.task_queue: List[Tuple[int, Dict]] = [] # Priority queue (using heapq, neg priority)
        # Basic resource pool tracking (can be enhanced by ResourceMonitor)
        self.resource_pool = {"cpu": 100.0, "memory": 100.0}
        self.agent_states: Dict[str, Dict] = {agent.name: {"state": "idle", "load": 0.0, "engagement": agent.get_engagement_state()} for agent in agents}
        self.domain_map = self._build_domain_map()
        print(f"MultiAgentCoordinator initialized with {len(agents)} agents.")

    def _build_domain_map(self) -> Dict[str, List[str]]:
        """Builds a map from task types/domains to capable agent names."""
        # This should be dynamic based on agent capabilities
        domain_map = {
            "physics_simulation": ["PhysicsAgent"], # Can match parts of agent names
            "quantum": ["QuantumAgent"], # Matches QuantumAgent_1, QuantumAgent_QA1 etc.
            "memory_task": ["MemoryAgent"],
            "science_fair_experiment": ["CollaborativeAgent"],
            "collaboration": ["CollaborativeAgent"],
            "temporal_forecast": ["TemporalPrimeAgent"],
            "organic_chemistry": ["OrganicChemistryAgent"],
            "molecular_biology": ["MolecularBiologyAgent"],
            "creative": ["CreativeAgent"],
            "information_theory": ["InformationTheoryAgent"],
            "data_science": ["DataScienceAgent"],
            "astrophysics": ["AstrophysicsAgent"],
            "robotics": ["RoboticsAgent"],
            "environmental_science": ["EnvironmentalScienceAgent"],
            "machine_learning": ["MachineLearningAgent"],
            "validation": ["ValidationAgent"],
            "chaos": ["PhysicsAgent", "FractalAgent"], # Example: Multiple agents can handle
            "fractal_generate": ["FractalAgent"],
            "hnn_update": ["HopfieldAgent"],
            "temporal_scaling": ["TemporalPrimeAgent"],
            "llada_task": ["LLaDATaskAgent"],
            "quantum_poetry": ["QuantumAIMLLLM"], # Handled by manager directly? Or a dedicated agent?
            "quantum_game": ["QuantumAgent"],
            "quantum_field": ["QuantumAgent"],
            "grover_search": ["QuantumAgent"],
            "shor_factor": ["QuantumAgent"],
            "quantum_circuit": ["QuantumAgent"],
            # Add more mappings as new agents/tasks are defined
        }
        print("Coordinator domain map built.")
        return domain_map

    def find_capable_agents(self, task_type: str) -> List[Agent]:
         """Finds agents whose names or declared capabilities match the task type."""
         capable_agents = []
         agent_dict = {agent.name: agent for agent in self.agents}

         for domain_key, agent_name_patterns in self.domain_map.items():
             if domain_key in task_type: # Simple substring matching for type
                 for pattern in agent_name_patterns:
                     for agent_name, agent in agent_dict.items():
                         if pattern in agent_name and agent not in capable_agents:
                             capable_agents.append(agent)

         # Fallback if no specific agent found
         if not capable_agents:
             print(f"No specific agent found for task type '{task_type}', assigning to general DFSNAgent.")
             # Find any generic DFSNAgent available
             general_agents = [agent for agent in self.agents if isinstance(agent, DFSNAgent) and not any(pattern in agent.name for patterns in self.domain_map.values() for pattern in patterns)]
             if general_agents:
                 capable_agents.append(random.choice(general_agents)) # Assign to a random general agent
             elif self.agents:
                  capable_agents.append(random.choice(self.agents)) # Assign to any agent if no general one exists

         # print(f"Found {len(capable_agents)} capable agents for task '{task_type}': {[a.name for a in capable_agents]}")
         return capable_agents


    def add_task(self, task: dict, priority: int):
        """Adds a task to the priority queue."""
        if 'type' not in task:
            print("Warning: Task added without 'type'. Assigning low priority.")
            task['type'] = 'unknown'
            priority = -10 # Use negative for min-heap

        print(f"Coordinator adding task: {task['type']} with priority {priority}")
        heappush(self.task_queue, (-priority, task)) # Use negative priority for max-heap behavior

    def assign_tasks(self, task: Dict) -> Dict:
         """Assigns a single task to the most suitable agent(s)."""
         task_type = task.get("type", "unknown")
         capable_agents = self.find_capable_agents(task_type)

         if not capable_agents:
             print(f"Error: No capable agent found for task type '{task_type}'.")
             return {"error": "No suitable agent found", "task_type": task_type}

         # Simple assignment: Assign to the least loaded capable agent
         # More complex: Consider engagement state, specialization score, etc.
         best_agent = min(capable_agents, key=lambda a: self.agent_states.get(a.name, {"load": 0.0})["load"])

         print(f"Assigning task '{task_type}' to agent: {best_agent.name} (Load: {self.agent_states.get(best_agent.name, {}).get('load', 0):.2f})")
         self.agent_states[best_agent.name]["load"] += task.get("complexity", 5.0) # Increment load estimate

         # Execute the task via the agent's _execute_single_task_iteration method
         result = best_agent._execute_single_task_iteration(task)

         # Decrement load after execution (simplified)
         self.agent_states[best_agent.name]["load"] -= task.get("complexity", 5.0)
         self.agent_states[best_agent.name]["load"] = max(0, self.agent_states[best_agent.name]["load"]) # Ensure load doesn't go negative

         # Synchronize states after task completion
         self.synchronize_flow_states({"success": "error" not in result})

         return result

    def execute_next_task(self) -> Optional[Dict]:
        """Retrieves and executes the highest priority task from the queue."""
        if not self.task_queue:
            # print("Coordinator task queue is empty.")
            return None

        neg_priority, task = heappop(self.task_queue)
        priority = -neg_priority
        print(f"Coordinator executing next task (Priority: {priority}): {task.get('type', 'Unknown')}")

        return self.assign_tasks(task) # Use the assign_tasks logic


    def synchronize_flow_states(self, task_feedback: Optional[Dict] = None):
        """Synchronizes agent states, potentially based on feedback or global state."""
        # print("Coordinator synchronizing flow states...")
        if not self.agents: return

        engagement_levels = [agent.get_engagement_state() for agent in self.agents]
        avg_engagement = np.mean(engagement_levels) if engagement_levels else 0
        # print(f"  Average engagement: {avg_engagement:.2f}")

        # Update internal tracking
        for agent in self.agents:
            self.agent_states[agent.name]["engagement"] = agent.get_engagement_state()
            # Optional: Adjust load based on engagement?

        # Basic synchronization: Agents adjust workload based on average (or peers)
        for agent in self.agents:
            # Pass average engagement, agent can decide how to react
             agent.adjust_workload(avg_engagement)

             # Optionally trigger DFSN adjustments based on global state
             # if hasattr(self.manager, 'flow_state_network'): # Requires link to manager
             #     self.manager.flow_state_network.adjust_flow_states(global_complexity_measure)
             pass

        # print("Flow state synchronization complete.")

class EntropyBotTurbo:
    """Monitors entropy of quantum states."""
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon # Small value to avoid log(0)
        self.math_module = PyZeroMathTorch()

    def compute_entropy(self, probs: np.ndarray) -> float:
        """Computes Shannon entropy for a probability distribution."""
        # Ensure probs are valid probabilities (non-negative, sum to 1)
        probs = np.maximum(probs, 0) # Ensure non-negative
        prob_sum = np.sum(probs)
        if prob_sum <= 0: return 0.0 # Entropy is 0 if all probs are 0
        probs = probs / prob_sum # Normalize

        # Use F0Z stabilized log
        probs_t = torch.tensor(probs, dtype=torch.float32)
        log_probs = torch.log2(self.math_module.f0z_stabilize(probs_t) + self.epsilon)
        entropy = -torch.sum(probs_t * log_probs)
        return entropy.item()

    def monitor_state(self, state_vector: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculates probabilities and entropy from a quantum state vector."""
        if not isinstance(state_vector, np.ndarray) or state_vector.ndim != 1:
             print("Warning (EntropyBot): Invalid state vector input.")
             return 0.0, np.array([1.0]) # Return default entropy and probability

        probabilities = np.abs(state_vector)**2
        entropy = self.compute_entropy(probabilities)
        # print(f"EntropyBot Turbo: State entropy = {entropy:.4f}")
        return entropy, probabilities


# --- LLaDA Agent (from Iteration 23) ---
# Needs LLaDA model definition - using placeholder
class LLaDA(nn.Module): # Placeholder
     def __init__(self, vocab_size, seq_len):
         super().__init__()
         self.vocab_size = vocab_size
         self.seq_len = seq_len
         # Dummy layers
         self.embedding = nn.Linear(vocab_size, 64)
         self.output = nn.Linear(64, vocab_size)
         print("LLaDA Model Placeholder Initialized.")

     def forward_process(self, x0, t): # Simulate diffusion process
         noise = torch.randn_like(x0.float()) * t
         xt = x0.float() + noise # Simplified noise addition
         mask = (torch.rand_like(x0.float()) < t * 0.5) # Random masking based on time t
         return xt, mask

     def loss(self, x0, xt, mask, t): # Simulate loss calculation
         # Dummy loss based on difference and mask
         diff = torch.abs(x0.float() - xt) * mask.float() # Calculate difference where masked
         return torch.mean(diff) * 10 # Scale loss for visibility


class LLaDATaskAgent(DFSNAgent): # Inherit from DFSNAgent
    """Agent designed to execute tasks related to the LLaDA model."""
    def __init__(self, name: str, bridge: Optional[ZSGQuantumBridge] = None, llada_model: Optional[LLaDA] = None):
        super().__init__(name)
        self.model = llada_model if llada_model else LLaDA(100, 10) # Default placeholder model
        self.bridge = bridge # Store bridge if provided
        # Agent state to store current task info and results
        self.state = {"current_todo": None, "task_completed": False, "quantum_state_vector": None}
        print(f"LLaDATaskAgent {self.name} initialized.")

    def _execute_single_task_iteration(self, task: Dict) -> Dict:
        """Executes LLaDA-specific tasks (diffusion process simulation)."""
        task_type = task.get("type")
        action = task.get("action")

        if task_type == "llada_diffusion":
             error = self.check_task_requirements(task, ["sequence", "t"])
             if error: return error
             # Store task in agent's state
             self.state["current_todo"] = task
             self.state["task_completed"] = False

             try:
                 # Prepare input tensor (ensure it's long type for embedding usually)
                 # Using float here based on LLaDA placeholder model's forward_process
                 seq_tensor = torch.tensor(task["sequence"], dtype=torch.float32)
                 time_t = task["t"]

                 # Simulate the forward (diffusion) process and loss calculation
                 xt, mask = self.model.forward_process(seq_tensor, time_t)
                 loss = self.model.loss(seq_tensor, xt, mask, time_t)

                 # Determine completion status (example: low loss)
                 completed = loss.item() < 5.0
                 self.state["task_completed"] = completed

                 # Generate or update quantum state representation if bridge exists
                 if self.bridge:
                      encoded_state = self.bridge.encode({"task_id": task.get("task_id"), "loss": loss.item()}, entangle=completed)
                      self.state["quantum_state_vector"] = encoded_state

                 print(f"  LLaDA Task: Sequence length={len(task['sequence'])}, t={time_t:.2f} -> Loss={loss.item():.4f}, Completed={completed}")
                 return {"result": {"loss": loss.item()}, "status": "Completed" if completed else "InProgress", "agent": self.name}

             except Exception as e:
                  print(f"Error during LLaDA task execution: {e}")
                  self.state["task_completed"] = False
                  return {"error": f"LLaDA execution failed: {e}", "status": "Failed", "agent": self.name}

        return super()._execute_single_task_iteration(task) # Delegate other tasks


    # Add convergence methods from LLaDA example
    def converge(self, peer: Agent, bridge: Optional[ZSGQuantumBridge] = None):
        """Simulates convergence towards a peer's state, potentially via entanglement."""
        if not bridge or not bridge.simulator:
            print(f"{self.name}: Cannot converge without quantum bridge.")
            return

        print(f"{self.name} attempting quantum convergence with {peer.name}...")
        if isinstance(peer, LLaDATaskAgent) and peer.bridge and peer.state.get("quantum_state_vector"):
             # Get own and peer's quantum state
             my_state_vec = self.state.get("quantum_state_vector")
             peer_state_vec = peer.state.get("quantum_state_vector")

             if my_state_vec and peer_state_vec and len(my_state_vec) == len(peer_state_vec):
                  # Simulate entanglement and projection (simplified: average states)
                  my_q_state = np.array(my_state_vec, dtype=complex)
                  peer_q_state = np.array(peer_state_vec, dtype=complex)
                  converged_state = 0.5 * (my_q_state + peer_q_state)
                  norm = LA.norm(converged_state)
                  if norm > 1e-9: converged_state /= norm

                  # Update own quantum state
                  self.state["quantum_state_vector"] = converged_state.tolist()
                  # Optionally update bridge simulator state if this agent 'owns' it
                  # bridge.simulator.state = converged_state
                  print(f"  {self.name} converged quantum state with {peer.name}.")
             else:
                  print(f"  Convergence failed: Invalid or incompatible quantum states.")
        else:
             print(f"  Convergence failed: Peer {peer.name} is not suitable or has no quantum state.")


    def deconverge(self, peer: Agent):
         """Simulates decoherence or moving away from a peer's state."""
         print(f"{self.name} deconverging from {peer.name} (simulated by randomizing state slightly).")
         if self.state.get("quantum_state_vector"):
              # Add small random noise to quantum state (simulate decoherence)
              q_state = np.array(self.state["quantum_state_vector"], dtype=complex)
              noise = (np.random.rand(*q_state.shape) + 1j * np.random.rand(*q_state.shape)) * 0.01 # Small complex noise
              noisy_state = q_state + noise
              norm = LA.norm(noisy_state)
              if norm > 1e-9: noisy_state /= norm
              self.state["quantum_state_vector"] = noisy_state.tolist()


class ZSGQueue:
     """Interface for interacting with a Redis queue for ZSG tasks."""
     def __init__(self, manager: Optional['ZSGManager'] = None, redis_host='localhost', redis_port=6379, redis_db=0):
         self.manager = manager # Optional link to manager for getting TODOs
         self.queue_name = "zsg_tasks"
         try:
             self.r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
             self.r.ping() # Check connection
             print(f"ZSGQueue connected to Redis at {redis_host}:{redis_port}")
         except Exception as e:
             print(f"Error connecting to Redis for ZSGQueue: {e}. Queue functionality disabled.")
             self.r = None

     def publish_todo(self, todo: ZSGTodo):
         """Publishes a single ZSGTodo item to the Redis queue."""
         if not self.r: return False
         try:
             self.r.lpush(self.queue_name, json.dumps(todo.to_json()))
             print(f"Published TODO {todo.task_id} to Redis queue '{self.queue_name}'.")
             return True
         except Exception as e:
             print(f"Error publishing TODO {todo.task_id} to Redis: {e}")
             return False

     def publish_todos_from_manager(self):
         """Publishes all TODOs currently held by a manager's component (e.g., processor batch)."""
         if not self.manager or not hasattr(self.manager, 'processor') or not hasattr(self.manager.processor, 'batch'):
              print("Error: Cannot publish TODOs, manager or batch processor not available.")
              return
         if not self.r: return

         todos_to_publish = self.manager.processor.batch # Publish tasks currently in the batch
         print(f"Publishing {len(todos_to_publish)} TODOs from manager batch to Redis...")
         count = 0
         for todo in todos_to_publish:
             if self.publish_todo(todo):
                 count += 1
         print(f"Successfully published {count} TODOs.")

     def subscribe_and_process(self, endpoint_name: str, processing_callback: callable):
         """Subscribes to the queue and processes tasks using the provided callback."""
         if not self.r:
             print(f"{endpoint_name}: Cannot subscribe, Redis connection failed.")
             return

         print(f"{endpoint_name}: Subscribing to Redis queue '{self.queue_name}'...")
         while True:
             try:
                 # Blocking right pop (list treated as queue) with timeout
                 task_json = self.r.brpop(self.queue_name, timeout=5)
                 if task_json:
                     # task_json is a tuple (queue_name, task_data_string)
                     task_data_str = task_json[1]
                     print(f"\n{endpoint_name} received task: {task_data_str[:100]}...")
                     try:
                         task_dict = json.loads(task_data_str)
                         # Call the processing function provided by the consumer
                         processing_callback(task_dict)
                     except json.JSONDecodeError:
                         print(f"  {endpoint_name} Error: Could not decode JSON task data.")
                     except Exception as e:
                          print(f"  {endpoint_name} Error processing task: {e}")
                 # else: print(f"{endpoint_name}: No task received (timeout).") # Optional: logging for timeout
             except redis.exceptions.ConnectionError:
                  print(f"{endpoint_name}: Redis connection error. Retrying...")
                  time.sleep(5)
             except KeyboardInterrupt:
                  print(f"{endpoint_name}: Subscription interrupted.")
                  break
             except Exception as e:
                  print(f"{endpoint_name}: Unexpected error in subscription loop: {e}")
                  time.sleep(1)




class MLIW: # Machine Learning Iterative Workflow
    """Manages the episode and iteration state for the MLIW."""
    def __init__(self):
        self.current_episode: int = 0
        self.current_iteration: int = 0
        print("MLIW Controller initialized.")

    def start_episode(self, episode: Optional[int] = None, iteration: Optional[int] = None):
        """Starts or advances the episode/iteration counter."""
        if episode is not None:
            self.current_episode = episode
        else:
            self.current_episode += 1

        if iteration is not None:
            self.current_iteration = iteration
        else:
            self.current_iteration = 1 # Reset iteration for new episode

        print(f"MLIW starting Episode {self.current_episode}, Iteration {self.current_iteration}")

    def next_iteration(self):
        """Advances to the next iteration within the current episode."""
        self.current_iteration += 1
        print(f"MLIW advanced to Episode {self.current_episode}, Iteration {self.current_iteration}")

    def get_state(self) -> Tuple[int, int]:
        """Returns the current episode and iteration."""
        return self.current_episode, self.current_iteration


class ZSGTodo:
    """Represents a task item within the ZSG framework."""
    def __init__(self, task_id: str, description: str, status: str, priority: float, mliw_step: str, data_payload: Dict):
        self.task_id = task_id # Unique ID (e.g., "T001", "QuantumTask_abc")
        self.description = description # E.g., "Optimize PNS sampling", "Run Grover Search"
        self.status = status # "Pending", "In Progress", "Completed", "Failed"
        self.priority = priority # Numerical priority (higher is more important)
        self.mliw_step = mliw_step # Which MLIW phase (e.g., "Analyze", "Modulate", "Test", "Generate", "Validate")
        self.data_payload = data_payload # Input data, parameters, or results needed/produced (dict)

        # Additional fields for tracking, if needed
        self.creation_time = time.time()
        self.assigned_agent: Optional[str] = None
        self.completion_time: Optional[float] = None

    def to_json(self) -> Dict:
        """Serializes the ZSGTodo object into a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "mliw_step": self.mliw_step,
            "data": self.data_payload, # Keep payload under 'data' key for consistency
            "creation_time": self.creation_time,
            "assigned_agent": self.assigned_agent,
            "completion_time": self.completion_time
        }

    def update_status(self, new_status: str, agent_name: Optional[str] = None):
        """Updates the status of the TODO item."""
        self.status = new_status
        if agent_name:
            self.assigned_agent = agent_name
        if new_status in ["Completed", "Failed"]:
            self.completion_time = time.time()
        print(f"TODO {self.task_id} status updated to {new_status}" + (f" by {agent_name}" if agent_name else ""))


class ChaosTheoryModule:
    """Module for simulating chaotic systems like Lorenz attractor."""
    def __init__(self):
        self.f0z_math = PyZeroMathTorch()
        print("ChaosTheoryModule initialized.")

    def _lorenz_ode(self, state, t, sigma, rho, beta):
        """Defines the Lorenz system ODEs."""
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        # Apply F0Z stabilization within the ODE calculation (optional, can affect dynamics)
        # dx_dt = self.f0z_math.f0z_stabilize(torch.tensor(dx_dt)).item()
        # dy_dt = self.f0z_math.f0z_stabilize(torch.tensor(dy_dt)).item()
        # dz_dt = self.f0z_math.f0z_stabilize(torch.tensor(dz_dt)).item()
        return [dx_dt, dy_dt, dz_dt]

    def simulate(self, system: str, params: Dict) -> Dict:
        """Simulates a chaotic system."""
        print(f"Simulating chaotic system: {system}")
        if system.lower() == "lorenz":
            # Default Lorenz parameters
            sigma = params.get("sigma", 10.0)
            rho = params.get("rho", 28.0)
            beta = params.get("beta", 8.0/3.0)
            x0 = params.get("x0", 0.01)
            y0 = params.get("y0", 0.01)
            z0 = params.get("z0", 0.01)
            t_max = params.get("t_max", 10.0)
            steps = params.get("steps", 200) # Fewer steps for faster simulation

            t_span = np.linspace(0, t_max, steps)
            initial_state = [x0, y0, z0]

            try:
                 # Integrate the ODEs
                 result = odeint(self._lorenz_ode, initial_state, t_span, args=(sigma, rho, beta))

                 # Apply F0Z stabilization to the final trajectory
                 stabilized_result = self.f0z_math.f0z_stabilize(torch.tensor(result, dtype=torch.float32), system_size=result.size).numpy()

                 # Basic chaos metric: Estimate Lyapunov exponent (very simplified)
                 # Calculate divergence of nearby trajectories (needs another simulation run)
                 pert_initial = [x0 + 1e-5, y0, z0]
                 pert_result = odeint(self._lorenz_ode, pert_initial, t_span, args=(sigma, rho, beta))
                 log_divergence = np.log(np.maximum(1e-9, np.linalg.norm(result - pert_result, axis=1)))
                 # Simple linear fit to estimate LE (not robust)
                 lyapunov_estimate = np.polyfit(t_span[1:], log_divergence[1:] / np.maximum(1e-9, t_span[1:]), 1)[0] if len(t_span) > 1 else 0.0

                 print(f"  Lorenz simulation complete. Estimated LE: {lyapunov_estimate:.3f}")
                 return {
                     "trajectory": stabilized_result.tolist(),
                     "chaos_metrics": {"lyapunov_estimate": lyapunov_estimate}
                 }
            except Exception as e:
                 print(f"Error during Lorenz simulation: {e}")
                 return {"trajectory": None, "chaos_metrics": {}, "error": str(e)}

        else:
            print(f"  Unsupported chaotic system: {system}")
            return {"trajectory": None, "chaos_metrics": {}, "error": "Unsupported system"}



    def sensitivity_analysis(self, system: str, params: Dict) -> Dict:
        """Analyzes sensitivity to initial conditions (placeholder)."""
        # This would typically involve running multiple simulations with perturbed initial conditions
        print(f"Performing sensitivity analysis for {system} (placeholder)...")
        # Simulate divergence based on params (e.g., higher rho in Lorenz -> more sensitive)
        simulated_divergence = 0.05 + params.get("rho", 28.0) * 0.001
        print(f"  Estimated divergence: {simulated_divergence:.3f}")
        return {"divergence_estimate": simulated_divergence}


class SparseLSTMModel(nn.Module):
    """LSTM Model with optional weight sparsity."""
    def __init__(self, input_size, hidden_size, sparsity=0.7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self._apply_sparsity()
        print(f"SparseLSTMModel initialized: Input={input_size}, Hidden={hidden_size}, Sparsity={sparsity}")

    def _apply_sparsity(self):
        """Applies sparsity mask to LSTM weights."""
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    mask = torch.rand_like(param) > self.sparsity
                    param.data *= mask.float()

    def forward(self, x):
        """Forward pass through the LSTM."""
        # Initialize hidden and cell states
        # Ensure batch size matches input x.size(0)
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Return the output of the last time step
        return out[:, -1, :]


class TemporalPrimeScalingModule:
    """Manages components for Temporal Prime Scaling forecasting."""
    def __init__(self, config: Dict):
        self.config = config
        # Ensure required keys are in config with defaults
        self.config.setdefault("prime_limit", 100)
        self.config.setdefault("memory_horizon", 20)
        self.config.setdefault("tau", 5.0)
        self.config.setdefault("use_deep_learning", False)
        self.config.setdefault("hidden_size", 10)
        self.config.setdefault("sparsity", 0.7)
        self.config.setdefault("hnn_size", 100) # For Hopfield agent config

        self.primes = self._generate_primes(self.config["prime_limit"])
        print("TemporalPrimeScalingModule initialized.")

    def _generate_primes(self, n): # Duplicated code
        primes = []
        is_prime = [True] * (n + 1)
        if n >= 0: is_prime[0] = False
        if n >= 1: is_prime[1] = False
        for p in range(2, int(np.sqrt(n)) + 1):
            if is_prime[p]:
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
        for p in range(2, n + 1):
            if is_prime[p]:
                primes.append(p)
        return primes if primes else [2, 3, 5]

    def initialize_agents(self) -> List[DFSNAgent]:
        """Initializes the specialized agents required for TPS."""
        print("Initializing Temporal Prime Scaling Agents...")
        agents = [
            TemporalPrimeAgent("TemporalAgent_TPS", self.config, self.primes),
            HopfieldAgent("HopfieldAgent_TPS", self.config, self.primes),
            FractalAgent("FractalAgent_TPS", self.config, self.primes)
        ]
        print("TPS Agents initialized.")
        return agents

class ZSGTodo:
    """Represents a task item within the ZSG framework."""
    def __init__(self, task_id: str, description: str, status: str, priority: float, mliw_step: str, data_payload: Dict):
        self.task_id = task_id # Unique ID (e.g., "T001", "QuantumTask_abc")
        self.description = description # E.g., "Optimize PNS sampling", "Run Grover Search"
        self.status = status # "Pending", "In Progress", "Completed", "Failed"
        self.priority = priority # Numerical priority (higher is more important)
        self.mliw_step = mliw_step # Which MLIW phase (e.g., "Analyze", "Modulate", "Test", "Generate", "Validate")
        self.data_payload = data_payload # Input data, parameters, or results needed/produced (dict)

        # Additional fields for tracking, if needed
        self.creation_time = time.time()
        self.assigned_agent: Optional[str] = None
        self.completion_time: Optional[float] = None

    def to_json(self) -> Dict:
        """Serializes the ZSGTodo object into a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "mliw_step": self.mliw_step,
            "data": self.data_payload, # Keep payload under 'data' key for consistency
            "creation_time": self.creation_time,
            "assigned_agent": self.assigned_agent,
            "completion_time": self.completion_time
        }

    def update_status(self, new_status: str, agent_name: Optional[str] = None):
        """Updates the status of the TODO item."""
        self.status = new_status
        if agent_name:
            self.assigned_agent = agent_name
        if new_status in ["Completed", "Failed"]:
            self.completion_time = time.time()
        print(f"TODO {self.task_id} status updated to {new_status}" + (f" by {agent_name}" if agent_name else ""))

# Real-Time Forecasting Manager (Adapted from prompt)
class RealTimeForecastingManager:
    """Manager for real-time forecasting using Temporal Prime Scaling agents."""
    def __init__(self, config: Dict):
        self.config = config
        # Initialize the TPS module which creates the agents
        self.module = TemporalPrimeScalingModule(config)
        self.agents = self.module.initialize_agents() # Get agents from module
        # Create DFSN with these specific agents
        self.flow_network = DynamicFlowStateNetwork(self.agents, task_complexity_threshold=13, max_agents=10)
        self.memory = MemorySystem() # Use a shared memory system
        # Assign agents for easier access
        self.temporal_agent = self._find_agent(TemporalPrimeAgent)
        self.hopfield_agent = self._find_agent(HopfieldAgent)
        self.fractal_agent = self._find_agent(FractalAgent)
        if not all([self.temporal_agent, self.hopfield_agent, self.fractal_agent]):
            raise RuntimeError("Failed to initialize all required TPS agents.")
        print("RealTimeForecastingManager initialized.")

    def _find_agent(self, agent_class: type) -> Optional[DFSNAgent]:
         """Utility to find the first agent of a specific class."""
         for agent in self.agents:
             if isinstance(agent, agent_class):
                 return agent
         return None

    def process_real_time_data(self, time_series_stream: List[float], t: int, iterations: int = 80):
        """Processes a single time step of data using the TPS agents (synchronous)."""
        if t < 0 or t >= len(time_series_stream):
            print(f"Error: Time index {t} out of bounds for time series length {len(time_series_stream)}.")
            return None

        print(f"\n--- Processing Real-Time Data at t={t} ---")
        current_complexity = 13.0 # Fixed complexity for this manager example
        # Adjust flow states and scale agents based on complexity
        self.flow_network.adjust_flow_states(current_complexity) # Let DFSN handle adjustments

        # Extract current data point and relevant history window
        x_t = time_series_stream[t]
        # Use a history window relevant for agents (e.g., memory horizon, HNN input)
        history_window = time_series_stream[max(0, t - self.config.get("memory_horizon", 20)) : t+1] # Includes current x_t
        hnn_input_window = time_series_stream[max(0, t - self.config.get("hnn_size", 100)) : t+1] # Window for HNN/Fractal


        # --- Agent Execution Sequence (Synchronous) ---

        # 1. Temporal Prime Agent: Calculate internal state z_t, predict next value
        # Parameters for temporal scaling task
        temp_params = {"beta": 0.1, "eta": 0.05, "lambda": 0.2, "delta": 0.15, "gamma": 0.1}
        temp_task = {
            "type": "temporal_scaling", "action": "temporal_scaling",
            **temp_params, "t": t, "x_t": x_t,
            "complexity": 7.0 # Assign complexity
        }
        print(f"Executing Temporal Agent Task...")
        temp_result_pkg = self.temporal_agent._execute_single_task_iteration(temp_task)
        if "error" in temp_result_pkg: print(f"  Error: {temp_result_pkg['error']}")
        temp_result = temp_result_pkg.get("result", {})
        z_t = temp_result.get("z_t", 0.0)
        pred = temp_result.get("pred", x_t) # Default prediction is current value
        error = temp_result.get("error", 0.0)
        # Get hidden state if DL was used
        h_t_from_temporal = self.temporal_agent.lstm(torch.tensor([[x_t]], dtype=torch.float32)).detach().numpy().flatten() if self.temporal_agent.use_dl and self.temporal_agent.lstm else np.zeros(self.config["hidden_size"])


        # 2. Hopfield Agent: Update state based on context and temporal results
        # Generate context vector (e.g., based on z_t or recent history)
        context_vector = np.random.rand(self.config["hnn_size"]) * z_t # Example context
        hnn_task = {
            "type": "hnn_update", "action": "hnn_update",
            "state": np.random.choice([-1, 1], self.config["hnn_size"]), # Needs previous state ideally
            "context_vector": context_vector.tolist(),
            "zero_equilibrium": 0.5, "alpha": 0.1, "t": t,
            "x_t": hnn_input_window, # Pass relevant window
            "delta": temp_params["delta"], "gamma": temp_params["gamma"],
            "h_t": h_t_from_temporal.tolist(), # Use hidden state from temporal agent
            "use_dl": self.config["use_deep_learning"],
            "complexity": 6.0
        }
        print(f"Executing Hopfield Agent Task...")
        hopfield_result_pkg = self.hopfield_agent._execute_single_task_iteration(hnn_task)
        if "error" in hopfield_result_pkg: print(f"  Error: {hopfield_result_pkg['error']}")
        hopfield_result = hopfield_result_pkg.get("result", {})
        hnn_state = hopfield_result.get("hnn_state", np.zeros(self.config["hnn_size"]))


        # 3. Fractal Agent: Generate fractal sequence based on HNN state and temporal context
        fractal_task = {
            "type": "fractal_generate", "action": "fractal_generate",
            "initial_c": z_t + 0.1j, # Use temporal state as part of initial complex value
            "iterations": iterations,
            "embeddings": np.random.rand(self.config["hnn_size"]), # Example embeddings
            "gamma_factor": 0.05,
            "hnn_state": hnn_state, # Use state from Hopfield agent
            "x_t": hnn_input_window, # Pass relevant window
            "delta": temp_params["delta"], "gamma": temp_params["gamma"],
            "h_t": h_t_from_temporal.tolist(), # Use hidden state from temporal agent
            "use_dl": self.config["use_deep_learning"],
            "complexity": 5.0
        }
        print(f"Executing Fractal Agent Task...")
        fractal_result_pkg = self.fractal_agent._execute_single_task_iteration(fractal_task)
        if "error" in fractal_result_pkg: print(f"  Error: {fractal_result_pkg['error']}")
        fractal_result = fractal_result_pkg.get("result", {})
        fractal_sequence = fractal_result.get("fractal_sequence", [])
        fractal_dimension = fractal_result.get("fractal_dimension", 0.0)

        # --- Store and Return Combined Results ---
        combined_results = {
            "time_step": t,
            "input_value": x_t,
            "temporal_state_z": z_t,
            "prediction": pred,
            "prediction_error": error,
            "hopfield_state_final_norm": float(np.linalg.norm(hnn_state)), # Example summary
            "fractal_dimension": fractal_dimension,
            # Optionally include full sequences if needed
            # "fractal_sequence": fractal_sequence,
        }

        # Store results in memory
        self.memory.store_episode(episode=1, iteration=t, results=combined_results) # Use t as iteration for simplicity

        print(f"--- Real-Time Processing Complete for t={t} ---")
        return combined_results



class ZSGManager:
    """Orchestrates the entire ZeroSumGame Framework."""
    def __init__(self, complexity: float = 6.0, vocab_size: int = 1000):
        print("Initializing ZSGManager...")
        self.complexity = complexity
        self.math_module = PyZeroMathTorch()
        self.f0z_algebra = F0ZAlgebra()

        # Core systems - Initialize containers first
        self.agents: List[DFSNAgent] = []
        self.agent_registry: Dict[str, Agent] = {}
        self.memory_system = MemorySystem()
        self.resource_monitor = ResourceMonitor()
        self.mliw = MLIW()
        self.chaos_module = ChaosTheoryModule()
        self.quantum_bridge = ZSGQuantumBridge(n_logical_qubits=4, n_physical_per_logical=5)

        # Initialize components that depend on the containers but not specific agents yet
        self.flow_state_network = DynamicFlowStateNetwork(self.agents, task_complexity_threshold=5.0, max_agents=15)
        self.multi_agent_coordinator = MultiAgentCoordinator(self.agents)

        # Initialize LLM interface Agent (doesn't go in self.agents by default)
        #self.f0z_llm_agent = F0ZAgent("F0ZAgent_LLM")
        self.f0z_llm_agent = F0ZAgent()

        # Initialize LLM for NLP parsing
        try:
             model_name = "gpt2"
             device = 0 if torch.cuda.is_available() else -1
             self.nlp_pipeline = pipeline("text-generation", model=model_name, device=device, max_new_tokens=150)
             print(f"ZSGManager NLP pipeline initialized with {model_name}.")
        except Exception as e:
             print(f"Warning: Failed to load NLP pipeline for ZSGManager: {e}. Using basic parsing.")
             self.nlp_pipeline = None

        self.episode_history = {}
        self.is_active = False

        # --- Initialize Agents SECTION ---
        # Initialize QRL Agent first as it uses the bridge
        self.qrl_agent = CuriosityQRLAgent("QRL_Agent_1", self.quantum_bridge, vocab_size)
        self._add_agent_instance(self.qrl_agent) # Add QRL agent to registry and agents list

        # Populate initial *other* default agents (calls _add_agent_instance)
        self._initialize_default_agents(vocab_size) # <<< MOVED EARLIER

        # --- Initialize Components Dependent on Agents ---
        # Now that agents exist, initialize components that need them
        self.batch_processor = DynamicBatchModeProcessor(self)
        # Science fair mode initialization now happens AFTER agents are created
        try:
            self.science_fair_mode = ZSGBatchModeScienceFair(self) # <<< MOVED HERE
        except RuntimeError as e:
             print(f"Critical Error initializing Science Fair Mode: {e}. Check agent dependencies.")
             # Decide how to handle this - maybe disable science fair mode?
             self.science_fair_mode = None # Disable if init fails

        print(f"ZSGManager initialized successfully with {len(self.agents)} agents.")


    def _initialize_default_agents(self, vocab_size):
         """Creates and adds the default set of specialized agents."""
         print("Initializing default ZSG agents...")
         # NOTE: QRL Agent is already added before this call in __init__
         default_agents_to_add = [
             PhysicsAgent("PhysicsAgent_1"),
             QuantumAgent("QuantumAgent_1"),
             MemoryAgent("MemoryAgent_1"),
             CollaborativeAgent("CollaborativeAgent_1"),
             TemporalPrimeAgent("TemporalPrimeAgent_1"),
             OrganicChemistryAgent("OrganicChemistryAgent_1"),
             MolecularBiologyAgent("MolecularBiologyAgent_1"),
             CreativeAgent("CreativeAgent_1"),
             InformationTheoryAgent("InformationTheoryAgent_1"),
             HypothesisAgent("HypothesisAgent_1"),
             DataScienceAgent("DataScienceAgent_1"),
             AstrophysicsAgent("AstrophysicsAgent_1"),
             RoboticsAgent("RoboticsAgent_1"),
             EnvironmentalScienceAgent("EnvironmentalScienceAgent_1"),
             MachineLearningAgent("MachineLearningAgent_1"),
             ValidationAgent("ValidationAgent_1"),
             FractalAgent("FractalAgent_1"),
             HopfieldAgent("HopfieldAgent_1"),
             LLaDATaskAgent("LLaDA_Agent_1", self.quantum_bridge)
         ]
         for agent in default_agents_to_add:
              # Check if agent already exists (e.g., if QRL was listed here too)
              if agent.name not in self.agent_registry:
                  self._add_agent_instance(agent)

         # Link collaborative peers (this logic is now safe)
         collab_agent = self.agent_registry.get("CollaborativeAgent_1")
         if collab_agent and isinstance(collab_agent, CollaborativeAgent):
             collab_agent.set_agent_registry(self.agent_registry)
             for agent_name in self.agent_registry:
                  if agent_name != collab_agent.name:
                      collab_agent.add_peer(agent_name)

         # Provide bridge to QuantumAgent (this logic is now safe)
         q_agent = self.agent_registry.get("QuantumAgent_1")
         if q_agent and isinstance(q_agent, QuantumAgent):
              q_agent.set_bridge(self.quantum_bridge)

         print("Default agents initialized and linked.")


    def _add_agent_instance(self, agent: Agent):
         """Adds an initialized agent instance to the manager."""
         if agent.name in self.agent_registry:
              print(f"Warning: Agent with name {agent.name} already exists. Skipping.")
              return
         if isinstance(agent, DFSNAgent): # Only add DFSNAgents to lists managed by DFSN/Coordinator
             self.agents.append(agent) # Add to the shared list
         self.agent_registry[agent.name] = agent
         # Update coordinator's view of agents (it uses the shared list reference)
         self.multi_agent_coordinator.agent_states[agent.name] = {"state": "idle", "load": 0.0, "engagement": agent.get_engagement_state()}
         # Update DFSN's view
         self.flow_state_network.agent_states[agent.name] = agent.flow_state if hasattr(agent, 'flow_state') else 'idle'
         # print(f"Agent {agent.name} added to ZSGManager.")


    def activate(self, episode: int, iteration: int):
        """Activates the ZSG framework components for an episode."""
        if self.is_active:
            print("ZSG Framework is already active.")
            return
        print(f"\n--- Activating ZSG Framework ---")
        self.mliw.start_episode(episode, iteration)
        # Start monitoring, enable dynamic states
        self.resource_monitor.start()
        self.flow_state_network.enable_dynamic_states()
        # Perform initial resource allocation based on current complexity
        self.calibrate_episode(f"Activation for Episode {episode}", self.complexity)
        self.is_active = True
        print("--- ZSG Framework Activated ---")

    def deactivate(self):
        """Deactivates the ZSG framework components."""
        if not self.is_active:
            print("ZSG Framework is already inactive.")
            return
        print(f"\n--- Deactivating ZSG Framework ---")
        self.resource_monitor.stop()
        self.flow_state_network.disable_dynamic_states()
        self.is_active = False
        # Clear task queue? Reset agent states?
        self.multi_agent_coordinator.task_queue = []
        print("--- ZSG Framework Deactivated ---")

    def calibrate_episode(self, prompt: str, complexity: float):
        """Calibrates system parameters for a new episode or complexity level."""
        print(f"Calibrating ZSG for complexity {complexity:.2f}. Prompt: '{prompt}'")
        self.complexity = complexity
        # Adjust DFSN threshold based on overall complexity
        self.flow_state_network.task_complexity_threshold = complexity * 0.8 # e.g., 80% of overall
        print(f"  DFSN Threshold set to: {self.flow_state_network.task_complexity_threshold:.2f}")
        # Pre-allocate resources
        self.resource_monitor.pre_allocate(self.agents, complexity)
        # Scale agents based on new complexity
        self.flow_state_network.scale_agents(complexity) # Note: Scaling might add conceptual agents

    def enforce_non_repeat(self, task: Dict, episode: int) -> Dict:
        """Checks if a similar task was done and modifies it slightly if needed."""
        # Simple check based on task type and key parameters
        task_signature = f"{task.get('type', '')}_{task.get('action', '')}"
        # Check history for similar signatures (this could be more sophisticated)
        for past_episode, past_result in self.episode_history.items():
             past_task = past_result.get("task_input") # Assuming task is stored in result
             if past_task:
                  past_signature = f"{past_task.get('type', '')}_{past_task.get('action', '')}"
                  if task_signature == past_signature:
                       # Found similar task type/action, modify current task slightly
                       print(f"Non-repeat: Modifying task {task_signature} based on past execution in {past_episode}")
                       task['complexity'] = task.get('complexity', 5.0) * 1.05 # Slightly increase complexity
                       # Modify a data parameter if possible
                       if 'data' in task and isinstance(task['data'], dict):
                            for key, val in task['data'].items():
                                 if isinstance(val, (int, float)):
                                      task['data'][key] = val * (1 + random.uniform(-0.05, 0.05)) # Perturb numeric data
                                      break # Modify one parameter and stop
                       # Mark task as modified
                       task['modified_for_non_repeat'] = True
                       return task # Return modified task
        return task # Return original task if no similar one found

    def process_task_with_zsg(self, task: Dict) -> Dict:
        """Main processing function for a single task within the ZSG framework."""
        if not self.is_active:
             print("Warning: ZSG Framework is not active. Processing task with defaults.")
             # Fallback: Execute directly via coordinator without full framework features
             agent = self.agent_registry.get(task.get("agent_name")) # Requires agent name in task
             if agent: return agent._execute_single_task_iteration(task)
             else: return {"error": "ZSG inactive and no specific agent found."}


        episode, iteration = self.mliw.get_state()
        print(f"\n=== ZSG Processing Task (E{episode}, I{iteration}) ===")
        print(f"Input Task: Type='{task.get('type', 'N/A')}', Action='{task.get('action', 'N/A')}', Complexity={task.get('complexity', self.complexity):.1f}")

        # 1. Enforce Non-Repetition
        task = self.enforce_non_repeat(task, episode)

        # 2. Assign task via Coordinator
        # The coordinator finds the right agent(s) and calls their _execute_single_task_iteration (which uses AIW internally)
        result_package = self.multi_agent_coordinator.assign_tasks(task)

        # 3. Update Resource Allocations (based on current agent states)
        self.resource_monitor.update_allocations(self.agents)

        # 4. Store result in Memory System
        self.memory_system.store_episode(episode, iteration, {"task_input": task, "result_package": result_package})
        # Store in history for non-repeat check
        self.episode_history[f"E{episode}_I{iteration}"] = {"task_input": task, "result_package": result_package}
        if len(self.episode_history) > 500: # Limit history size
             self.episode_history.pop(list(self.episode_history.keys())[0])


        # 5. Advance MLIW iteration
        self.mliw.next_iteration()

        print(f"=== ZSG Task Processing Complete ===\n")
        return result_package


    def process_nlp_command(self, command: str) -> Dict:
         """Parses an NLP command and executes the corresponding ZSG task."""
         if not self.is_active:
             self.activate(1, 1) # Auto-activate if command received while dormant

         print(f"\n--- Processing NLP Command: '{command}' ---")
         # Use LLM for parsing if available, otherwise use simple regex/keyword matching
         task_dict = self.nlp_parse(command)

         result = {}
         explanation = "Could not generate explanation."

         if task_dict.get("type") == "unknown":
              result = {"error": "Could not understand command."}
         elif task_dict.get("type") == "batch_science_fair":
             # Trigger science fair mode
              experiments = task_dict.get("experiments", ["Default Experiment 1", "Default Experiment 2"])
              sf_results = self.science_fair_mode.run_science_fair(experiments)
              result = {"science_fair_summary": sf_results.get("final_summary", "Error in summary")}
         elif task_dict.get("type") == "chaos":
              result = self.chaos_module.simulate(task_dict["system"], task_dict["params"])
              # Optionally trigger DFSN adjustment based on chaos result
              if "chaos_metrics" in result:
                   self.flow_state_network.handle_chaos(result["chaos_metrics"])
         elif task_dict.get("type") == "quantum_gate":
              try:
                  gate_func = getattr(self.quantum_bridge.simulator, f"{task_dict['gate'].lower()}_gate")
                  gate_func(*task_dict['qubits']) # Apply gate
                  result = {"result": {"quantum_state_vector": self.quantum_bridge.simulator.state.tolist()}, "agent": "QuantumBridge"}
              except Exception as e:
                   result = {"error": f"Failed to apply quantum gate: {e}"}
         elif task_dict.get("type") == "llm_inference":
               result = self.f0z_llm_agent._execute_single_task_iteration(task_dict)
         elif task_dict.get("type") == "f0z_simulation":
              result = self.f0z_llm_agent._execute_single_task_iteration(task_dict)
         else:
              # Assign as a standard task to the coordinator
              self.multi_agent_coordinator.add_task(task_dict, priority=int(task_dict.get("complexity", 5)))
              result = self.multi_agent_coordinator.execute_next_task()
              if result is None: result = {"status": "No task executed (queue might be empty or assignment failed)"}


         # Generate explanation using LLM if available
         if self.nlp_pipeline:
             try:
                  explanation_prompt = f"Explain the following ZSG framework result for command '{command}'. Result: {json.dumps(result, default=lambda o: '')[:200]}..."
                  explanation_result = self.nlp_pipeline(explanation_prompt, max_length=100)
                  explanation = explanation_result[0]['generated_text']
                  # Clean explanation
                  if explanation.startswith(explanation_prompt):
                       explanation = explanation[len(explanation_prompt):].strip()
             except Exception as e:
                  explanation = f"Error generating explanation: {e}"
         else:
             explanation = f"NLP explanation disabled. Task Type: {task_dict.get('type')}, Result Status: {'error' if 'error' in result else 'success'}"

         print(f"--- NLP Command Processing Complete ---")
         return {"result": result, "explanation": explanation}


    def nlp_parse(self, command: str) -> Dict:
         """Parses natural language command into a task dictionary (improved)."""
         cmd = command.lower().strip()
         task = {"type": "unknown", "complexity": 5.0, "action": None, "data": {}} # Default structure

         # Keyword/Regex based parsing (example rules)
         if re.search(r"(run|start) science fair", cmd):
              task["type"] = "batch_science_fair"
              task["complexity"] = 10.0
              # Extract experiment descriptions if possible
              exps = re.findall(r'experiments? "?(.*?)"? (and|with|featuring) "?(.*?)"?', cmd)
              if exps: task["experiments"] = [e.strip() for group in exps for e in group if e not in ['and', 'with', 'featuring'] and e]
              elif "experiments" in cmd: task["experiments"] = [s.strip() for s in cmd.split("experiments")[-1].split(',') if s.strip()]
              else: task["experiments"] = ["Quantum Entanglement Effects", "F0Z Stability Analysis"]

         elif re.search(r"simulate lorenz", cmd):
              task["type"] = "chaos"
              task["system"] = "lorenz"
              task["params"] = {"t_max": 15.0, "steps": 300} # Default params
              # Try extracting params
              if "rho=" in cmd: task["params"]["rho"] = float(cmd.split("rho=")[-1].split()[0])
              task["complexity"] = 7.0

         elif re.search(r"(apply|run) (h|hadamard|x|y|z|rz|ry|cnot|cx|cz) gate", cmd):
              task["type"] = "quantum_gate"
              gate_match = re.search(r"(h|hadamard|x|y|z|rz|ry|cnot|cx|cz) gate", cmd)
              task["gate"] = gate_match.group(1).upper() if gate_match else "H"
              if task["gate"] == "HADAMARD": task["gate"] = "H"
              if task["gate"] == "CX": task["gate"] = "CNOT"

              qubit_matches = re.findall(r"qubit (\d+)", cmd)
              qubits = [int(q) for q in qubit_matches] if qubit_matches else [0] # Default qubit 0
              task["qubits"] = qubits
              # Adjust structure for specific gates
              if task["gate"] in ["RZ", "RY"]:
                    task["qubit"] = qubits[0] # Single target
                    angle_match = re.search(r"angle ([0-9.]+)", cmd)
                    task["angle"] = float(angle_match.group(1)) if angle_match else np.pi/2
              elif task["gate"] in ["CNOT", "CZ"]:
                    if len(qubits) < 2: qubits = [0, 1] # Default control/target
                    task["control"] = qubits[0]
                    task["target"] = qubits[1]
              task["complexity"] = 4.0

         elif re.search(r"ask llm|infer|explain f0z", cmd):
              task["type"] = "llm_inference"
              task["action"] = "infer"
              query_match = re.search(r"(?:ask llm about|infer|explain f0z) (.*)", command, re.IGNORECASE) # Capture query after keyword
              task["query"] = query_match.group(1).strip() if query_match else "What is the Formula for Zero?"
              task["complexity"] = 3.0

         elif re.search(r"simulate f0z reaction", cmd):
               task["type"] = "f0z_simulation"
               task["action"] = "simulate_f0z_reaction"
               dpd_match = re.search(r"d/pd ratio of ([0-9.]+)", cmd)
               task["D_Pd_ratio"] = float(dpd_match.group(1)) if dpd_match else 0.8
               task["complexity"] = 6.0

         elif re.search(r"train ml model", cmd):
              task["type"] = "machine_learning"
              task["action"] = "train"
              # Requires data source definition - simplified: use dummy data
              task["X"] = [[1,1], [2,1], [3,2], [4,2]]
              task["Y"] = [[2], [3], [5], [6]]
              task["complexity"] = 8.0

         elif re.search(r"predict with ml model", cmd):
               task["type"] = "machine_learning"
               task["action"] = "predict"
               task["X"] = [[5,3], [6,3]]
               task["complexity"] = 3.0

         elif re.search(r"validate last result", cmd):
               task["type"] = "validation"
               task["action"] = "verify_result"
               # Requires mechanism to get last result and task data
               last_hist_key = list(self.episode_history.keys())[-1] if self.episode_history else None
               if last_hist_key and "result_package" in self.episode_history[last_hist_key] and "task_input" in self.episode_history[last_hist_key]:
                    task["original_result"] = self.episode_history[last_hist_key]["result_package"].get("result")
                    task["task_data"] = self.episode_history[last_hist_key]["task_input"]
                    task["complexity"] = 4.0
               else:
                    task["error"] = "Cannot validate: No previous result found in history."
                    task["type"] = "error_task"


         # Add more parsing rules here...

         print(f"NLP Parsed Task: {task}")
         return task

# --- CLI Shell ---
class ScienceFairShell:
    """Command Line Interface for interacting with the ZSG Manager."""
    def __init__(self):
        self.manager = ZSGManager()
        # self.session = PromptSession(history=None) # Keep using input for now
        print("\n--- ZSG Science Fair Shell Initialized ---")
        print("Type NLP commands (e.g., 'simulate lorenz', 'apply h gate qubit 0', 'ask llm about f0z', 'run science fair') or 'exit'.")

    def run(self):
        """Starts the interactive shell."""
        self.manager.activate(episode=1, iteration=1)

        while True:
            try: # Main try block for input and processing
                command = input("Science Fair (basic)> ")
                command = command.strip()

                if not command: continue
                if command.lower() == "exit":
                    print("Exiting Science Fair Shell.")
                    break # Correctly exits the loop

                # Process the command using the manager's NLP handler
                output = self.manager.process_nlp_command(command)

                # Print results and explanation nicely
                print("\n--- Result ---")
                try: # Inner try specifically for JSON dumping
                    print(json.dumps(output.get('result', {}), indent=2, default=lambda o: ''))
                except Exception as json_e:
                     print(f"Could not format result as JSON: {json_e}")
                     print(output.get('result', {})) # Print raw result if JSON fails

                print("\n--- Explanation ---")
                print(output.get('explanation', "No explanation available."))
                print("-" * 17 + "\n") # Separator


            except EOFError:
                 print("\nEOF detected (input stream ended). Exiting.")
                 break # Exit loop
            except KeyboardInterrupt:
                 print("\nKeyboard interrupt detected. Exiting.")
                 break # Exit loop
            except Exception as e:
                 print(f"\nShell Error: An unexpected error occurred during command processing: {e}")
                 # Optionally add more detailed logging or decide whether to continue/break
                 # traceback.print_exc() # For debugging


        # Deactivate manager AFTER the while loop finishes
        self.manager.deactivate()

# --- CLI Shell ---
# (ScienceFairShell class definition - use corrected version from previous step)
# ...

# --- Example Usage & Main Execution ---
if __name__ == "__main__":
    print("*"*70); print(" ZSG Framework - Science Fair Edition (Qiskit Integrated) "); print("*"*70)
    try:
        shell = ScienceFairShell()
        shell.run()
    except ImportError as e:
         print(f"\nCRITICAL ERROR DURING STARTUP: {e}")
         print("Please ensure all required libraries, including Qiskit and its dependencies, are installed.")
    except Exception as e:
         print(f"\nUNEXPECTED ERROR DURING STARTUP OR EXECUTION: {e}")
         import traceback
         traceback.print_exc()
    finally:
        print("\nZSG Framework execution finished.")

# --- End of Merged File ---
