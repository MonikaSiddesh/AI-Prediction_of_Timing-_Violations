import pandas as pd
import random

# Define possible gate types
gate_types = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR", "NOT"]

# Function to generate a random circuit
def generate_circuit(circuit_id):
    num_gates = random.randint(5, 50)  # Randomly select number of gates in the circuit
    gates = random.choices(gate_types, k=num_gates)  # Select gate types randomly
    
    max_path_length = random.randint(2, num_gates)  # Maximum depth in the circuit
    logic_depth = max_path_length  # Longest path leading to a flip-flop

    return [circuit_id, num_gates, ",".join(gates), max_path_length, logic_depth]

# Generate dataset with 150 rows
num_samples = 150
data = [generate_circuit(i) for i in range(1, num_samples + 1)]

# Create DataFrame
columns = ["Circuit_ID", "Num_Gates", "Gate_Types", "Max_Path_Length", "Logic_Depth"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("logic_depth_dataset.csv", index=False)

print("Dataset generated successfully: logic_depth_dataset.csv")
