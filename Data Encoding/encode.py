# Functions for amplitude encoding and decoding

import numpy as np
from qiskit import QuantumCircuit, Aer, execute

norm = 0

def encode(dataset):
    global norm

    # Calculate the number of qubits needed to encode the values
    num_qubits = int(np.ceil(np.log2(len(dataset))))
    qc = QuantumCircuit(num_qubits)

    # Initialize the quantum circuit to the desired state
    norm = np.linalg.norm(dataset)
    desired_state = dataset / norm
    desired_state = np.pad(desired_state, (0, 2 ** num_qubits - len(desired_state)), mode='constant', constant_values=0)
    qc.initialize(desired_state, list(range(num_qubits)))

    # Return the quantum circuit
    return qc

def decode(qc, size):
    global norm
    
    shots = 100000

    # Measure the quantum circuit
    qc.measure_all()
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Use the counts to estimate the values
    decoded = np.zeros(size)
    for i, j in counts.items():
        decoded[int(i, 2)] = round((j / shots) ** 0.5 * norm)
    
    return decoded