from qiskit.circuit import QuantumCircuit, Parameter
import matplotlib.pyplot as plt
import math

desired_state = [
    1 / math.sqrt(16) * 2,
    1 / math.sqrt(16) * 2,
    1 / math.sqrt(16) * -2,
    1 / math.sqrt(16) * 2]

qc = QuantumCircuit(2)
qc.initialize(desired_state, [0,1])

qc.decompose().decompose().decompose().decompose().decompose().draw(output="mpl")
plt.show()