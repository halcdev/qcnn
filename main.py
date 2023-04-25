from qiskit.circuit import QuantumCircuit, Parameter
import matplotlib.pyplot as plt

theta = Parameter('θ')

qc = QuantumCircuit(2)
qc.rz(theta, 0)
qc.crz(theta, 0, 1)
fig = qc.draw(output="mpl")

# Display the figure in a plot window
plt.show()