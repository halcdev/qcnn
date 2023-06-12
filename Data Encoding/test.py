# Test of amplitude encoding and decoding functions on MNIST data

import mnist
from encode import encode, decode
from qiskit import Aer, execute
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

# Use one image from the MNIST dataset as a test input
train_images = mnist.read_images('MNIST/raw/train-images-idx3-ubyte')
dataset = train_images[0]
# dataset = [1, 3, 4, 2]

# Encode the dataset
qc = encode(dataset)

# Plot the qubit states
backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()
statevector = result.get_statevector()
plot_bloch_multivector(statevector)
plt.show()

# Decode the dataset and print the values
print(decode(qc, len(dataset)))