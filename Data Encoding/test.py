# Test of amplitude encoding and decoding functions on MNIST data

import mnist
from encode import encode, decode
from qiskit import Aer, execute
from qiskit.visualization import plot_bloch_multivector
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Use one image from the MNIST dataset as a test input
train_images = mnist.read_images('/Users/haddyalchaer/Documents/python_files/qcnn/MNIST/raw/train-images-idx3-ubyte')
dataset = train_images[0]
# dataset = [1, 3, 4, 2]

# Assuming `dataset` is the 1D array of size 728
dataset_2d = np.reshape(dataset, (28, 28))

# Downscale the image to 16x16
downscaled_image = zoom(dataset_2d, (16/28, 16/28), order=1)

# Convert the downscaled image back to a 1D array
downscaled_1d = np.reshape(downscaled_image, (256,))

# Encode the dataset
qc = encode(downscaled_1d)

# Plot the qubit states
backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()
statevector = result.get_statevector()
plot_bloch_multivector(statevector)

# Display the original and encoded/decoded values as images
plt.figure("Original Dataset")
plt.imshow(downscaled_image.reshape(16, 16), cmap='gray')
plt.figure("Decoded Dataset")
plt.imshow(decode(qc).reshape(16, 16), cmap='gray')
plt.show()