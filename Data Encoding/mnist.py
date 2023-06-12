# Functions for reading the MNIST data

import numpy as np

# Read file with images and store values in a numpy array
def read_images(filename):
    with open(filename, 'rb') as fid:
        magic_number = int.from_bytes(fid.read(4), 'big')
        num_images = int.from_bytes(fid.read(4), 'big')
        num_rows = int.from_bytes(fid.read(4), 'big')
        num_cols = int.from_bytes(fid.read(4), 'big')
        images = np.frombuffer(fid.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
    return images

# Read file with labels and store values in a numpy array
def read_labels(filename):
    with open(filename, 'rb') as fid:
        magic_number = int.from_bytes(fid.read(4), 'big')
        num_labels = int.from_bytes(fid.read(4), 'big')
        labels = np.frombuffer(fid.read(), dtype=np.uint8)