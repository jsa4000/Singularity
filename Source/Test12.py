import numpy as np
import matplotlib.pyplot as plt
import singularity as S
from singularity.components.layers import *
from singularity.components.optimizers import *
from singularity.components.regularizers import *
from singularity.components.models import *
from singularity.utils import datasets
from singularity.utils import hdf5

batch_size = 16
original_dim = 784
latent_dim = 2
intermediate_dim = 128
 #epsilon_std = 0.01
epsilon_std = 0.01
nb_epoch = 1

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = datasets.load_mnist()

x_train = x_train.astype(S.floatX()) / 255.
x_test = x_test.astype(S.floatX()) / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

###################
## SAVE HDF5 TEST #
###################

#Save the model into a hdf5 file format
#hdf5.save("test2.hdf", _x_decoded_mean, "root")

# Load the data settings
_x_decoded_mean = hdf5.load("test2.hdf")


generator = GraphModel(outputs = _x_decoded_mean)
generator.build()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon_std
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
