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

x = InputLayer((None, original_dim))
h = DenseLayer(intermediate_dim, activation=S.relu)(x)

# This it's like directly putting the inputs from h and this function will return the values using previous outputs from h
z_mean = DenseLayer(latent_dim)(h)
z_log_std = DenseLayer(latent_dim)(h)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = S.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + S.exp(z_log_std) * epsilon

def sampling2(args):
    z_mean, z_log_std = args
    return S.sigmoid(z_mean + z_log_std)

#z = ActivationLayer(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
z = ActivationLayer(sampling)([z_mean, z_log_std])

#z = DenseLayer(latent_dim,activation=sampling, shared = True)([z_mean, z_log_std])
#z = DenseLayer(latent_dim,activation=sampling2, shared = True)([z_mean, z_log_std])

# we instantiate these layers separately so as to reuse them later
decoder_h = DenseLayer(intermediate_dim, activation=S.relu)
decoder_mean = DenseLayer(original_dim, activation=S.sigmoid)
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def vae_loss(x, x_decoded_mean):
    xent_loss = S.binary_cross_entropy(x, x_decoded_mean)
    kl_loss = - 0.5 * S.mean(1 + z_log_std.outputs - S.square(z_mean.outputs) - S.exp(z_log_std.outputs), axis=-1)
    return S.sum(xent_loss + kl_loss)
 
vae = GraphModel(x, x_decoded_mean)
vae.build(optimizer=RMSProp(), loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = datasets.load_mnist()

x_train = x_train.astype(S.floatX()) / 255.
x_test = x_test.astype(S.floatX()) / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.train(x_train, x_train,
        iterations=nb_epoch,
        batch_size=batch_size)

# build a model to project inputs on the latent space
encoder = GraphModel(x, z_mean)
encoder.build()

# display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test[0:15])
x_test_encoded = encoder.predict(x_test)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = InputLayer((None,latent_dim))
_h_decoded = decoder_h(decoder_input, False)
_x_decoded_mean = decoder_mean(_h_decoded, False)
generator = GraphModel(decoder_input, _x_decoded_mean)
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

###################
## SAVE HDF5 TEST #
###################

#Save the model into a hdf5 file format
hdf5.save("test2.hdf", _x_decoded_mean, "root")

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
