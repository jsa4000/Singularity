import singularity as S
from singularity.components import layers, models, optimizers, regularizers
from singularity.utils import datasets 

import numpy as np
import matplotlib.pyplot as plt

# Training variables
batch_size = 16
nb_epoch = 1

# Variational Auto-encoder configuration
original_dim = 784
intermediate_dim = 128
latent_dim = 2

epsilon_std = 0.01

# Two main layer for variational auto-encoder
x = layers.InputLayer(output_shape = (None, original_dim))
h = layers.Dense(intermediate_dim, input_layer = x, activation = S.relu)  # TODO:_Doesn't like output_shape by default since for the user is not as clear as other variables.


z_mean = layers.Dense(latent_dim, input_layer = h)
z_log_std = layers.Dense(latent_dim, input_layer = h)

# Variational auto encoder implements it's own function for the last encoder
def sampling (args):
    z_mean, z_log_std = args
    epsilon = S.normal(shape=(batch_size, latent_dim), mean=0., std=epsilon_std)
    return z_mean + S.exp(z_log_std) * epsilon

z = layers.Dense(latent_dim, activation = sampling)

# we instantiate these layers separately so as to reuse them later

#decoder_h = layers.Dense(intermediate_dim, activation=S.relu)
#decoder_mean = layers.Dense(original_dim, activation=S.sigmoid)

h_decoded = layers.Dense(intermediate_dim, input_layer = z, activation=S.relu)
x_decoded_mean = layers.Dense(original_dim, input_layer = h_decoded, activation=S.sigmoid)


def vae_loss(x, x_decoded_mean):
    xent_loss = S.binary_cross_entropy(x, x_decoded_mean)
    kl_loss = - 0.5 * S.mean(1 + z_log_std - S.square(z_mean) - S.exp(z_log_std), axis=-1)
    return xent_loss + kl_loss

#vae = Model(x, x_decoded_mean)
#vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = datasets.load_mnist()

x_train = x_train.astype(S.floatX()) / 255.
x_test = x_test.astype(S.floatX()) / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#vae.fit(x_train, x_train,
#        shuffle=True,
#        nb_epoch=nb_epoch,
#        batch_size=batch_size,
#        validation_data=(x_test, x_test))

## build a model to project inputs on the latent space
#encoder = Model(x, z_mean)

## display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()

## build a digit generator that can sample from the learned distribution
#decoder_input = Input(shape=(latent_dim,))
#_h_decoded = decoder_h(decoder_input)
#_x_decoded_mean = decoder_mean(_h_decoded)
#generator = Model(decoder_input, _x_decoded_mean)

## display a 2D manifold of the digits
#n = 15  # figure with 15x15 digits
#digit_size = 28
#figure = np.zeros((digit_size * n, digit_size * n))
## we will sample n points within [-15, 15] standard deviations
#grid_x = np.linspace(-15, 15, n)
#grid_y = np.linspace(-15, 15, n)

#for i, yi in enumerate(grid_x):
#    for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]]) * epsilon_std
#        x_decoded = generator.predict(z_sample)
#        digit = x_decoded[0].reshape(digit_size, digit_size)
#        figure[i * digit_size: (i + 1) * digit_size,
#               j * digit_size: (j + 1) * digit_size] = digit

#plt.figure(figsize=(10, 10))
#plt.imshow(figure)
#plt.show()
