import numpy as np
import matplotlib.pyplot as plt
import singularity as S
from singularity.components.layers import *
from singularity.components.optimizers import *
from singularity.components.regularizers import *
from singularity.components.models import *
from singularity.utils import datasets
import h5py

from scipy.misc import imsave
import time
import os

train = True

# dimensions of the generated pictures for each filter.
img_width = 128
img_height = 128

# build the VGG16 network
model = OverlayModel()
model.add(InputLayer((None, 3, img_width, img_height)))
model.add(PaddingLayer(1))

model.add(Conv2DLayer(64, (3, 3), activation=S.relu, name='conv1_1'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(64, (3, 3), activation=S.relu, max_pool_shape = (2, 2), name='conv1_2'))

model.add(PaddingLayer(1))
model.add(Conv2DLayer(128, (3, 3), activation=S.relu, name='conv2_1'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(128, (3, 3), activation=S.relu, max_pool_shape = (2, 2), name='conv2_2'))

model.add(PaddingLayer(1))
model.add(Conv2DLayer(256, (3, 3), activation=S.relu, name='conv3_1'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(256, (3, 3), activation=S.relu, name='conv3_2'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(256, (3, 3), activation=S.relu,max_pool_shape = (2, 2), name='conv3_3'))

model.add(PaddingLayer(1))
model.add(Conv2DLayer(512, (3, 3), activation=S.relu, name='conv4_1'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(512, (3, 3), activation=S.relu, name='conv4_2'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(512, (3, 3), activation=S.relu, max_pool_shape = (2, 2), name='conv4_3'))

model.add(PaddingLayer(1))
model.add(Conv2DLayer(512, (3, 3), activation=S.relu, name='conv5_1'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(512, (3, 3), activation=S.relu, name='conv5_2'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(512, (3, 3), activation=S.relu, max_pool_shape = (2, 2), name='conv5_3'))
 
if (train):
    model.add(FlattenLayer())
    model.add(DenseLayer(4096, activation=S.relu))
    model.add(DropoutLayer(0.5))
    model.add(DenseLayer(4096, activation=S.relu))
    model.add(DropoutLayer(0.5))
    model.add(DenseLayer(1000, activation=S.relu))


#model.build(loss = S.categorical_cross_entropy , optimizer = Adam())

first_layer = model.layers[0]
# this is a placeholder tensor that will contain our generated images
input_img = first_layer.placeholder

# the name of the layer we want to visualize (see model definition below)
layer_name = 'conv5_1'

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

## path to the model weights file.
weights_path = 'C:\\Users\\javier.santos\\Downloads\\vgg16_weights.h5'

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    print (len(weights))
    #model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer.outputs) for layer in model.layers])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (S.sqrt(S.mean(S.square(x))) + 1e-5)

#kept_filters = []
##for filter_index in range(0, 200):  
#for filter_index in range(0, 200):  
#    # we only scan through the first 200 filters,
#    # but there are actually 512 of them
#    print('Processing filter %d' % filter_index)
#    start_time = time.time()

#    # we build a loss function that maximizes the activation
#    # of the nth filter of the layer considered
#    layer_output = layer_dict[layer_name].output
#    loss = K.mean(layer_output[:, filter_index, :, :])

#    # we compute the gradient of the input picture wrt this loss
#    grads = K.gradients(loss, input_img)[0]

#    # normalization trick: we normalize the gradient
#    grads = normalize(grads)

#    # this function returns the loss and grads given the input picture
#    iterate = K.function([input_img], [loss, grads])

#    # step size for gradient ascent
#    step = 1.

#    # we start from a gray image with some random noise
#    input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.

#    # we run gradient ascent for 20 steps
#    for i in range(20):
#        loss_value, grads_value = iterate([input_img_data])
#        input_img_data += grads_value * step

#        print('Current loss value:', loss_value)
#        if loss_value <= 0.:
#            # some filters get stuck to 0, we can skip them
#            break

#    # decode the resulting input image
#    if loss_value > 0:
#        img = deprocess_image(input_img_data[0])
#        kept_filters.append((img, loss_value))
#    end_time = time.time()
#    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

## we will stich the best 64 filters on a 8 x 8 grid.
#n = 8

## the filters that have the highest loss are assumed to be better-looking.
## we will only keep the top 64 filters.
#kept_filters.sort(key=lambda x: x[1], reverse=True)
#kept_filters = kept_filters[:n * n]

## build a black picture with enough space for
## our 8 x 8 filters of size 128 x 128, with a 5px margin in between
#margin = 5
#width = n * img_width + (n - 1) * margin
#height = n * img_height + (n - 1) * margin
#stitched_filters = np.zeros((width, height, 3))

## fill the picture with our saved filters
#for i in range(n):
#    for j in range(n):
#        img, loss = kept_filters[i * n + j]
#        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
#                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

## save the result to disk
#imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
