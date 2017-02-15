import numpy as np
import singularity as S
from singularity.components.layers import *
from singularity.components import layers, optimizers, regularizers, models
from singularity.utils import datasets
from time import sleep

layer01 = InputLayer((None, 1, 28, 28))
layer021 = Conv2DLayer(4,(3, 3), input_layers = layer01, activation = S.relu)
layer022 = Conv2DLayer(4,(3, 3), input_layers = layer01, activation = S.relu)
layer03 = Conv2DLayer(8,(3, 3), input_layers = [layer021, layer022], max_pool_shape = (2, 2), activation = S.relu)
layer04 = DropoutLayer(0.25, input_layers = layer03)
layer05 = FlattenLayer(2, input_layers = layer04)
layer06 = DenseLayer(144, input_layers = layer05, activation = S.relu)
layer07 = DropoutLayer(0.5, input_layers = layer06)
layer08 = DenseLayer(10, input_layers = layer07, activation = S.softmax)
    


def printlayers (x, lastlayer):
    #print lastlayer.__class__.__name__
    for layer in lastlayer._input_nodes:
        x += printlayers(x, layer)
    return x + " " + lastlayer.__class__.__name__


x = ""

printlayers (x, layer08)

