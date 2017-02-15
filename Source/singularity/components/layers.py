from ..core import *
from .nodes import *

###############
# Input Layer #
###############   
         
class InputLayer(InputNode):
    #constructor
    def __init__(self, shape = None):
        params = extract_args_kwargs(locals())
        super(InputLayer, self).__init__(**params)
         
###############
# Dense Layer #
###############

class DenseLayer(Node):
    #constructor
    def __init__(self, dimension, 
                init_weights = glorot_uniform, init_bias = zeros, activation = tanh, trainable = True, **kwargs):
        params = extract_args_kwargs(locals())
        super(DenseLayer, self).__init__(**params)
    ################################################
    #overridemethod
    def _get_shape_from_inputs(self):
        return (None, self._dimension)
    #overridemethod
    def _prepare_params(self, input_node):
        W = variable(self._init_weights((input_node.shape[1], self._dimension) ))
        b = variable(self._init_bias((self._dimension,)))
        return {'W': W, 'b':b}
    def _set_settings(self, settings): # Could be more than 2 parameters. Check inputs and order.
        W = variable(params[1])
        b = variable(params[0])
        self._params.append({'W': W, 'b':b})
    #overridemethod
    def _pre_activation(self, data, params):
        return dot(data, params['W']) + params['b']
    def _pre_back_activation(self, data, params):
        return dot(data, params['W'].T) + params['b']
      
################
# Conv2D Layer #
################

class Conv2DLayer(Node):
    #constructor
    def __init__(self, feature_maps, filter_shape, max_pool_shape = None, padding = 0, 
                init_weights = glorot_uniform, init_bias = zeros, activation = relu, trainable = True, **kwargs):
        params = extract_args_kwargs(locals())
        super(Conv2DLayer, self).__init__(**params)
    ################################################
    #overridemethod
    def _get_shape_from_inputs(self):     
        input_node = self._inputs[0]
        conv_shape = (self._feature_maps, input_node.shape[1], self._filter_shape[0], self._filter_shape[1])
        reduction = self._padding - conv_shape[2] // 2 
        output_shape = np.asarray(input_node.shape)
        output_shape[1] = conv_shape[0]
        output_shape[2:] = output_shape[2:] + (reduction * 2)
        if (self._max_pool_shape is not None):
            output_shape[2:] = output_shape[2:] // self._max_pool_shape[0]
        return tuple(output_shape)
    #overridemethod
    def _prepare_params(self, input_node):
        conv_shape = (self._feature_maps, input_node.shape[1], self._filter_shape[0], self._filter_shape[1])
        W = variable(self._init_weights(conv_shape))
        b = variable(self._init_bias((conv_shape[0],)))
        return {'W': W, 'b':b}
    #overridemethod
    def _pre_activation(self, data, params):
        if (self._padding != 0):
            padding(data, self._padding)
        return convolution2d(data, params['W']) + dimshuffle(params['b'],(None, 0, None, None))
    #overridemethod
    def _post_activation(self, outputs):
        if (self._max_pool_shape is not None):
            outputs = max_pool_2d(outputs, self._max_pool_shape)
        return outputs
 
#################
# Padding Layer #
#################

class PaddingLayer(Node):
    #constructor
    def __init__(self, padding = 1, inputs = None):
        params = extract_args_kwargs(locals())
        super(PaddingLayer, self).__init__(**params)
    ################################################
    #overridemethod
    def _get_shape_from_inputs(self):
        return get_padded_shape(self._inputs[0].shape, self._padding)
    #overridemethod
    def _post_activation(self, outputs):
        return padding(outputs, self._padding)

#################
# Flatten Layer #
#################

class FlattenLayer(Node):
    #constructor
    def __init__(self, dimensions = 2, inputs = None):
        params = extract_args_kwargs(locals())
        super(FlattenLayer, self).__init__(**params)
    ################################################
    #overridemethod
    def _get_shape_from_inputs(self):
        return get_flattened_shape(self._inputs[0].shape, self._dimensions)
    #overridemethod
    def _post_activation(self, outputs):
        return flatten(outputs, self._dimensions)

#################
# Dropout Layer #
#################

class DropoutLayer(Node):
    #constructor
    def __init__(self, p = 0.5, inputs = None):
        params = extract_args_kwargs(locals())
        super(DropoutLayer, self).__init__(**params)
    ################################################
    #overridemethod
    def _get_shape_from_inputs(self):
        return self._inputs[0].shape
    #overridemethod
    def _post_activation(self, outputs):
        return random_binomial(outputs,self._p)             

#####################
# Activation Layer #
#####################

class ActivationLayer(Node):
    #constructor
    def __init__(self, activation, inputs = None):
        params = extract_args_kwargs(locals())
        super(ActivationLayer, self).__init__(**params)
    ################################################
    #overridemethod
    def _get_shape_from_inputs(self):
        return self._inputs[0].shape


