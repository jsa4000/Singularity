from ..core import *
from collections import OrderedDict

########################
# Optimizer Base Class #
########################

class Optimizer (object):
    def __init__(self):
        pass
    def optimize(self, cost, params):
        pass

################################
# Stochastic Gradient Descent  #
################################

class SGD (Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False):
        super(SGD,self).__init__()
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._nesterov = nesterov
    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = []
        for p, g in zip(params, grads):
            mparam_i = variable(zeros(p.get_value().shape))
            v = self._momentum * mparam_i - self._learning_rate * g
            updates.append((mparam_i, v))
            if (self._nesterov):
                updates.append((p, p + self._momentum * v - self._learning_rate * g ))
            else:
                updates.append((p, p + v))
        return updates

###########
# RMSProp #
###########

class RMSProp (Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        super(RMSProp,self).__init__()
        self._learning_rate = learning_rate
        self._rho = rho
        self._epsilon = epsilon
    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = OrderedDict()
        one = constant(1)
        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = variable(zeros(value.shape), dtype=value.dtype,
                                 broadcastable=param.broadcastable)
            accu_new = self._rho * accu + (one - self._rho) * grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (self._learning_rate * grad /
                                      sqrt(accu_new + self._epsilon))
        return updates

###########
# Adagrad #
###########

class Adagrad (Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-6):
        super(Adagrad,self).__init__()
        self._learning_rate = learning_rate
        self._epsilon = epsilon
    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = OrderedDict()
        one = constant(1)
        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            accu_new = accu + grad ** 2
            updates[accu] = accu_new
            updates[param] = param - ((self._learning_rate * grad) /
                                      (sqrt(accu_new) + self._epsilon))
        return updates

############
# AdaDELTA #
############

class AdaDELTA (Optimizer):
    def __init__(self, rho=0.9, epsilon=1e-6):
        super(AdaDELTA,self).__init__()
        self._rho = rho
        self._epsilon = epsilon
    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = OrderedDict()
        one = constant(1)
        for param, grad in zip(params, grads):
            value = param.get_value(borrow = True)
            accu = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            delta_accu = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            accu_new = self._rho * accu + (one - self._rho) * (grad ** 2 )
            update = - sqrt (delta_accu + self._epsilon) * grad / sqrt (accu_new + self._epsilon)
            delta_accu_new = self._rho * delta_accu + (one - self._rho) * (update ** 2)
            updates[accu] = accu_new
            updates[delta_accu] = delta_accu_new
            updates[param] = param + update
        return updates

########
# Adam #
########

class Adam (Optimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9,beta2=0.999, epsilon=1e-8):
        super(Adam,self).__init__()
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = OrderedDict()
        one = constant(1)
        t_prev = variable(0.)
        t = t_prev + 1
        a_t = self._learning_rate * sqrt(one - self._beta2**t)/(one - self._beta1**t)
        for param, grad in zip(params, grads):
            value = param.get_value(borrow = True)
            accu = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            moment = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            moment_new = self._beta1 * moment + (one - self._beta1) * grad 
            accu_new = self._beta2 * accu + (one - self._beta2) * (grad ** 2 )
            update = a_t * moment_new / (sqrt(accu_new) + self._epsilon)
            updates[moment] = moment_new
            updates[accu] = accu_new
            updates[param] = param - update
        updates[t_prev] = t
        return updates
