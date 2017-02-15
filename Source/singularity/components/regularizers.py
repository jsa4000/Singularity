from ..core import *

##################
# L1 Regularizer #
##################

def L1(lambda_p = 0.001):
    def wrap(params):
        return (sum(list(abs(param).sum() for param in params))) * lambda_p
    return wrap

##################
# L2 Regularizer #
##################

def L2(lambda_p = 0.0001):
    def wrap(params):
        return sum(list(power(param,2).sum() for param in params)) * lambda_p
    return wrap
