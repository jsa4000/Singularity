########################
# Base Exception Class #
########################

class singularity_exception(Exception):
    #constructor 
    def __init__(self, value):
         self.value = value
    ################################################
    #overrideglobal
    def __str__(self):
         return str(self.value)

#######################
# Standard Exceptions #
#######################

backend_not_implemented_exception = singularity_exception('Backend not implemented')
not_implemented_exception = singularity_exception('Not implemented')