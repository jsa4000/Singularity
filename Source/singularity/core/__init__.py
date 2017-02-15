from .exceptions import *
from .commons import *

backends = ('theano', 'numpy')
_backend = backends[0]

if (_backend == 'theano'):
    print ("Theano backend initialized.")
    from .theano_backend import *
elif (_backend == 'numpy'):
    print ("Numpy backend initialized.")
    from .numpy_backend import *
else:
    raise exceptions.backend_not_implemented_exception

# Delay the load unitl loading the backend
from .initializers import *