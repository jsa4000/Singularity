import theano
from theano import tensor as T


######################
"""
 This example is to demotrate that "y" functions
 doesn't need to be initiliazied as symbolic value 
 like inputs (x) or shared variables.
"""
###################### 

def operation (x, layer):
    result = 1
    for i in range(layer):
        result *= x
    return result

layer = 6
x = T.scalar()
#y = T.scalar() # Not need to be initialized lile x
y = operation (x, layer)

myfunc = theano.function ([x], y)
print (myfunc(2))


######################
"""
This example is to see how functions works
in sybolic way. If this is true then the inputs
nodes can create symbolic values for theirself and
using withouf be passed by parameters when propaging the
function.
"""
###################### 


x = T.scalar()    
z = T.scalar() 
 
def computeX():
    return x 

def computeZ():
    return z 

def computeXZ():
    return computeX() + computeZ()

y = computeXZ()

myfunc = theano.function([x,z], y)

value1 = 3
value2 = 5

print (myfunc(3,5))





