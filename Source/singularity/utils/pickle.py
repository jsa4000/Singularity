import sys
import gzip
from six.moves import cPickle

#####################
# cPickle Functions #
#####################

def save(filename, data, protocol = 0, compress = False):
    if (compress):
        with gzip.GzipFile(filename + ".gz", 'wb') as f:
            f.write(cPickle.dumps(data, protocol))
    else:
        with open(filename, 'wb') as f:
            cPickle.dump(data, f)

def load(filename):
    if filename.endswith(".gz"):
        f = gzip.open(filename, 'rb')
    else:
        f = open(filename, 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding="bytes")
    f.close()
    return data  

def serialize(data, protocol = 0):
  return cPickle.dumps(data, protocol)

def deserialize(data):
  return cPickle.loads(data)

