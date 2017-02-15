from ..core import *
from .pickle import *
import h5py
import datetime

#######################
# Serialize Functions #
#######################

_SERIALIZED_PREFIX = "_S%"
_SERIALIZED_SUFFIX = "%S_"

def _get_deserialized_data(data):
    if (_is_serialized(data)):
        return deserialize(data[len(_SERIALIZED_PREFIX):-(len(_SERIALIZED_SUFFIX))])
    else:
        return data

def _get_serialized_data(data):
    return _SERIALIZED_PREFIX + serialize(data) + _SERIALIZED_SUFFIX

def _is_serialized(data):
    if (isinstance(data, str) and data.startswith(_SERIALIZED_PREFIX) and data.endswith(_SERIALIZED_SUFFIX)):
        return True
    else:
        return False

##################
# HDF5 Functions #
##################

_TYPE_ATTR = "__type__"
_CLASS_ATTR = "__class__"

def save(filename, data, rootname = "settings"):
    def _save(name, data, hdf5):
        if (not is_none_or_empty(name)):
            if (is_collection(data) and is_complex(data)):
                group = hdf5.create_group(name)
                group.attrs[_TYPE_ATTR] = type(data).__name__
                if (isinstance(data, (tuple, list))):
                    for index, item in enumerate(data):
                        _save(str(index), item, group)
                else:
                    for key, item in data.iteritems():
                        _save(key, item, group)
            else:
                if (isinstance(data,BaseClass)):
                    group = hdf5.create_group(name)
                    group.attrs[_TYPE_ATTR] = BaseClass.__name__
                    group.attrs[_CLASS_ATTR] = _get_serialized_data(type(data))
                    for key, item in data.attrs.iteritems():
                        _save(key, item, group)
                elif (not is_complex(data) or is_variable(data)):
                    if (is_variable(data)):
                        data = get_value(data)
                    if (is_collection(data)):
                        try:
                            hdf5.create_dataset(name, data=data, compression="gzip", compression_opts=9)
                        except:
                            hdf5.attrs[name] = _get_serialized_data(data)
                    else:
                        hdf5.attrs[name] = data
                else:
                    try:
                        hdf5.attrs[name] = _get_serialized_data(data)
                    except:
                        hdf5.attrs[name] = type(data).__name__
    file = h5py.File(filename, "w")
    _save(rootname,data, file)
    file.close()  
    
def load(filename):
    def _set_data(data, key, item):
        if (isinstance(data, (list))):
            data.append(item)
        elif (isinstance(data, (tuple))):
            data = list(data)
            data.append(item)
            data = tuple(data)
        elif (isinstance(data,dict)):
            data[key] = item
        elif (isinstance(data,BaseClass)):
            data.set_attr(key,item)
        else:
            data = item
        return data
    def _load(hdf5):
        foo = None
        if _TYPE_ATTR in hdf5.attrs:
            type = hdf5.attrs[_TYPE_ATTR]
            foo = eval(type + "()")
        for key in hdf5.attrs:
            if (key != _TYPE_ATTR):
                attr = hdf5.attrs[key]
                if (_is_serialized(attr)):
                    attr = _get_deserialized_data(attr)
                foo = _set_data(foo,key,attr)
        for key in hdf5.keys():
            item = hdf5.get(key)
            if (isinstance(item, h5py.Dataset)):
                foo = _set_data(foo,key,np.array(item))
            else:
                foo = _set_data(foo,key,_load(item))
        if (isinstance(foo,BaseClass)):
            derived_class = _get_deserialized_data(foo.attrs[_CLASS_ATTR])
            foo = BaseClass(**foo.attrs)
            cast(foo, derived_class)
        return foo
    file = h5py.File(filename, "r")
    data = _load(file)
    file.close()
    return data

