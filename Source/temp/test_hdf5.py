import h5py
import numpy as np
import datetime

# Layer 1
W11 = np.asarray(range(36)).reshape((3,4,3))
W12 = np.asarray(range(36)).reshape((4,3,3))

print (W11)
print (W12)

# Layer 2
W21 = np.asarray(range(36)).reshape((6,6))
W22 = np.asarray(range(36))

print (W21)
print (W22)

# Create the file
filename = "myhdf5.h5"
f = h5py.File(filename, "w")
f.attrs['version'] = 'Version de Test'
f.attrs['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
f.attrs['HDF5_Version'] = h5py.version.hdf5_version
f.attrs['h5py_version'] = h5py.version.version

# Create the group with the layer 1
# In order to set the compression     g1.create_dataset('dataset1', data = m1, compression="gzip", compression_opts=9)
layer1 = f.create_group("L1")
layer1.attrs['Type'] = 'Dense'
layer1.create_dataset('W11', data=W11)
layer1.create_dataset('W12',  data=W12, compression="gzip", compression_opts=9)

# Create the group with the layer 2
layer2 = f.create_group("L2")
layer1.attrs['Type'] = 'Softmax'
layer2.create_dataset('W21', data=W21, compression="gzip", compression_opts=9)
layer2.create_dataset('W22',  data=W22)

print ('Saved')

f.close()   # be CERTAIN to close the file

# Open the file
with h5py.File(filename,'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    for layer in hf.keys():
        data = hf.get(layer)
        print('List of arrays in this data set: \n', data.keys())
        for weights in data.keys():
            data2 = data.get(weights)
            np_data = np.array(data2)
            print('Shape of the array ' + weights + ': \n', data2.shape)
            print (np_data)




