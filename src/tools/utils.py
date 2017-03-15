import json
import h5py
import os

def CheckAndCreateDir(dir_path):
	if not os.path.isdir(dir_path):
		print "Directory doesn't exist, create one: {}".format(dir_path)
		os.makedirs(dir_path)

def LoadHdf5Data(hdf5_path):
	loaded_data = {}
	hdf5_data = h5py.File(hdf5_path, 'r')
	for key in hdf5_data.keys():
		loaded_data[key] = hdf5_data[key].value
	return loaded_data

def SaveHdf5Data(save_path, hdf5_data):	
	with h5py.File(save_path, 'w') as f:
		for key, numpy_data in hdf5_data.iteritems():
			f.create_dataset(key, dtype=numpy_data.dtype, data=numpy_data)
		f.close()
