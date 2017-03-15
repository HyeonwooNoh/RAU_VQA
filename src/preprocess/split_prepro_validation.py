import h5py
import json
import numpy as np
import os

from src.tools import utils

PREPRO_VALID_DIR="data/VQA_prepro/data_train_val"
JSON_FILE="data_prepro.json"
HDF5_FILE="data_prepro.h5"

VALID_1_ANNO="data/VQA_anno/mscoco_val-1-2014_annotations.json"
VALID_2_ANNO="data/VQA_anno/mscoco_val-2-2014_annotations.json"

PREPRO_VALID_1_DIR="data/VQA_prepro/data_train_val-1"
PREPRO_VALID_2_DIR="data/VQA_prepro/data_train_val-2"

def CopyHdf5Data(split_data, hdf5_data, key):
	split_data[key] = hdf5_data[key].value

def CopySubsetHdf5Data(split_data, hdf5_data, key, qid_to_index,
                       valid_question_ids):
	hdf5_value = hdf5_data[key].value
	split_data[key] = np.array([hdf5_value[qid_to_index[qid]] for qid \
		in valid_question_ids])

def SplitHdf5Data(hdf5_data, qid_to_index, valid_question_ids):
	split_data = {}

	# Copy training information
	copy_keys = ['answers', 'img_pos_train', 'ques_length_train',
                'ques_train', 'question_id_train']
	for key in copy_keys:
		CopyHdf5Data(split_data, hdf5_data, key)

	# Selectively add testing information
	copy_subset_keys = ['MC_ans_test', 'img_pos_test', 'ques_length_test',
                       'ques_length_test', 'ques_test', 'question_id_test']
	for key in copy_subset_keys:
		CopySubsetHdf5Data(split_data, hdf5_data, key, qid_to_index,
                         valid_question_ids)	

	return split_data	

def main():
	json_path = os.path.join(PREPRO_VALID_DIR, JSON_FILE)
	hdf5_path = os.path.join(PREPRO_VALID_DIR, HDF5_FILE)

	json_data = json.load(open(json_path, 'r'))
	print "Loading json data is done: {}".format(json_path)
	print "Keys for loaded json data:"
	print '\t-'+'\n\t-'.join(json_data.keys())

	hdf5_data = h5py.File(hdf5_path, 'r')
	print "Loading hdf5 data is done: {}".format(hdf5_path)
	print "Keys for loaded hdf5 data:"
	print '\t-'+'\n\t-'.join(hdf5_data.keys())

	print "Load valid 1 split annotations: {}".format(VALID_1_ANNO)
	valid_1_anno = json.load(open(VALID_1_ANNO, 'r'))
	print "Load valid 2 split annotations: {}".format(VALID_2_ANNO)
	valid_2_anno = json.load(open(VALID_2_ANNO, 'r'))

	valid_1_question_ids = [anno['question_id'] for anno in \
		valid_1_anno['annotations']]
	print "Number of valid 1 question ids: {}".format(len(valid_1_question_ids))
	valid_2_question_ids = [anno['question_id'] for anno in \
		valid_2_anno['annotations']]
	print "Number of valid 2 question ids: {}".format(len(valid_2_question_ids))

	qid_to_index = {qid:i for i, qid in enumerate(hdf5_data['question_id_test'])}

	valid_1_hdf5 = SplitHdf5Data(hdf5_data, qid_to_index, valid_1_question_ids)
	print "Spliting hdf5 data for valid 1 is done."
	print "Keys for splited valid 1 hdf5 data:"
	print '\t-'+'\n\t-'.join(valid_1_hdf5.keys())

	valid_2_hdf5 = SplitHdf5Data(hdf5_data, qid_to_index, valid_2_question_ids)	
	print "Spliting hdf5 data for valid 2 is done."
	print "Keys for splited valid 2 hdf5 data:"
	print '\t-'+'\n\t-'.join(valid_2_hdf5.keys())

	# Save valid 1 split.
	utils.CheckAndCreateDir(PREPRO_VALID_1_DIR)
	valid_1_hdf5_path = os.path.join(PREPRO_VALID_1_DIR, HDF5_FILE)
	utils.SaveHdf5Data(valid_1_hdf5_path, valid_1_hdf5)
	print "Valid 1 hdf5 data is saved in: {}".format(valid_1_hdf5_path)

	valid_1_json_path = os.path.join(PREPRO_VALID_1_DIR, JSON_FILE)
	json.dump(json_data, open(valid_1_json_path, 'w'))
	print "Original json data is copied as valid 1 json: {}".format(
		valid_1_json_path)

	# Save valid 2 split.
	utils.CheckAndCreateDir(PREPRO_VALID_2_DIR)
	valid_2_hdf5_path = os.path.join(PREPRO_VALID_2_DIR, HDF5_FILE)
	utils.SaveHdf5Data(valid_2_hdf5_path, valid_2_hdf5)
	print "Valid 2 hdf5 data is saved in: {}".format(valid_2_hdf5_path)

	valid_2_json_path = os.path.join(PREPRO_VALID_2_DIR, JSON_FILE)
	json.dump(json_data, open(valid_2_json_path, 'w'))
	print "Original json data is copied as valid 2 json: {}".format(
		valid_2_json_path)

	
if __name__ == "__main__":
	main()
