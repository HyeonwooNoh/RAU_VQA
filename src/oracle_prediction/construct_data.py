import argparse
import h5py
import json
import numpy as np
import os

from evaluation.vqaTools import vqa
from evaluation.vqaTools import vqaEval

from src.tools import utils
from src.oracle_prediction import utils as oracle_utils

_VALID_1_ANNO="data/VQA_anno/mscoco_val-1-2014_annotations.json"
_VALID_2_ANNO="data/VQA_anno/mscoco_val-2-2014_annotations.json"

_PREPRO_VALID_1_DIR="data/VQA_prepro/data_train_val-1"
_PREPRO_VALID_2_DIR="data/VQA_prepro/data_train_val-2"

_JSON_FILE="data_prepro.json"
_HDF5_FILE="data_prepro.h5"
	
def _GetArgs():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--result_dir",
		default="experiments/Ours_MS/save_result_vqa_448_val2014",
		help="Directory containing results for multiple hops")
	parser.add_argument("--task_type", default=oracle_utils.TaskType.OPEN_ENDED,
		help="Task type: [{}|{}]".format(
		oracle_utils.TaskType.OPEN_ENDED, oracle_utils.TaskType.MULTIPLE_CHOICE))
	parser.add_argument("--num_steps", default=8,
		help="Number of steps used for the target recurrent answering units")
	parser.add_argument("--num_epochs", default=40,
		help="Number of epochs for trained models")
	parser.add_argument("--algorithm_name",
		default="LstmAttCtrlGradNoiseDontSelect448Pool5",
		help="Algorithm name for target model")
	parser.add_argument("--annotation_json",
		default="data/VQA_anno/mscoco_val2014_annotations.json",
		help="Annotation json file")
	parser.add_argument("--question_json",
		default="data/VQA_anno/OpenEnded_mscoco_val2014_questions.json",
		help="Question json file")
	args = parser.parse_args()
	return args

def ConstructBestStepTrainingData(is_best_per_question,
                                  best_shortest_per_question,
                                  has_correct_answer_per_question):
	valid_1_hdf5_path = os.path.join(_PREPRO_VALID_1_DIR, _HDF5_FILE)
	valid_1_hdf5 = utils.LoadHdf5Data(valid_1_hdf5_path)
	print "Hdf5 for valid 1 is loaded from {}:".format(valid_1_hdf5_path)

	valid_2_hdf5_path = os.path.join(_PREPRO_VALID_2_DIR, _HDF5_FILE)
	valid_2_hdf5 = utils.LoadHdf5Data(valid_2_hdf5_path)
	print "Hdf5 for valid 2 is loaded from {}:".format(valid_2_hdf5_path)

	best_step_train_hdf5 = {}
	# Valid 1 becomes training split
	best_step_train_hdf5['img_pos_train'] = valid_1_hdf5['img_pos_test']
	best_step_train_hdf5['ques_length_train'] = valid_1_hdf5['ques_length_test']
	best_step_train_hdf5['ques_train'] = valid_1_hdf5['ques_test']
	best_step_train_hdf5['question_id_train'] = valid_1_hdf5['question_id_test']

	best_step_train_hdf5['is_best_train'] = np.array([is_best_per_question[qid] \
		for qid in best_step_train_hdf5['question_id_train']])
	best_step_train_hdf5['best_shortest_train'] = np.array(
		[best_shortest_per_question[qid] for qid in \
		best_step_train_hdf5['question_id_train']]) + 1
	best_step_train_hdf5['has_correct_answer_train'] = np.array(
		[has_correct_answer_per_question[qid] for qid in \
		best_step_train_hdf5['question_id_train']])

	# Valid 2 becomes test split
	best_step_train_hdf5['img_pos_test'] = valid_2_hdf5['img_pos_test']
	best_step_train_hdf5['ques_length_test'] = valid_2_hdf5['ques_length_test']
	best_step_train_hdf5['ques_test'] = valid_2_hdf5['ques_test']
	best_step_train_hdf5['question_id_test'] = valid_2_hdf5['question_id_test']

	best_step_train_hdf5['is_best_test'] = np.array([is_best_per_question[qid] \
		for qid in best_step_train_hdf5['question_id_test']])
	best_step_train_hdf5['best_shortest_test'] = np.array(
		[best_shortest_per_question[qid] for qid in \
		best_step_train_hdf5['question_id_test']]) + 1
	best_step_train_hdf5['has_correct_answer_test'] = np.array(
		[has_correct_answer_per_question[qid] for qid in \
		best_step_train_hdf5['question_id_test']])

	valid_1_json_path = os.path.join(_PREPRO_VALID_1_DIR, _JSON_FILE)
	valid_json = json.load(open(valid_1_json_path, 'r'))
	print "json for valid is loaded from {}:".format(valid_1_json_path)

	best_step_train_json = {}
	best_step_train_json['ix_to_word'] = valid_json['ix_to_word']
	best_step_train_json['unique_img_train'] = valid_json['unique_img_test']
	best_step_train_json['unique_img_test'] = valid_json['unique_img_test']

	return best_step_train_hdf5, best_step_train_json

def main():
	args = _GetArgs()
	params = vars(args) # conver to ordinary dict
	print "Parsed input parameters:"
	print json.dumps(params, indent=4)

	print "Load annotation and questions for validation set."
	anno_json = json.load(open(params['annotation_json'], 'r'))
	ques_json = json.load(open(params['question_json'], 'r'))

	vqa_data = vqa.VQA(anno_json, ques_json)

	print "Compute accuracy per step per question."
	res_jsons_per_step = oracle_utils.LoadResultJsonMultipleSteps(params)
	eval_results_per_step = oracle_utils.EvaluateMultipleSteps(
		res_jsons_per_step, vqa_data, ques_json)
	acc_per_step_per_question = oracle_utils.GetAccPerStepPerQuestion(
		eval_results_per_step)

	print "Compute best step labels per question."
	is_best_per_question, best_shortest_per_question,\
		has_correct_answer_per_question = \
		oracle_utils.GetBestStepLabelsPerQuestion(acc_per_step_per_question)

	best_step_train_hdf5, best_step_train_json = ConstructBestStepTrainingData(
		is_best_per_question, best_shortest_per_question,
		has_correct_answer_per_question)

	oracle_data_dir = oracle_utils.GetOraclePredictionDataDir(params)
	utils.CheckAndCreateDir(oracle_data_dir)

	oracle_data_hdf5_path = os.path.join(oracle_data_dir, _HDF5_FILE)
	utils.SaveHdf5Data(oracle_data_hdf5_path, best_step_train_hdf5)
	print "Processed hdf5 oracle training data is saved in: {}".format(
		oracle_data_hdf5_path)

	oracle_data_json_path = os.path.join(oracle_data_dir, _JSON_FILE)
	json.dump(best_step_train_json, open(oracle_data_json_path, 'w'))
	print "Processed json oracle training data is saved in: {}".format(
		oracle_data_json_path)


if __name__ == "__main__":
	main()
