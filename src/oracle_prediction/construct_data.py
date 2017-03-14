import argparse
import h5py
import json
import numpy as np
import os

from evaluation.vqaTools import vqa
from evaluation.vqaTools import vqaEval

from src.tools import utils

_VALID_1_ANNO="data/VQA_anno/mscoco_val-1-2014_annotations.json"
_VALID_2_ANNO="data/VQA_anno/mscoco_val-2-2014_annotations.json"

_PREPRO_VALID_1_DIR="data/VQA_prepro/data_train_val-1"
_PREPRO_VALID_2_DIR="data/VQA_prepro/data_train_val-2"

_JSON_FILE="data_prepro.json"
_HDF5_FILE="data_prepro.h5"

class TaskType():
	OPEN_ENDED = "OpenEnded"
	MULTIPLE_CHOICE = "MultipleChoice"

def GetResultJsonFilePath(params, step):
	return os.path.join(params['result_dir'], 'results',\
		"hop_{0:02d}/vqa_{1}_mscoco_val2014_{2}{0:02d}hop-{3:.2f}_results.json"\
		.format(step, params['task_type'], params['algorithm_name'],\
		params['num_epochs']))

def GetOraclePredictionDataDir(params):
	subdirectory_name = "{0}_{1:2d}_steps_{2:2d}_epochs".format(
		params['task_type'], params['num_steps'], params['num_epochs'])
	return os.path.join(params['result_dir'], 'oracle_prediction',\
		subdirectory_name)
	
def _GetArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("--result_dir",
		default="experiments/Ours_MS/save_result_vqa_448_val2014",
		help="Directory containing results for multiple hops")
	parser.add_argument("--task_type", default=TaskType.OPEN_ENDED,
		help="Task type: [{}|{}]".format(
		TaskType.OPEN_ENDED, TaskType.MULTIPLE_CHOICE))
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

def GetAccPerStepPerQuestion(params, vqa_data, ques_json):
	acc_per_step_per_question = {}
	for step in range(1, params['num_steps']+1):
		# Load result file
		res_json_path = GetResultJsonFilePath(params, step)
		res_json = json.load(open(res_json_path, 'r'))
		# Compute accuracy
		res = vqa_data.loadRes(res_json, ques_json)
		vqa_eval = vqaEval.VQAEval(vqa_data, res)
		res_acc = vqa_eval.evaluate()
		# Add accuracy per steps
		for ques, acc in res_acc.iteritems():
			if not acc_per_step_per_question.has_key(ques):
				acc_per_step_per_question[ques] = []	
			acc_per_step_per_question[ques].append(acc)
	return acc_per_step_per_question

def GetBestStepLabelsPerQuestion(acc_per_step_per_question):
	is_best_per_question = {}
	best_shortest_per_question = {}
	has_correct_answer_per_question = {}
	for ques, acc_list in acc_per_step_per_question.iteritems():
		max_acc = max(acc_list)
		is_best = [int(acc == max_acc) for acc in acc_list]
		is_best_per_question[ques] = is_best
		best_shortest_per_question[ques] = acc_list.index(max_acc)
		has_correct_answer_per_question[ques] = int(max_acc > 0)
	return is_best_per_question, best_shortest_per_question, \
		has_correct_answer_per_question

def LoadHdf5File(hdf5_path):
	loaded_data = {}
	hdf5_data = h5py.File(hdf5_path, 'r')
	for key in hdf5_data.keys():
		loaded_data[key] = hdf5_data[key].value
	return loaded_data
	

def ConstructBestStepTrainingData(is_best_per_question,
                                  best_shortest_per_question,
                                  has_correct_answer_per_question):
	valid_1_hdf5_path = os.path.join(_PREPRO_VALID_1_DIR, _HDF5_FILE)
	valid_1_hdf5 = LoadHdf5File(valid_1_hdf5_path)
	print "Hdf5 for valid 1 is loaded from {}:".format(valid_1_hdf5_path)

	valid_2_hdf5_path = os.path.join(_PREPRO_VALID_2_DIR, _HDF5_FILE)
	valid_2_hdf5 = LoadHdf5File(valid_2_hdf5_path)
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
	question_dict = {q['question_id']:q['question'] for q in \
		ques_json['questions']}

	vqa_data = vqa.VQA(anno_json, ques_json)

	print "Compute accuracy per step per question."
	acc_per_step_per_question = GetAccPerStepPerQuestion(params, vqa_data,
                                                        ques_json)
	print "Compute best step labels per question."
	is_best_per_question, best_shortest_per_question,\
	has_correct_answer_per_question = \
		GetBestStepLabelsPerQuestion(acc_per_step_per_question)

	best_step_train_hdf5, best_step_train_json = ConstructBestStepTrainingData(
		is_best_per_question, best_shortest_per_question,
		has_correct_answer_per_question)

	oracle_data_dir = GetOraclePredictionDataDir(params)
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
