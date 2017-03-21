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
_VALID_1_QUESTIONS="data/VQA_anno/OpenEnded_mscoco_val-1-2014_questions.json"
_VALID_2_QUESTIONS="data/VQA_anno/OpenEnded_mscoco_val-2-2014_questions.json"

def GetSelectedStep(oracle_selection, selection_option='best_shortest'):
	if selection_option == 'best_shortest':
		qid_to_selected_step = {select['question_id']:select['best_shortest'] \
			for select in oracle_selection}
	else:
		raise ValueError('Unknown selection option')
	return qid_to_selected_step


def PerformStepSelection(res_jsons_per_step, qid_to_selected_step):
	qid_to_res_json_array = {}
	for res_json in res_jsons_per_step:
		for entry in res_json:
			qid = entry['question_id']
			if not qid_to_res_json_array.has_key(qid):
				qid_to_res_json_array[qid] = []
			qid_to_res_json_array[qid].append(entry)
	# Perform step selection
	selected_res_json = [qid_to_res_json_array[qid][step-1] for qid, step in \
		qid_to_selected_step.iteritems()]
	return selected_res_json

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
	parser.add_argument("--meta_controller_epoch", default=10, type=int,
		help="Number of epochs for training meta controller")
	parser.add_argument("--algorithm_name",
		default="LstmAttCtrlGradNoiseDontSelect448Pool5",
		help="Algorithm name for target model")
	args = parser.parse_args()
	return args

def main():
	args = _GetArgs()
	params = vars(args) # conver to ordinary dict
	print "Parsed input parameters:"
	print json.dumps(params, indent=4)

	print "Increment num_steps by 4. as additional 4 steps contains:"
	print "1) ensemble, 2) dopred selection, 3) hard_selection, 4)soft_selection"
	original_num_steps = params['num_steps']
	params['num_steps'] = params['num_steps'] + 4

	print "Load annotation and questions for validation 2 set."
	anno_json = json.load(open(_VALID_2_ANNO, 'r'))
	ques_json = json.load(open(_VALID_2_QUESTIONS, 'r'))

	vqa_data = vqa.VQA(anno_json, ques_json)

	res_jsons_per_step = oracle_utils.LoadResultJsonMultipleSteps(params)
	res_jsons_per_step = [oracle_utils.FilterResultsByQuestions(res_json,
		ques_json) for res_json in res_jsons_per_step]

	oracle_selection_results = {}
	eval_results_per_step = oracle_utils.EvaluateMultipleSteps(
		res_jsons_per_step, vqa_data, ques_json)
	oracle_selection_results["ensemble"] = eval_results_per_step[original_num_steps]
	oracle_selection_results["do_pred_selection"] = eval_results_per_step[original_num_steps+1]
	oracle_selection_results["hard_selection"] = eval_results_per_step[original_num_steps+2]
	oracle_selection_results["soft_selection"] = eval_results_per_step[original_num_steps+3]
	eval_results_per_step = eval_results_per_step[:original_num_steps]
	for i, eval_result in enumerate(eval_results_per_step):
		oracle_selection_results["step_{}".format(i+1)] = eval_result

	# Compute oracle best shortest prediction
	acc_per_step_per_question = oracle_utils.GetAccPerStepPerQuestion(
		eval_results_per_step)
	is_best_per_question, best_shortest_per_question,\
		has_correct_answer_per_question, is_correct_answer_per_question = \
		oracle_utils.GetBestStepLabelsPerQuestion(acc_per_step_per_question)
	qid_to_oracle_best_shortest_step = {qid: index+1 for qid, index in \
		best_shortest_per_question.iteritems()}
	oracle_res_json = PerformStepSelection(res_jsons_per_step,
                                          qid_to_oracle_best_shortest_step)
	oracle_selection_results['oracle_best_shortest'] = \
		oracle_utils.EvaluateResult(oracle_res_json, vqa_data, ques_json)

	# Output json path
	output_json_path = oracle_utils.GetComprehensiveResultJsonFilePath(params)
	utils.CheckAndCreateDir(os.path.dirname(output_json_path))
	json.dump(oracle_selection_results, open(output_json_path, 'w'))

	# Print results
	print "{},\toverall,\tyes/no,\tnumber,\tother".format('name'.ljust(20))
	for name, result in oracle_selection_results.iteritems():
		ansType = result['perAnswerType']
		print "{},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f}".format(
			name.ljust(20), result['overall'], ansType['yes/no'], ansType['number'],
			ansType['other'])

if __name__ == "__main__":
	main()
