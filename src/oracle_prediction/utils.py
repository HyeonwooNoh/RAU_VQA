import json
import os

from evaluation.vqaTools import vqa
from evaluation.vqaTools import vqaEval

class TaskType():
	OPEN_ENDED = "OpenEnded"
	MULTIPLE_CHOICE = "MultipleChoice"

def GetResultJsonFilePath(params, step):
	return os.path.join(params['result_dir'], 'results',\
		"hop_{0:02d}/vqa_{1}_mscoco_val2014_{2}{0:02d}hop-{3:.2f}_results.json"\
		.format(step, params['task_type'], params['algorithm_name'],\
		params['num_epochs']))

def GetOraclePredictionDataDir(params):
	subdirectory_name = "{0}_{1:02d}_steps_{2:02d}_epochs".format(
		params['task_type'], params['num_steps'], params['num_epochs'])
	return os.path.join(params['result_dir'], 'oracle_prediction',\
		subdirectory_name)

def GetOracleSelectionJsonFilePath(params):
	oracle_prediction_data_dir = GetOraclePredictionDataDir(params)
	return os.path.join(oracle_prediction_data_dir, 'meta_controller_best_shortest',\
		'results', "oracle_selection_{:.2f}_epoch.json".format(\
		params['meta_controller_epoch']))

def GetStepSelectionResultJsonFilePath(params):
	oracle_prediction_data_dir = GetOraclePredictionDataDir(params)
	return os.path.join(oracle_prediction_data_dir, 'meta_controller_best_shortest',\
		'selection_results', "step_selection_{:.2f}_epoch.json".format(\
		params['meta_controller_epoch']))

def GetComprehensiveResultDir(params):
	return os.path.join(params['result_dir'], 'comprehensive_result')

def GetComprehensiveResultJsonFilePath(params):
	subdir = GetComprehensiveResultDir(params)
	return os.path.join(subdir, "comprehensive_result_{:.2f}_epoch.json".format(
		params['meta_controller_epoch']))
	

def LoadOracleSelectionData(params):
	json_path = GetOracleSelectionJsonFilePath(params)
	json_data = json.load(open(json_path, 'r'))
	return json_data

""" Filter results by question.

Filter results by question. The main assumption is that the result is a subset
of the questions.
"""
def FilterResultsByQuestions(res_json, ques_json):
	qid_to_res_json = {entry['question_id']:entry for entry in res_json}
	question_ids = [entry['question_id'] for entry in ques_json['questions']]
	filtered_res_json = [qid_to_res_json[qid] for qid in question_ids]
	return filtered_res_json

def LoadResultJsonMultipleSteps(params):
	res_jsons_per_step = []
	for step in range(1, params['num_steps']+1):
		# Load result file
		res_json_path = GetResultJsonFilePath(params, step)
		res_json = json.load(open(res_json_path, 'r'))
		res_jsons_per_step.append(res_json)
	return res_jsons_per_step

def EvaluateResult(res_json, vqa_data, ques_json):
	res = vqa_data.loadRes(res_json, ques_json)
	vqa_eval = vqaEval.VQAEval(vqa_data, res)
	acc_per_qid = vqa_eval.evaluate()
	eval_result = {}
	eval_result['overall'] = vqa_eval.accuracy['overall']
	eval_result['perAnswerType'] = vqa_eval.accuracy['perAnswerType']
	eval_result['perQuestionType'] = vqa_eval.accuracy['perQuestionType']
	eval_result['perQuestionId'] = acc_per_qid
	return eval_result

def EvaluateMultipleSteps(res_jsons_per_step, vqa_data, ques_json):
	eval_results_per_step = []
	for i, res_json  in enumerate(res_jsons_per_step):
		eval_result = EvaluateResult(res_json, vqa_data, ques_json)
		eval_results_per_step.append(eval_result)
	return eval_results_per_step

def GetAccPerStepPerQuestion(eval_results_per_step):
	acc_per_step_per_question = {}
	for eval_result in eval_results_per_step:
		res_acc = eval_result['perQuestionId']
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
	
