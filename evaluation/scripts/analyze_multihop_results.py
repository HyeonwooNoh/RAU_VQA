from evaluation.vqaTools import vqa
from evaluation.vqaTools import vqaEval
import json

ANNO_JSON="data/VQA_anno/mscoco_val2014_annotations.json"
QUES_JSON="data/VQA_anno/OpenEnded_mscoco_val2014_questions.json"

eval_option = 'MS'

if eval_option == 'MS':
	res_json_format = "experiments/Ours_MS/save_result_vqa_448_val2014/results/hop_{0:02d}/vqa_OpenEnded_mscoco_val2014_LstmAttCtrlGradNoiseDontSelect448Pool5{0:02d}hop-{1:.2f}_results.json"
	save_path="MS_results.json"
elif eval_option == 'Full':
	res_json_format = "experiments/Ours_Full/save_result_vqa_val2014/results/hop_{0:02d}/vqa_OpenEnded_mscoco_val2014_LstmAttCtrlGradNoiseDontSelect448Pool5{0:02d}hop-{1:.2f}_results.json"
	save_path="Full_results.json"
else:
	raise ValueError('Unknown eval_option')

anno_json = json.load(open(ANNO_JSON, 'r'))
ques_json = json.load(open(QUES_JSON, 'r'))
question_dict = {q['question_id']:q['question'] for q in ques_json['questions']}

vqa_data = vqa.VQA(anno_json, ques_json)

nepoch = 40
nhop = 8
print "Start analyzing multi-hop results. # epoch: {}, # hop: {}".format(nepoch, nhop)

res_jsons = {}
performance_per_hop = []
score_per_hops_per_question = {}
for h in range(nhop):
	print "# {} hop".format(h+1)
	#res_json = json.load(open(res_json_format.format(h+9, nepoch), 'r'))
	res_json = json.load(open(res_json_format.format(h+1, nepoch), 'r'))
	for res_entry in res_json:
		if not res_jsons.has_key(res_entry['question_id']):
			res_jsons[res_entry['question_id']] = []
		res_jsons[res_entry['question_id']].append(res_entry)
	res = vqa_data.loadRes(res_json, ques_json)
	vqa_eval = vqaEval.VQAEval(vqa_data, res)
	res_acc = vqa_eval.evaluate()
	performance_per_hop.append({'overall': vqa_eval.accuracy['overall'],
		'perAnswerType': vqa_eval.accuracy['perAnswerType']})
	for ques, acc in res_acc.iteritems():
		if not score_per_hops_per_question.has_key(ques):
			score_per_hops_per_question[ques] = []
		score_per_hops_per_question[ques].append(acc)
	
best_answer_hist = [0] * nhop
best_answer_soft_hist = [0.0] * nhop
unique_best_answer_hist = [0] * nhop
unique_best_question_list = {}
for h in range(nhop):
	unique_best_question_list[h+1] = []
unique_best_question_type_hist = {}
for h in range(nhop):
	unique_best_question_type_hist[h+1] = {}
oracle_acc = {}
oracle_res_json = []
for ques, acc_list in score_per_hops_per_question.iteritems():
	max_val = max(acc_list)
	max_index = acc_list.index(max_val)
	if max_val > 0.0:
		is_best = [int(k == max_val) for k in acc_list]
		best_answer_hist = [k[0]+k[1] for k in zip(best_answer_hist, is_best)]
		best_answer_soft_hist = [k[0] + float(k[1]) / float(sum(is_best)) for k in zip(best_answer_soft_hist, is_best)]
		if sum(is_best) == 1:
			unique_best_answer_hist = [k[0]+k[1] for k in zip(unique_best_answer_hist, is_best)]
			unique_best_question_list[max_index+1].append(
				question_dict[res_jsons[ques][max_index]['question_id']])
			ques_type = res_jsons[ques][max_index]['question_type']
			unique_best_question_type_hist[max_index+1][ques_type] = \
				unique_best_question_type_hist[max_index+1].get(ques_type, 0) + 1
	oracle_acc[ques] = max_val
	oracle_res_json.append(res_jsons[ques][max_index])

print "Evaluating oracle performance"
oracle_res = vqa_data.loadRes(oracle_res_json, ques_json)
oracle_vqa_eval = vqaEval.VQAEval(vqa_data, oracle_res)
oracle_res_acc = oracle_vqa_eval.evaluate()
oracle_performance = {'overall': oracle_vqa_eval.accuracy['overall'],
	'perAnswerType': oracle_vqa_eval.accuracy['perAnswerType']}

print "Evaluating ensemble performance"
ensemble_res_json = json.load(open(res_json_format.format(nhop+1, nepoch), 'r'))
ensemble_res = vqa_data.loadRes(ensemble_res_json, ques_json)
ensemble_vqa_eval = vqaEval.VQAEval(vqa_data, ensemble_res)
ensemble_res_acc = ensemble_vqa_eval.evaluate()
ensemble_performance = {'overall': ensemble_vqa_eval.accuracy['overall'],
	'perAnswerType': ensemble_vqa_eval.accuracy['perAnswerType']}

analysis_results = {}
analysis_results['ensemble_performance'] = ensemble_performance
analysis_results['ensemble_res_acc'] = ensemble_res_acc
analysis_results['oracle_performance'] = oracle_performance
analysis_results['oracle_res_acc'] = oracle_res_acc
analysis_results['best_answer_hist'] = best_answer_hist
analysis_results['best_answer_soft_hist'] = best_answer_soft_hist
analysis_results['unique_best_answer_hist'] = unique_best_answer_hist
analysis_results['unique_best_question_list'] = unique_best_question_list
analysis_results['unique_best_question_type_hist'] = unique_best_question_type_hist
analysis_results['performance_per_hop'] = performance_per_hop
analysis_results['score_per_hops_per_question'] = score_per_hops_per_question
json.dump(analysis_results, open(save_path, 'w'))
"""
"""
