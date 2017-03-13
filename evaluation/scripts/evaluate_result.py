from evaluation.vqaTools import vqa
from evaluation.vqaTools import vqaEval
import json

ANNO_JSON="data/VQA_anno/mscoco_val2014_annotations.json"
QUES_JSON="data/VQA_anno/OpenEnded_mscoco_val2014_questions.json"
RES_JSON="experiments/Ours_MS/save_result_vqa_448_val2014/results/hop_01/vqa_OpenEnded_mscoco_val2014_LstmAttCtrlGradNoiseDontSelect448Pool501hop-40.00_results.json"

anno_json = json.load(open(ANNO_JSON, 'r'))
ques_json = json.load(open(QUES_JSON, 'r'))
res_json = json.load(open(RES_JSON, 'r'))

vqa_val = vqa.VQA(anno_json, ques_json)

res = vqa_val.loadRes(res_json, ques_json)
vqa_eval = vqaEval.VQAEval(vqa_val, res)

res_acc = vqa_eval.evaluate()

print vqa_eval.accuracy['overall']
print vqa_eval.accuracy['perAnswerType']
