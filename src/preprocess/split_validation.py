import copy
import json
import random
from tqdm import tqdm

VALID_ANNO="data/VQA_anno/mscoco_val2014_annotations.json"
VALID_1_ANNO="data/VQA_anno/mscoco_val-1-2014_annotations.json"
VALID_2_ANNO="data/VQA_anno/mscoco_val-2-2014_annotations.json"

VALID_QUES_OPENENDED = "data/VQA_anno/OpenEnded_mscoco_val2014_questions.json"
VALID_1_QUES_OPENENDED = "data/VQA_anno/OpenEnded_mscoco_val-1-2014_questions.json"
VALID_2_QUES_OPENENDED = "data/VQA_anno/OpenEnded_mscoco_val-2-2014_questions.json"

VALID_QUES_MULTIPLECHOICE = "data/VQA_anno/MultipleChoice_mscoco_val2014_questions.json"
VALID_1_QUES_MULTIPLECHOICE = "data/VQA_anno/MultipleChoice_mscoco_val-1-2014_questions.json"
VALID_2_QUES_MULTIPLECHOICE = "data/VQA_anno/MultipleChoice_mscoco_val-2-2014_questions.json"

def CheckAndDeepCopy(parent_dict, child_dict, key):
	if parent_dict.has_key(key):
		child_dict[key] = copy.deepcopy(parent_dict[key])

def CopyDefaultEntries(parent_dict, child_dict):
	CheckAndDeepCopy(parent_dict, child_dict, 'info')
	CheckAndDeepCopy(parent_dict, child_dict, 'license')
	CheckAndDeepCopy(parent_dict, child_dict, 'data_type')
	CheckAndDeepCopy(parent_dict, child_dict, 'task_type')
	CheckAndDeepCopy(parent_dict, child_dict, 'num_choices')

def SplitValidSet(valid, valid_1_question_ids,
                  valid_2_question_ids, entries_key):
	valid_1 = {}
	valid_2 = {}

	# Copy default entries.
	CopyDefaultEntries(valid, valid_1)
	CopyDefaultEntries(valid, valid_2)

	# data_subtype
	valid_1['data_subtype'] = 'val-1-2014'
	valid_2['data_subtype'] = 'val-2-2014'

	qid_to_entries = {entry['question_id']:entry for entry in valid[entries_key]}
	valid_1[entries_key] = [qid_to_entries[qid] for qid in valid_1_question_ids]
	valid_2[entries_key] = [qid_to_entries[qid] for qid in valid_2_question_ids]

	return valid_1, valid_2

def CollectQuestionIdsByImageIds(annotations, image_ids):
	question_ids = []
	for anno in tqdm(annotations):
		if anno['image_id'] in image_ids:
			question_ids.append(anno['question_id'])
	return question_ids

def main():
	print "Loading validation annotation file.. ",
	valid_anno = json.load(open(VALID_ANNO, 'r'))
	print "Done."

	# Specify seed for reproducing.
	seed_value = 123
	print "Seed for reproduction is: {}".format(seed_value)
	random.seed(seed_value)

	image_ids = list(set([anno['image_id'] for anno in valid_anno['annotations']]))
	random.shuffle(image_ids)
	print "Total number of image ids: {}".format(len(image_ids))
	print "First five image ids:", image_ids[:5]

	print "Split validation with 3:1 ratio."
	valid_1_image_ids = image_ids[:len(image_ids)*3/4]
	valid_2_image_ids = image_ids[len(image_ids)*3/4:]
	print "Number of valid 1 image ids: {}".format(len(valid_1_image_ids))
	print "Number of valid 2 image ids: {}".format(len(valid_2_image_ids))

	print "Collect validation 1 question ids."
	valid_1_question_ids = CollectQuestionIdsByImageIds(
		valid_anno['annotations'], valid_1_image_ids)
	print "Number of valid 1 question ids: {}".format(len(valid_1_question_ids))

	print "Collect validation 2 question ids."
	valid_2_question_ids = CollectQuestionIdsByImageIds(
		valid_anno['annotations'], valid_2_image_ids)
	print "Number of valid 2 question ids: {}".format(len(valid_2_question_ids))

	# Split annotations.
	valid_1_anno, valid_2_anno = SplitValidSet(valid_anno, valid_1_question_ids,
                                              valid_2_question_ids,
                                              'annotations')
	print "Save valid 1 annotations to {}".format(VALID_1_ANNO)
	json.dump(valid_1_anno, open(VALID_1_ANNO, 'w'))
	print "Save valid 2 annotations to {}".format(VALID_2_ANNO)
	json.dump(valid_2_anno, open(VALID_2_ANNO, 'w'))

	# Split open ended questions.
	print "Spliting open ended validation questions.. ",
	valid_ques_oe = json.load(open(VALID_QUES_OPENENDED, 'r'))
	valid_1_ques_oe, valid_2_ques_oe = SplitValidSet(valid_ques_oe,
                                                    valid_1_question_ids,
                                                    valid_2_question_ids,
                                                    'questions')
	print "Done."

	print "Save valid 1 open-ended questions to {}".format(
		VALID_1_QUES_OPENENDED)	
	json.dump(valid_1_ques_oe, open(VALID_1_QUES_OPENENDED, 'w'))
	print "Save valid 2 open-ended questions to {}".format(
		VALID_2_QUES_OPENENDED)
	json.dump(valid_2_ques_oe, open(VALID_2_QUES_OPENENDED, 'w'))

	# Split multiple choice questions.
	print "Spliting multiple choice validation questions.. ",
	valid_ques_mc = json.load(open(VALID_QUES_MULTIPLECHOICE, 'r'))
	valid_1_ques_mc, valid_2_ques_mc = SplitValidSet(valid_ques_mc,
                                                    valid_1_question_ids,
                                                    valid_2_question_ids,
                                                    'questions')
	print "Done."

	print "Save valid 1 multiple-choice questions to {}".format(
		VALID_1_QUES_MULTIPLECHOICE)	
	json.dump(valid_1_ques_mc, open(VALID_1_QUES_MULTIPLECHOICE, 'w'))
	print "Save valid 2 multiple-choice questions to {}".format(
		VALID_2_QUES_MULTIPLECHOICE)
	json.dump(valid_2_ques_mc, open(VALID_2_QUES_MULTIPLECHOICE, 'w'))


if __name__ == "__main__":
	main()
