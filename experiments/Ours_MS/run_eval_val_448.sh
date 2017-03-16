CHECKPOINT_PATH=$1
STEP_SELECTOR_PATH=$2
PREDICTION_TYPE=$3
SAVE_DIR=$4

if [[ "${CHECKPOINT_PATH}" != *.t7 ]]; then
	echo "Wrong first argument: path to the checkpoint (*.t7)"
	exit
fi
if [[ "${STEP_SELECTOR_PATH}" != *.json ]]; then
	echo "Wrong second argument: path to step selection result file (*.json)"
	exit
fi
if [ "${PREDICTION_TYPE}" != "is_best" ] && [ "${PREDICTION_TYPE}" != "best_shortest" ]; then
	echo "Wrong third argument: prediction_type [is_best|best_shortest]"
	exit
fi
if [ "${SAVE_DIR}" == "" ]; then
	echo "Wrong fourth argument: save_dir (ex: save_result_eval_ms_vgg_448)"
	exit
fi

# Evaluating trained Ours_ResNet on VQA test-dev set.
th eval.lua \
	-visatt false \
	-alg_name 'OursVGGMSConv448' \
	-save_dir ${SAVE_DIR} \
	-split 'val2014' \
	-vqa_dir './data/VQA_prepro/data_train_val-2' \
	-feat_dir './data/vqa_VGG16Conv_pool5_448/feat_448x448' \
	-cnnout_w 14 -cnnout_h 14 -cnnout_dim 512 \
	-test_interval 1 -free_interval 1 \
	-batch_size 80 \
	-init_from ${CHECKPOINT_PATH} \
	-step_selector_path ${STEP_SELECTOR_PATH} \
	-prediction_type ${PREDICTION_TYPE}
