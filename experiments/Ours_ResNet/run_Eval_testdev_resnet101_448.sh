# Evaluating trained Ours_ResNet on VQA test-dev set.
th Eval.lua -visatt false -alg_name 'OursResNet101Conv448' -save_dir 'save_result_eval_vqa_resnet101_448' -split 'test-dev2015' -vqa_dir './data/VQA_prepro/data_train-val_test-dev' -feat_dir './data/vqa_resnet_101_convfeat_448' -cnnout_w 14 -cnnout_h 14 -cnnout_dim 2048 -test_interval 1 -free_interval 1 -batch_size 80 -init_from $1
