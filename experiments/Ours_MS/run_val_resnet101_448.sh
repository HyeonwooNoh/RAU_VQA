# script for training and testing Ours_ResNet model on VQA train-val / test-dev split
th LstmAttCtrlGradNoiseDontSelect.lua -visatt false -alg_name 'ResNet101MSsz448' -save_dir 'save_result_vqa_resnet101_448' -split 'val2014' -vqa_dir './data/VQA_prepro/data_train_val' -feat_dir './data/vqa_resnet_101_convfeat_448' -cnnout_w 14 -cnnout_h 14 -cnnout_dim 2048 -test_interval 20 -max_epochs 50 -free_interval 1 -batch_size 80 -gpuid 0
