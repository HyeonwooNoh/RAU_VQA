# You can download model trained in our system by running this sciprt.
# Once you download this trained model, you can check performance by submitting results files in the directory to
# the evaluation server, or you can get a new evaluationg results by running evaluation code with trained weights in
# snapshot directory.
wget http://cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/2016_NIPS/Ours_Full/save_result_vqa_test-dev2015.tar.gz
tar -zxvf save_result_vqa_test-dev2015.tar.gz
rm -rf save_result_vqa_test-dev2015.tar.gz
