# This script is for downloading VGG 16 pool5 features for every MSCOCO images.
# For extracting feature, we use VGG 16 pre-trained model which can be downloaded from the
# following link: https://github.com/torch/torch7/wiki/ModelZoo
# Images are resized to 448x448 size before feature extraction.
wget cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/data/vqa_VGG16Conv_pool5_448.tar.gz
tar -zxvf vqa_VGG16Conv_pool5_448.tar.gz
rm -rf vqa_VGG16Conv_pool5_448.tar.gz

