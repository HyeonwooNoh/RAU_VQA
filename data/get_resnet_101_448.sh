# This script is for downloading resnet 101 features for every MSCOCO images.
# ResNet 101 from following link is used: https://github.com/facebook/fb.resnet.torch/tree/master/pretrained
# We employ activations just before the global average pooling layer of ResNet 101 as features.
# Images are resized to 448x448 size before feature extraction. 
wget cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/data/vqa_resnet_101_convfeat_448.tar.gz
tar -zxvf vqa_resnet_101_convfeat_448.tar.gz
rm -rf vqa_resnet_101_convfeat_448.tar.gz
