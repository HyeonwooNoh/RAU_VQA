# This script download preprocessed VQA data.
# Preprocessing is based on https://github.com/VT-vision-lab/VQA_LSTM_CNN and we add
# more splits for efficient evaluations.

cd VQA_prepro

mkdir data_train_val
cd data_train_val

wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/data_train_val.zip
unzip data_train_val.zip
rm -rf data_train_val.zip

cd ..

wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_val/data_train-val_test.zip
unzip data_train-val_test.zip
rm -rf data_train-val_test.zip

wget http://cvlab.postech.ac.kr/research/imageqa/data/vqa_prepro/data_train-val_test-dev.tar.gz
tar -zxvf data_train-val_test-dev.tar.gz
rm -rf data_train-val_test-dev.tar.gz

wget http://cvlab.postech.ac.kr/research/imageqa/data/vqa_prepro/data_train_train.tar.gz
tar -zxvf data_train_train.tar.gz
rm -rf data_train_train.tar.gz

wget http://cvlab.postech.ac.kr/research/imageqa/data/vqa_prepro/comprehend.tar.gz
tar -zxvf comprehend.tar.gz
rm -rf comprehend.tar.gz

cd ..
