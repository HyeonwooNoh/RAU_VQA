cd VQA_anno

wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip
unzip Annotations_Train_mscoco.zip
rm -rf Annotations_Train_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip
unzip Annotations_Val_mscoco.zip
rm -rf Annotations_Val_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
rm -rf Questions_Train_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip
unzip Questions_Val_mscoco.zip
rm -rf Questions_Val_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip
unzip Questions_Test_mscoco.zip
rm -rf Questions_Test_mscoco.zip

cd ..
