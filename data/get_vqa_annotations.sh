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

# Valid 1, Valid 2 annotations
wget cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/data/annotations/mscoco_val-1-2014_annotations.json
wget cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/data/annotations/mscoco_val-2-2014_annotations.json
wget cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/data/annotations/OpenEnded_mscoco_val-1-2014_questions.json
wget cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/data/annotations/OpenEnded_mscoco_val-2-2014_questions.json
wget cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/data/annotations/MultipleChoice_mscoco_val-1-2014_questions.json
wget cvlab.postech.ac.kr/~hyeonwoonoh/research/imageqa/data/annotations/MultipleChoice_mscoco_val-2-2014_questions.json

cd ..
