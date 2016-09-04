## Experiments

This directory contains experimental scripts used for the paper [Training Recurrent Answering Units with Joint Loss Minimization for VQA](http://arxiv.org/abs/1606.03647).

Each directory [Ours_SS](Ours_SS), [Ours_MS](Ours_MS), [Ours_Full](Ours_Full) and [Ours_ResNet](Ours_ResNet) coincides with the entries of Table 1 of the main paper.

### Notes: 
Here are few things you should keep in mind using this codes.

1. We train every model for 40 epochs. You can run the scripts and it will make evaluation results in every epoch with all possible number of prediction steps.
We call prediction number of steps as "hop_##".
***To see the results in the paper, you can submit the results file with "hop_01", "epoch40" to the [VQA evaluation server](https://www.codalab.org/competitions/6961).***

2. As a result of training, the script will create a result directory named "save_result_vqa_448...".
This directory includes training and testing log, graphs, attention figure (if you set the visatt option), evaluation result and trained weights (snapshot).
Please refer the next section for the details of result directory.

3. For training and testing, you should run script such as run_testdev_448.sh or run_testdev_resnet101_448.sh.

4. You can also download the our training results by running script named download_trained_model.sh

### Result directory:
Result directory contains files such as training, testing log, graphs, attention figure, evaluation results and trained model weights.
Result directory contains following sub-directories.
* **figures**: If you set visatt option, the script will dump the visual attention score map. Resulting visual attention maps are saved in this directory. 
* **graphs**: You can find some useful graphs for training procedure in this directory.
* **results**: The script will perform evaluation on every iteration and generate results file in this directory.
You can submit this result file directory to the VQA evaluation server after compressing each file as results.zip file.
In this directory, there are sub-directories named hop_##. Each directory contains results from different prediction steps.
To obtain scores in the paper, you should see the results on hop_01 directory.
* **snapshot**: Trained model weights are saved in this directory. You can refer [Ours_ResNet/Eval.lua](experiments/Ours_ResNet/Eval.lua) to learn how you can read the weights and use it for prediction.
* **training_log**: You can see the training and testing logs in this directory.
