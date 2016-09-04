## RAU_VQA: Training Recurrent Answering Units with Joint Loss Minimization for VQA

Created by [Hyeonwoo Noh](http://cvlab.postech.ac.kr/~hyeonwoonoh/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at [POSTECH cvlab](http://cvlab.postech.ac.kr/lab/)

### Introduction

RAU_VQA includes codes used for experiments of the arxiv paper: 

[Training Recurrent Answering Units with Joint Loss Minimization for VQA](http://arxiv.org/abs/1606.03647)

This codes includes training, evaluation codes for [Ours_SS](experiments/Ours_SS), [Ours_MS](experiments/Ours_MS), [Ours_Full](experiments/Ours_Full) and [Ours_ResNet](experiments/Ours_ResNet) of the paper.

### Citation

If you're using this code in a publication, please cite our papers.

    @article{noh2016training,
      title={Training Recurrent Answering Units with Joint Loss Minimization for VQA},
      author={Noh, Hyeonwoo and Han, Bohyung},
      journal={arXiv preprint arXiv:1606.03647},
      year={2016}
    }

### Licence

This software is for research purpose only.
Check LICENSE file for details.

### System Requirements

  * This software is tested on Ubuntu 14.04 LTS (64bit).
  * At least 12GB gpu memory is required (NVIDIA tital-x gpu is used for training).

### Dependencies

  * Torch [https://github.com/torch/distro]
  * Torch-gnuplot [https://github.com/torch/gnuplot]
  * Torch-image [https://github.com/torch/image]
  * Torch-display [https://github.com/szym/display]
  * Lua-cjson [https://www.kyne.com.au/~mark/software/lua-cjson-manual.html]
  
### Directories
  * **./experiments**: Training, testing scripts for experiments. You can also download trained-model, evaluation results of our experiments by running download script.
  * **./data**: Data used for training / testing
  * **./model**: Trained model parameters, model definitions, layer implementations
  * **./utils**: Utilities (loading training data, loading models ...)
