# FCN-implementation-on-Pytorch
FCN implementation in Pytorch with and without pretrained weights

Prerequisites - 
1. Python
2. Numpy
3. PyTorch
4. Pillow
5. Pandas
6. tensorboard_logger

Install numpy and pandas from here  - https://www.scipy.org/install.html \n
Install Pytorch from here - http://pytorch.org/ \n
Install Pillow from her - https://pypi.python.org/pypi/Pillow/2.2.1 \n
Install tensorboard_logger from here - https://pypi.python.org/pypi/tensorboard_logger \n

FCN8s_woVGG executes FCN8s without pretrained weights and with sigmoid + binary cross entropy loss \n
FCN8s_wVGG executes FCN8s with pretrained weights and with sigmoid + binary cross entropy loss \n
FCN8s_wVGG_mk2 executes FCN8s with pretrained weights and with softmax + cross entropy loss \n
FCN8s_wVGG_mk2 executes FCN8s with pretrained weights and with softmax + weighted cross entropy loss \n
\n
To run, simply execute train_xx.py depending on which version you run. If using PyTorch with multiple GPU's use CUDA_VISIBLE_DEVICES=X python train_xx.py where X is the device id of the GPU. \n

Fill in the following in the code - 
1. root_dir in conv___.py files to signify where the dataset is stored
2. model_dir in train___.py files to signify where the models are to be stored
3. root_dir in train___.py files to signify where the dataset is stored
4. Fill ... in val function in all train.py to signify where the validation images should be stored

