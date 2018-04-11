# FCN-implementation-on-Pytorch
FCN implementation in Pytorch with and without pretrained weights

Prerequisites - 
1. Python
2. Numpy
3. PyTorch
4. Pillow
5. Pandas
6. tensorboard_logger

Install numpy and pandas from here  - https://www.scipy.org/install.html 
Install Pytorch from here - http://pytorch.org/ 
Install Pillow from her - https://pypi.python.org/pypi/Pillow/2.2.1 
Install tensorboard_logger from here - https://pypi.python.org/pypi/tensorboard_logger 

FCN8s_woVGG executes FCN8s without pretrained weights and with sigmoid + binary cross entropy loss 
FCN8s_wVGG executes FCN8s with pretrained weights and with sigmoid + binary cross entropy loss 
FCN8s_wVGG_mk2 executes FCN8s with pretrained weights and with softmax + cross entropy loss 
FCN8s_wVGG_mk2 executes FCN8s with pretrained weights and with softmax + weighted cross entropy loss 

To implement, download a dataset for segmentation like the PASCAL VOC dataset available here - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

To run, simply execute train_xx.py depending on which version you run. If using PyTorch with multiple GPU's use CUDA_VISIBLE_DEVICES=X python train_xx.py where X is the device id of the GPU. 

Follow the given instructions - 
1. Fill in root_dir in conv___.py files to signify where the dataset is stored
2. Create a names.csv file in root_dir with the names of all the images that have to be read organised as follows - 
   on the left column, the names of the input images, on the right column the names of the associated ground truth images.
3. Add or remove particular lines in the code as required and mentioned at appropriate places depending on application.
4. Fill in model_dir in train___.py files to signify where the models are to be stored
5. Fill in root_dir in train___.py files to signify where the dataset is stored
6. Fill ... in val function in all train.py to signify where the validation images should be stored

