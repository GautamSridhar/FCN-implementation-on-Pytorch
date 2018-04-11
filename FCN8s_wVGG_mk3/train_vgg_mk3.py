"""
Code for training the FCN8s with pretrained weights. Consists of two functions, train and val.
Train perfoms the training with a loss function - Negative Log Likelihood Loss with a log probabilities calculated  via Softmax.
Validation calculates the validation loss and stores the performance on a particular image at random
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms,utils,models
import FCN_VGG as fcn
import convgg_mk3 as Dat

from PIL import Image
import numpy as np
import pandas as pd
import time
import sys
import os

from tensorboard_logger import configure, log_value

#import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"]= 4

configure("runs/run_11",flush_secs=5)

model_dir = ""

n_class = 12

batch_size = 5
epochs = 26
lr = 9e-5
momentum = 0
w_decay = 0
#step_size = 15
gamma = 0.5

use_gpu = torch.cuda.is_available()

#Load the training data
root_dir = '' #enter root directory to dataset with images here

names = pd.read_csv(root_dir + 'names.csv')
dataset_train = Dat.OpDataset(csv_file=names,root_dir=root_dir,transform=transforms.Compose([Dat.RandomCrop((224,224)),Dat.RandomFlipHorizontal(),Dat.ToTensor()]))

#Test the dataset
#sample = dataset_train[3]
#print(3, sample['image'].size(), sample['gtruths'].size())
#time.sleep(60)
#plt.imshow(sample['gtruths'][3,:,:], cmap = 'gray')
#plt.show()
#print(sample['gtruths'][10,:,:].max())
#time.sleep(60)

train_loader = DataLoader(dataset_train,batch_size=5,shuffle=True) #possible num_workers = ?

#Load validation data 
dataset_val = Dat.OpDataset(csv_file=names,root_dir=root_dir,transform=transforms.Compose([Dat.RandomCrop((224,224)),Dat.RandomFlipHorizontal(),Dat.ToTensor()]))

#Test the dataset
#sample = dataset_val[3]
#print(3, sample['image'].type(), sample['gtruths'].type())
#time.sleep(60)

val_loader = DataLoader(dataset_val,batch_size=5,shuffle=True)

vgg = models.vgg16(pretrained=True)
fcn_model = fcn.FCN8s(n_class,vgg) 
#time.sleep(60)

if use_gpu:                           #Load the model to the gpu if available
	ts = time.time()
	fcn_model = fcn_model.cuda()
	print("Finished cuda loading, time elapsed {}".format(time.time()-ts))

last_layer = nn.LogSoftmax(dim=1)    #SoftMax followed by logarithm for log probabilities
weights = torch.Tensor([1,100,100,100,100,100,100,100,100,100,100,100])  #Weights to stress more on the foreground segmentation rather than background
weights = weights.cuda()
criterion = nn.NLLLoss(weight = weights)                                                                #Negative Log likelihood loss
#optimiser = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)      #RMSprop optimiser
#optimiser = optim.SGD(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)          #SGD optimiser
optimiser = optim.Adam(fcn_model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=w_decay)    #Adam optimiser
scheduler = lr_scheduler.ExponentialLR(optimiser,gamma=gamma)                                           #Learning Rate annealing

#create directory for saving the score

def train():
	for epoch in range(epochs):
		scheduler.step()
		
		ts = time.time()
		for iter,batch in enumerate(train_loader):
			optimiser.zero_grad()
			
			if use_gpu:
				inputs = batch['image'].float()
				labels = batch['gtruths']
				inputs = Variable(inputs.cuda())
				labels = Variable(labels.cuda())
			else:
				inputs,labels = Variable(batch['image']),Variable(batch['gtruths'])
				
			output_nn = fcn_model(inputs)
			outputs = last_layer(output_nn) 
			loss = criterion(outputs,labels)
			loss.backward()
			summ = loss
			optimiser.step()
			
			print ("epoch {}, iter {}, loss: {}".format(epoch, iter, loss.data[0]))
			
		
		log_value('Training Loss', summ, epoch)	
		print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
		
		
		if epoch%5==0:
			val(epoch)               #Perform Validation every 5th epoch
		
		if epoch==2500 or epoch==5000:                  #Save the model on every 2500th or 5000th epoch
			model_path = os.path.join(model_dir,"epoch_{}/".format(epoch))
			if not os.path.exists(model_path):
				os.makedirs(model_path)
			torch.save(fcn_model,model_path+"epoch_{}".format(epoch))
		
		
def val(epoch):
	fcn_model.eval()
	pixel_accs = []
	for iter,batch in enumerate(val_loader):
		if use_gpu:
			inputs = batch['image'].float()
			labels = batch['gtruths']
			inputs = Variable(inputs.cuda())
			labels = Variable(labels.cuda())
		else:
			inputs,labels = Variable(batch['image']),Variable(batch['gtruths'])
		
		outputs_nn = fcn_model(inputs)
		outputs = last_layer(outputs_nn)
		val_loss = criterion(outputs,labels)
		val_summ = val_loss.data[0]
		
	print ("epoch {}, validation_loss: {}".format(epoch, val_summ))
		
	log_value("Validation Loss", val_summ, epoch)
	
	i = np.random.randint(0,44)
	sample = dataset_val[i]                #Randomly sample from dataset and evaluate on the model
	inp = sample['image']                  #and save the output images to check how well the dataset is doing  
	targ = sample['gtruths'].cpu().numpy()
	targ = targ.astype(np.uint8)
	
	t = np.zeros((224,224),dtype = np.uint8)
	for i in range(11):
		idx = np.where(targ == i+1)
		idx_x = idx[0]
		idx_y = idx[1]
		for j in range(idx_x.shape[0]):
			t[idx_x[j],idx_y[j]] = i+1+70
		
	inp = inp.unsqueeze(0)
	inp = Variable(inp.float().cuda())
	o_nn = fcn_model(inp)
	smax = nn.Softmax2d()
	o = smax(o_nn) 
	o = o.data.cpu().numpy()

	tim = Image.fromarray(t)
	tim.save('.../epoch_{}_target_img.png'.format(epoch))   #Fill ... with the folder where the images will be stored
	for j in range(o.shape[1]):
		a = o[0,j,:,:]

		a = a*255
		b = a.astype(np.uint8)

		im = Image.fromarray(b)
		im.save('.../epoch_{}_layer_{}_o_img.png'.format(epoch,j)) #Fill ... with the folder where the images will be stored

if __name__ == "__main__":
	val(0)
	train()
		
	
	
	
	
		

	

