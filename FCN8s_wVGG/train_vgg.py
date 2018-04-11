"""
Code for training the FCN8s with pretrained weights. Consists of two functions, train and val.
Train perfoms the training with a loss function - Binary Cross Entropy Loss with Logits i.e performs a sigmoid on the layer before the loss.
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
import convgg as Dat

from PIL import Image
import numpy as np
import pandas as pd
import time
import sys
import os

from tensorboard_logger import configure, log_value

#import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"]= 4

configure("runs/run_1",flush_secs=5)

model_dir = "" #enter root directory to store saved models

n_class = 11

batch_size = 5
epochs = 5001
lr = 1e-4
momentum = 0
w_decay = 0
step_size = 25
gamma = 0.3

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

if use_gpu:                        #Load the model to the gpu if available
	ts = time.time()
	fcn_model = fcn_model.cuda()
	print("Fnished cuda loading, time elapsed {}".format(time.time()-ts))

criterion = nn.BCEWithLogitsLoss()               #Binary cross entropy loss with logits
#optimiser = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)   #RMSprop optimiser
#optimiser = optim.SGD(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)       #SGD optimiser
optimiser = optim.Adam(fcn_model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=w_decay) #Adam optimiser
scheduler = lr_scheduler.StepLR(optimiser, step_size=step_size,gamma=gamma)                          #Learning Rate annealing

#create directory for saving the score

def train():
	for epoch in range(epochs):
		scheduler.step()
		
		ts = time.time()
		for iter,batch in enumerate(train_loader):
			optimiser.zero_grad()
			
			if use_gpu:
				inputs = batch['image'].float()
				labels = batch['gtruths'].float()
				inputs = Variable(inputs.cuda())
				labels = Variable(labels.cuda())
			else:
				inputs,labels = Variable(batch['image']),Variable(batch['gtruths'])
				
			outputs = fcn_model(inputs)
			loss = criterion(outputs,labels)
			loss.backward()
			summ = loss
			optimiser.step()
			
			print ("epoch {}, iter {}, loss: {}".format(epoch, iter, loss.data[0]))
			
		
		log_value('Training Loss', summ, epoch)	
		print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
		
		
		if epoch%5==0:
			val(epoch)              #Perform Validation every 5th epoch
		
		if epoch==2500 or epoch==5000:                    #Save the model on every 2500th or 5000th epoch
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
			labels = batch['gtruths'].float()
			inputs = Variable(inputs.cuda())
			labels = Variable(labels.cuda())
		else:
			inputs,labels = Variable(batch['image']),Variable(batch['gtruths'])
		
		outputs = fcn_model(inputs)
		val_loss = criterion(outputs,labels)
		val_summ = val_loss.data[0]
		
	print ("epoch {}, validation_loss: {}".format(epoch, val_summ))
		
	log_value("Validation Loss", val_summ, epoch)
	
	i = np.random.randint(0,44)
	sample = dataset_val[i]

	inp = sample['image']                    #Randomly sample from dataset and evaluate on the model       
	targ = sample['gtruths'].cpu().numpy()   #and save the output images to check how well the dataset is doing
	inp = inp.unsqueeze(0)
	inp = Variable(inp.float().cuda())
	o = fcn_model(inp)
	sig = nn.Sigmoid()
	o = sig(o)
	o = o.data.cpu().numpy()
	for j in range(o.shape[1]):
		a = o[0,j,:,:]
		t = targ[j,:,:]
		a = a*255
		t = t*255
		b = a.astype(np.uint8)
		t = t.astype(np.uint8)
		im = Image.fromarray(b)
		im.save('/home/sridharg/Documents/Biomed_Imaging/images/epoch_{}_layer_{}_o_img.png'.format(epoch,j))
		tim = Image.fromarray(t)
		tim.save('/home/sridharg/Documents/Biomed_Imaging/images/epoch_{}_layer_{}_target_img.png'.format(epoch,j))
	
	
	#t = sample['gtruths'].byte().cpu().numpy()
	#tim = Image.fromarray(t)
	#tim.save('/home/sridharg/Documents/Biomed_Imaging/images/epoch_{}_target_img.png'.format(epoch))
	
			
	#pixel_accs = np.array(pixel_accs).mean()
	#print('Epoch{}, Pixel Accuracy {}'.format(epoch, pixel_accs))
	#log_value('Pixel Accuracy',pixel_accs,epoch) 

if __name__ == "__main__":
	val(0)
	train()
		
	
	
	
	
		

	

