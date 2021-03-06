"""
Architecture definition for Fully Convolutional Neural Networks (FCN8s)
Initialised with pretrained VGG weights
Weights initialised by bilinear upsampling for convolutional transpose layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
import time

#os.environ["CUDA_VISIBLE_DEVICES"]= "4"

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

	
def get_upsampling_weight(in_channels,out_channels,kernel_size):
	""" Make a 2D bilinear kernel suitable for upsampling"""
	factor = (kernel_size+1)//2
	if kernel_size%2 == 1:
		center = factor-1
	else:
		center = factor-1
	og = np.ogrid[:kernel_size, :kernel_size]
	filt = (1 - abs(og[0] - center)/factor) * (1 - abs(og[1] - center)/factor)
	weight = np.zeros((in_channels,out_channels,kernel_size,kernel_size), dtype = np.float64)
	weight[range(in_channels), range(out_channels),:,: ]
	return torch.from_numpy(weight).float()
	
class FCN8s(nn.Module):
	def __init__(self,n_class,vgg):
		
		super(FCN8s,self).__init__()
		self.vgg = vgg                                         #VGG architecture definition   
		#conv1
		self.conv1_1 = nn.Conv2d(3,64,3,padding=100)
		self.relu1_1 = nn.ReLU(inplace=True)
		self.bn1_1 = nn.BatchNorm2d(64)
		self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
		self.relu1_2 = nn.ReLU(inplace=True)
		self.bn1_2 = nn.BatchNorm2d(64)
		self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #1/2 dimension reduction
		
		#conv2
		self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
		self.relu2_1 = nn.ReLU(inplace=True)
		self.bn2_1 = nn.BatchNorm2d(128)
		self.conv2_2 = nn.Conv2d(128,128,3,padding=1)
		self.relu2_2 = nn.ReLU(inplace=True)
		self.bn2_2 = nn.BatchNorm2d(128)
		self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/4 dimension reduction 
		
		
		#conv3 
		self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
		self.relu3_1 = nn.ReLU(inplace = True)
		self.bn3_1 = nn.BatchNorm2d(256)
		self.conv3_2 = nn.Conv2d(256,256,3,padding=1)
		self.relu3_2 = nn.ReLU(inplace = True)
		self.bn3_2 = nn.BatchNorm2d(256)
		self.conv3_3 = nn.Conv2d(256,256,3,padding=1)
		self.relu3_3 = nn.ReLU(inplace = True)
		self.bn3_3 = nn.BatchNorm2d(256)
		self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8 dimension reduction
		
		#conv4
		self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
		self.relu4_1 = nn.ReLU(inplace=True)
		self.bn4_1 = nn.BatchNorm2d(512)
		self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu4_2 = nn.ReLU(inplace=True)
		self.bn4_2 = nn.BatchNorm2d(512)
		self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu4_3 = nn.ReLU(inplace=True)
		self.bn4_3 = nn.BatchNorm2d(512)
		self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/16 dimension reduction
		
		#conv5
		self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_1 = nn.ReLU(inplace=True)
		self.bn5_1 = nn.BatchNorm2d(512)
		self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_2 = nn.ReLU(inplace=True)
		self.bn5_2 = nn.BatchNorm2d(512)
		self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_3 = nn.ReLU(inplace=True)
		self.bn5_3 = nn.BatchNorm2d(512)
		self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/32 dimension reduction
		
		#fc6
		self.fc6 = nn.Conv2d(512,4096,7)
		self.relu6 = nn.ReLU(inplace=True)
		self.bn6 = nn.BatchNorm2d(4096)
		self.drop6 = nn.Dropout2d()
		
		#fc7
		self.fc7 = nn.Conv2d(4096,4096,1)
		self.relu7 = nn.ReLU(inplace = True)
		self.bn7 = nn.BatchNorm2d(4096)
		self.drop7 = nn.Dropout2d()
		
		self.score_fr = nn.Conv2d(4096,n_class,1)             #Skip Layer defintions
		self.bn_fr = nn.BatchNorm2d(n_class)
		self.score_pool3 = nn.Conv2d(256,n_class,1)
		self.bn_pool3 = nn.BatchNorm2d(n_class)
		self.score_pool4 = nn.Conv2d(512,n_class,1)
		self.bn_pool4 = nn.BatchNorm2d(n_class)
		
		self.upscore2 = nn.ConvTranspose2d(n_class,n_class,4,stride=2,bias=False)     #Upsampling layer defintions
		self.bn_upscore2 = nn.BatchNorm2d(n_class)
		self.upscore8 = nn.ConvTranspose2d(n_class,n_class,16,stride=8,bias=False)
		self.upscore_pool4 = nn.ConvTranspose2d(n_class,n_class,4,stride=2,bias=False)
		self.bn_upscore_pool4 = nn.BatchNorm2d(n_class)
		
		self._copy_params_from_vgg16()
									
	def forward(self,x):
		h = x
		h = self.bn1_1(self.relu1_1(self.conv1_1(h)))
		h = self.bn1_2(self.relu1_2(self.conv1_2(h)))
		h = self.pool1(h)
			
		h = self.bn2_1(self.relu2_1(self.conv2_1(h)))
		h = self.bn2_2(self.relu2_2(self.conv2_2(h)))
		h = self.pool2(h)
			
		h = self.bn3_1(self.relu3_1(self.conv3_1(h)))
		h = self.bn3_2(self.relu3_2(self.conv3_2(h)))
		h = self.bn3_3(self.relu3_3(self.conv3_3(h)))
		h = self.pool3(h)
		pool3 = h # 1/8
			
		h = self.bn4_1(self.relu4_1(self.conv4_1(h)))
		h = self.bn4_2(self.relu4_2(self.conv4_2(h)))
		h = self.bn4_3(self.relu4_3(self.conv4_3(h)))
		h = self.pool4(h)
		pool4 = h # 1/16
			
		h = self.bn5_1(self.relu5_1(self.conv5_1(h)))
		h = self.bn5_2(self.relu5_2(self.conv5_2(h)))
		h = self.bn5_3(self.relu5_3(self.conv5_3(h)))
		h = self.pool5(h)
		
			
		h = self.bn6(self.relu6(self.fc6(h)))
		h = self.drop6(h)
			
		h = self.bn7(self.relu7(self.fc7(h)))
		h = self.drop7(h)
			
		h = self.bn_fr(self.score_fr(h))
		h = self.bn_upscore2(self.upscore2(h))
		upscore2 = h  # 1/16
			
		h = self.bn_pool4(self.score_pool4(pool4))
		h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
		score_pool4c = h
			
		h = upscore2 + score_pool4c # 1/16
		h = self.bn_upscore_pool4(self.upscore_pool4(h))
		upscore_pool4 = h # 1/8
			
		h = self.bn_pool3(self.score_pool3(pool3))
		h = h[:,:,9:9+upscore_pool4.size()[2],9:9+upscore_pool4.size()[3]]
		score_pool3c = h
			
		h = upscore_pool4 + score_pool3c
			
		h = self.upscore8(h)
		h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
				
		return h
	
	def _copy_params_from_vgg16(self):
        #Copy VGG parameters from a pretrained VGG16 net available on Pytorch
        #Generate weights for all layers not part of VGG16 by either Xavier or Bilinear upsampling

		features = [self.conv1_1, self.relu1_1, self.conv1_2, self.relu1_2, self.pool1, self.conv2_1, self.relu2_1, self.conv2_2, self.relu2_2, self.pool2,self.conv3_1, self.relu3_1, self.conv3_2, self.relu3_2, self.conv3_3, self.relu3_3, self.pool3,self.conv4_1, self.relu4_1, self.conv4_2, self.relu4_2, self.conv4_3, self.relu4_3, self.pool4, self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5,]
		for l1,l2 in zip(self.vgg.features,features):
			if isinstance(l1,nn.Conv2d) and isinstance(l2,nn.Conv2d):
				assert l1.weight.size() == l2.weight.size()
				assert l1.bias.size() == l2.bias.size()
				l2.weight.data.copy_(l1.weight.data).double()
				l2.bias.data.copy_(l1.bias.data).double()
				l2.bias.data.copy_(l1.bias.data).double()

		classifier = [self.fc6,self.relu6,self.drop6,self.fc7,self.relu7,self.drop7,self.score_fr,self.score_pool3,self.score_pool4,self.upscore2,self.upscore8,self.upscore_pool4]
		for i in classifier:
			if isinstance(i,nn.Conv2d):
				n = i.kernel_size[0] * i.kernel_size[1] *i.out_channels
				i.weight.data.normal_(0,math.sqrt(2./n))
				if i.bias is not None:
					i.bias.data.zero_()
			if isinstance(i,nn.ConvTranspose2d):
				assert i.kernel_size[0] == i.kernel_size[1]
				initial_weight = get_upsampling_weight(i.in_channels,i.out_channels,i.kernel_size[0])
				i.weight.data.copy_(initial_weight)
					   	
"""
#Test Code			
if __name__ == "__main__":
	batch_size, n_class, h,w = 5,11,224,224
	
	#test the output size
	
	fcn_model = FCN8s(n_class)
	if use_gpu:
		ts = time.time()
		fcn_model = fcn_model.cuda()
		print ("Finsished loading CUDA, time elapsed {}".format(time.time()-ts))
	
	input = torch.autograd.Variable(torch.randn(batch_size,3,h,w).cuda())
	print("hello")
	output = fcn_model(input)
	print(output.size())
	
	#To check whether training properly
	y = torch.autograd.Variable(torch.randn(batch_size,n_class,h,w).cuda())
	criterion = nn.BCEWithLogitsLoss()
	optimiser = optim.SGD(fcn_model.parameters(),lr=1e-3,momentum=0.9)
	for iter in range(10):
		optimiser.zero_grad()
		output = fcn_model(input)
		loss = criterion(output,y)
		loss.backward()
		print("iter {}, loss {}".format(iter, loss.data[0]))
		optimiser.step()
"""
