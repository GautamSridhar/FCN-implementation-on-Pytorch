"""
Class and operations for loading the dataset for PyTorch

Class Descriptions - 

OpDataset - Reading images from the datset folder and creating a tuple with ground truths
            The names of the dataset images are saved in a csv file, with first column 
            images and second column having he corresponding ground truth images

Rescale -  Rescales the image to the desired output size

RandomCrop -  Does a Random crop from the image of the specified dimensions

RandomFlipVetical - Performs a random vertical flip on the image dependent on probability p. 
                    p=0.5 by default

RandomFlipHorizontal - Performs a random horizontal flip on the image dependent on 
                    probability p. p=0.5 by default

ToTensor -  Converts the image and ground truth image to Pytorch Tensor format
"""
import os
import torch
import pandas as pd
from skimage import io, transform,img_as_float
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils
import time

import warnings

#import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

root_dir = ''  #enter root directory to dataset with images here

names = pd.read_csv(root_dir + 'names.csv')

class MiceDataset(Dataset):
	"""Mice dataset loading"""
	def __init__(self,csv_file,root_dir,transform=None):
		"""
		Args:
				csv_file(string): Path to the csv file with annotations
				root_dir(string): Directory with all the images
				transform(callable,optional): Optional transform to be applied on a sample
		"""
		self.names = names
		self.root_dir = root_dir
		self.transform = transform
		
	def __len__(self):
		return len(self.names)
		
	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir,self.names.iloc[idx,0])
		image = img_as_float(io.imread(img_name))

		gtruth_name = os.path.join(self.root_dir,self.names.iloc[idx,1])
		gtruths = io.imread(gtruth_name)
		
		sample = {'image':image, 'gtruths':gtruths}
		
		if self.transform:
			sample = self.transform(sample)
		
		return sample

class Rescale(object):
	""" Rescale the image in a sample to a given size.
	Args:
		 Output_size(tuple or int): Desired output size. If tuple, output is matched to output_size. 
		 If int, smaller of image edges is matched to output_size keeping aspect ratio the same
	"""
	
	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size
		
	def __call__(self,sample):
		image,gtruths = sample['image'], sample['gtruths']
		
		h,w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h>w:
				new_h, new_w = self.output_size*w/h, self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size
		else:
			new_h, new_w = self.output_size
		
		new_h, new_w = int(new_h),int(new_w)
		
		img = img_as_ubyte(transform.resize(image,(new_h,new_w)))
		
		gtrut = img_as_ubyte(transform.resize(gtruths,(new_h,new_w)))
		
		return {'image':img, 'gtruths':gtrut}
		
class RandomCrop(object):
	""" Crop randomly the image in a sample
	Args:
		output_size(tuple or int): Desired output size. if int, square crop
	"""
	def __init__(self,output_size):
		assert isinstance(output_size, (int,tuple))
		if isinstance(output_size,int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
			
	def __call__(self,sample):
		image,gtruths = sample['image'],sample['gtruths']
		
		h,w = image.shape[:2]
		new_h, new_w = self.output_size
		
		left = np.random.randint(20, w-new_w)
		
		image = image[40:40 + new_h, left:left+new_w]
		gtruths = gtruths[40:40 + new_h, left:left+new_w]
		
		return{'image':image, 'gtruths':gtruths}
		
class RandomFlipVertical(object):
	""" Flip an image randomly on its vertical axis
	Args:
		probability p :  probability of the flip being carried out. Default p = 0.5
	"""
	def __init__(self,p = None):
		if p is None:
			self.p = 0.5
		else:
			self.p = p
	
	def __call__(self,sample):
		image,gtruths = sample['image'],sample['gtruths']
		a = np.random.uniform()
		if (a <= self.p):
			im = np.zeros((image.shape))
			gt = np.zeros((image.shape))
			for j in range(image.shape[1]):
				for i in range(image.shape[0]):
					im[i,j] = image[image.shape[0]-1-i,j]
					gt[i,j] = gtruths[image.shape[0]-1-i,j]
		else:
			im = image 
			gt = gtruths 
			
		return {'image':im, 'gtruths':gt}
		
class RandomFlipHorizontal(object):
	""" FLip an image randomly on its horizontal
	Args:
		probability p : probability of the flip being carried out. Default p = 0.5
	"""
	def __init__(self,p = None):
		if p is None:
			self.p = 0.5
		else:
			self.p = p
	def __call__(self,sample):
		image,gtruths = sample['image'],sample['gtruths']
		a = np.random.uniform()
		if (a <= self.p):
			im = np.zeros((image.shape))
			gt = np.zeros((image.shape))
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					im[i,j] = image[i,image.shape[1]-1-j]
					gt[i,j] = gtruths[i,gtruths.shape[1]-j-1]
		else:
			im = image
			gt = gtruths	
		return {'image':im, 'gtruths':gt}
		
class ToTensor(object):
	""" Convert ndarrays in sample to Tensors, convert the gtruth arrays to required format"""
	
	def __call__(self,sample):
		image,gtruths = sample['image'] , sample['gtruths']
		#image = (image - 0.09444)/0.2270                                #Mean normalisation of data followed by division by standard deviation
		#image = (image - 0.09444)                                       #Mean normalisation of data
		img = np.zeros((3,image.shape[0],image.shape[1]))                #Since this is a version with vgg pretrained weights, the input image if
		img[0,:,:] = image                                               #only in greyscale as in case od medical images has to be converted to 3
		img[1,:,:] = image                                               #channel image. Leave as is otherwise
		img[2,:,:] = image
		c_size = 12
		h,w = gtruths.shape[:2]

		gt_new = np.zeros((h,w), dtype = np.uint8)                      #Background pixels considered part of segmentation
		for i in range(c_size-1):                                       #Channels increased from 11-12
			idx = np.where(gtruths == i+1+70)
			idx_x = idx[0]
			idx_y = idx[1]
			for j in range(idx_x.shape[0]):
				gt_new[idx_x[j],idx_y[j]] = i+1										
		return {'image':torch.from_numpy(img).double(),'gtruths':torch.from_numpy(gt_new).long()}
		#return {'image':image,'gtruths':gt_new}
"""		
#Test Code				
if __name__ == "__main__":
	
	dataset_train = MiceDataset(csv_file=names,root_dir=root_dir,transform = transforms.Compose([ToTensor()]))
	
	i = 3
	sample = dataset_train[3]
	print(i, sample['gtruths'].type())
	plt.imshow(sample['gtruths'], cmap = 'gray')
	plt.show()
"""				