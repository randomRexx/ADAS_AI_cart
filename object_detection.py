from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pyrealsense2 as rs
from math import tan, pi
import json

def AI_start():
	device = torch.device("cudo:0" if torch.cuda.is_available() else "cpu")
	
	class UBNeck(nn.Module):
		def __init__(self, in_channels, out_channels, relu=False, projection_ratio=4):
			
			super().__init__()
			
			# Define class variables
			self.in_channels = in_channels
			self.reduced_depth = int(in_channels / projection_ratio)
			self.out_channels = out_channels
			
			
			if relu:
				activation = nn.ReLU()
			else:
				activation = nn.PReLU()
			
			self.unpool = nn.MaxUnpool2d(kernel_size = 2,
										 stride = 2)
			
			self.main_conv = nn.Conv2d(in_channels = self.in_channels,
										out_channels = self.out_channels,
										kernel_size = 1)
			
			self.dropout = nn.Dropout2d(p=0.1)
			
			self.convt1 = nn.ConvTranspose2d(in_channels = self.in_channels,
								   out_channels = self.reduced_depth,
								   kernel_size = 1,
								   padding = 0,
								   bias = False)
			
			
			self.prelu1 = activation
			
			self.convt2 = nn.ConvTranspose2d(in_channels = self.reduced_depth,
									  out_channels = self.reduced_depth,
									  kernel_size = 3,
									  stride = 2,
									  padding = 1,
									  output_padding = 1,
									  bias = False)
			
			self.prelu2 = activation
			
			self.convt3 = nn.ConvTranspose2d(in_channels = self.reduced_depth,
									  out_channels = self.out_channels,
									  kernel_size = 1,
									  padding = 0,
									  bias = False)
			
			self.prelu3 = activation
			
			self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
			self.batchnorm2 = nn.BatchNorm2d(self.out_channels)
			
		def forward(self, x, indices):
			x_copy = x
			
			# Side Branch
			x = self.convt1(x)
			x = self.batchnorm(x)
			x = self.prelu1(x)
			
			x = self.convt2(x)
			x = self.batchnorm(x)
			x = self.prelu2(x)
			
			x = self.convt3(x)
			x = self.batchnorm2(x)
			
			x = self.dropout(x)
			
			# Main Branch
			
			x_copy = self.main_conv(x_copy)
			x_copy = self.unpool(x_copy, indices, output_size=x.size())
			
			# Concat
			x = x + x_copy
			x = self.prelu3(x)
			
			return x
		
	class RDDNeck(nn.Module):
		def __init__(self, dilation, in_channels, out_channels, down_flag, relu=False, projection_ratio=4, p=0.1):
			
			super().__init__()
			
			# Define class variables
			self.in_channels = in_channels
			
			self.out_channels = out_channels
			self.dilation = dilation
			self.down_flag = down_flag

			if down_flag:
				self.stride = 2
				self.reduced_depth = int(in_channels // projection_ratio)
			else:
				self.stride = 1
				self.reduced_depth = int(out_channels // projection_ratio)
			
			if relu:
				activation = nn.ReLU()
			else:
				activation = nn.PReLU()
			
			self.maxpool = nn.MaxPool2d(kernel_size = 2,
										  stride = 2,
										  padding = 0, return_indices=True)
			

			
			self.dropout = nn.Dropout2d(p=p)

			self.conv1 = nn.Conv2d(in_channels = self.in_channels,
								   out_channels = self.reduced_depth,
								   kernel_size = 1,
								   stride = 1,
								   padding = 0,
								   bias = False,
								   dilation = 1)
			
			self.prelu1 = activation
			
			self.conv2 = nn.Conv2d(in_channels = self.reduced_depth,
									  out_channels = self.reduced_depth,
									  kernel_size = 3,
									  stride = self.stride,
									  padding = self.dilation,
									  bias = True,
									  dilation = self.dilation)
									  
			self.prelu2 = activation
			
			self.conv3 = nn.Conv2d(in_channels = self.reduced_depth,
									  out_channels = self.out_channels,
									  kernel_size = 1,
									  stride = 1,
									  padding = 0,
									  bias = False,
									  dilation = 1)
			
			self.prelu3 = activation
			
			self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
			self.batchnorm2 = nn.BatchNorm2d(self.out_channels)
			
			
		def forward(self, x):
			
			bs = x.size()[0]
			x_copy = x
			
			# Side Branch
			x = self.conv1(x)
			x = self.batchnorm(x)
			x = self.prelu1(x)
			
			x = self.conv2(x)
			x = self.batchnorm(x)
			x = self.prelu2(x)
			
			x = self.conv3(x)
			x = self.batchnorm2(x)
					
			x = self.dropout(x)
			
			# Main Branch
			if self.down_flag:
				x_copy, indices = self.maxpool(x_copy)
			  
			if self.in_channels != self.out_channels:
				out_shape = self.out_channels - self.in_channels
				extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
				extras = extras.to(device)
				x_copy = torch.cat((x_copy, extras), dim = 1)

			# Sum of main and side branches
			x = x + x_copy
			x = self.prelu3(x)
			
			if self.down_flag:
				return x, indices
			else:
				return x
			
	class InitialBlock(nn.Module):
		def __init__ (self,in_channels = 3,out_channels = 13):
			super().__init__()


			self.maxpool = nn.MaxPool2d(kernel_size=2, 
										  stride = 2, 
										  padding = 0)

			self.conv = nn.Conv2d(in_channels, 
									out_channels,
									kernel_size = 3,
									stride = 2, 
									padding = 1)

			self.prelu = nn.PReLU(16)

			self.batchnorm = nn.BatchNorm2d(out_channels)
	  
		def forward(self, x):
			
			main = self.conv(x)
			main = self.batchnorm(main)
			
			side = self.maxpool(x)
			
			x = torch.cat((main, side), dim=1)
			x = self.prelu(x)
			
			return x
	class ENet_encoder(nn.Module):
		def __init__(self, C):
			super().__init__()
			
			# Define class variables
			self.C = C
			
			# The initial block
			self.init = InitialBlock()
			
			
			# The first bottleneck
			self.b10 = RDDNeck(dilation=1, 
							   in_channels=16, 
							   out_channels=64, 
							   down_flag=True, 
							   p=0.01)
			
			self.b11 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   p=0.01)
			
			self.b12 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   p=0.01)
			
			self.b13 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   p=0.01)
			
			self.b14 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   p=0.01)
			
			
			# The second bottleneck
			self.b20 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=128, 
							   down_flag=True)
			
			self.b21 = RDDNeck(dilation=1, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b22 = RDDNeck(dilation=2, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b23 = ASNeck(in_channels=128, 
							  out_channels=128)
			
			self.b24 = RDDNeck(dilation=4, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b25 = RDDNeck(dilation=1, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b26 = RDDNeck(dilation=8, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b27 = ASNeck(in_channels=128, 
							  out_channels=128)
			
			self.b28 = RDDNeck(dilation=16, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			
			# The third bottleneck
			self.b31 = RDDNeck(dilation=1, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b32 = RDDNeck(dilation=2, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b33 = ASNeck(in_channels=128, 
							  out_channels=128)
			
			self.b34 = RDDNeck(dilation=4, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b35 = RDDNeck(dilation=1, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b36 = RDDNeck(dilation=8, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b37 = ASNeck(in_channels=128, 
							  out_channels=128)
			
			self.b38 = RDDNeck(dilation=16, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)

			
			# Final ConvTranspose Layer
			self.fullconv_encoder = nn.ConvTranspose2d(in_channels=128,
											   out_channels=self.C, 
											   kernel_size=1,
											   stride=1,
											   padding=0,
											   output_padding=0,
											   bias=False)
			
			
		def forward(self, x):
			
			# The initial block
			x = self.init(x)
			
			# The first bottleneck
			x, i1 = self.b10(x)
			x = self.b11(x)
			x = self.b12(x)
			x = self.b13(x)
			x = self.b14(x)
			
			# The second bottleneck
			x, i2 = self.b20(x)
			x = self.b21(x)
			x = self.b22(x)
			x = self.b23(x)
			x = self.b24(x)
			x = self.b25(x)
			x = self.b26(x)
			x = self.b27(x)
			x = self.b28(x)
			
			# The third bottleneck
			x = self.b31(x)
			x = self.b32(x)
			x = self.b33(x)
			x = self.b34(x)
			x = self.b35(x)
			x = self.b36(x)
			x = self.b37(x)
			x = self.b38(x)
			
			# Final ConvTranspose Layer
			x = self.fullconv_encoder(x)
			
			return x
	class ENet(nn.Module):
		def __init__(self, C):
			super().__init__()
			
			# Define class variables
			self.C = C
			
			# The initial block
			self.init = InitialBlock()
			
			
			# The first bottleneck
			self.b10 = RDDNeck(dilation=1, 
							   in_channels=16, 
							   out_channels=64, 
							   down_flag=True, 
							   p=0.01)
			
			self.b11 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   p=0.01)
			
			self.b12 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   p=0.01)
			
			self.b13 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   p=0.01)
			
			self.b14 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   p=0.01)
			
			
			# The second bottleneck
			self.b20 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=128, 
							   down_flag=True)
			
			self.b21 = RDDNeck(dilation=1, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b22 = RDDNeck(dilation=2, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b23 = ASNeck(in_channels=128, 
							  out_channels=128)
			
			self.b24 = RDDNeck(dilation=4, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b25 = RDDNeck(dilation=1, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b26 = RDDNeck(dilation=8, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b27 = ASNeck(in_channels=128, 
							  out_channels=128)
			
			self.b28 = RDDNeck(dilation=16, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			
			# The third bottleneck
			self.b31 = RDDNeck(dilation=1, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b32 = RDDNeck(dilation=2, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b33 = ASNeck(in_channels=128, 
							  out_channels=128)
			
			self.b34 = RDDNeck(dilation=4, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b35 = RDDNeck(dilation=1, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b36 = RDDNeck(dilation=8, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			self.b37 = ASNeck(in_channels=128, 
							  out_channels=128)
			
			self.b38 = RDDNeck(dilation=16, 
							   in_channels=128, 
							   out_channels=128, 
							   down_flag=False)
			
			
			# The fourth bottleneck
			self.b40 = UBNeck(in_channels=128, 
							  out_channels=64, 
							  relu=True)
			
			self.b41 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   relu=True)
			
			self.b42 = RDDNeck(dilation=1, 
							   in_channels=64, 
							   out_channels=64, 
							   down_flag=False, 
							   relu=True)
			
			
			# The fifth bottleneck
			self.b50 = UBNeck(in_channels=64, 
							  out_channels=16, 
							  relu=True)
			
			self.b51 = RDDNeck(dilation=1, 
							   in_channels=16, 
							   out_channels=16, 
							   down_flag=False, 
							   relu=True)
			
			
			# Final ConvTranspose Layer
			self.fullconv = nn.ConvTranspose2d(in_channels=16,
											   out_channels=self.C, 
											   kernel_size=3, 
											   stride=2, 
											   padding=1, 
											   output_padding=1,
											   bias=False)
			
			
		def forward(self, x):
			
			# The initial block
			x = self.init(x)
			
			# The first bottleneck
			x, i1 = self.b10(x)
			x = self.b11(x)
			x = self.b12(x)
			x = self.b13(x)
			x = self.b14(x)
			
			# The second bottleneck
			x, i2 = self.b20(x)
			x = self.b21(x)
			x = self.b22(x)
			x = self.b23(x)
			x = self.b24(x)
			x = self.b25(x)
			x = self.b26(x)
			x = self.b27(x)
			x = self.b28(x)
			
			# The third bottleneck
			x = self.b31(x)
			x = self.b32(x)
			x = self.b33(x)
			x = self.b34(x)
			x = self.b35(x)
			x = self.b36(x)
			x = self.b37(x)
			x = self.b38(x)
			
			# The fourth bottleneck
			x = self.b40(x, i2)
			x = self.b41(x)
			x = self.b42(x)
			
			# The fifth bottleneck
			x = self.b50(x, i1)
			x = self.b51(x)
			
			# Final ConvTranspose Layer
			x = self.fullconv(x)
			
			return x
	class ASNeck(nn.Module):
		def __init__(self, in_channels, out_channels, projection_ratio=4):
			
			super().__init__()
			
			# Define class variables
			self.in_channels = in_channels
			self.reduced_depth = int(in_channels / projection_ratio)
			self.out_channels = out_channels
			
			self.dropout = nn.Dropout2d(p=0.1)
			
			self.conv1 = nn.Conv2d(in_channels = self.in_channels,
								   out_channels = self.reduced_depth,
								   kernel_size = 1,
								   stride = 1,
								   padding = 0,
								   bias = False)
			
			self.prelu1 = nn.PReLU()
			
			self.conv21 = nn.Conv2d(in_channels = self.reduced_depth,
									  out_channels = self.reduced_depth,
									  kernel_size = (1, 5),
									  stride = 1,
									  padding = (0, 2),
									  bias = False)
			
			self.conv22 = nn.Conv2d(in_channels = self.reduced_depth,
									  out_channels = self.reduced_depth,
									  kernel_size = (5, 1),
									  stride = 1,
									  padding = (2, 0),
									  bias = False)
			
			self.prelu2 = nn.PReLU()
			
			self.conv3 = nn.Conv2d(in_channels = self.reduced_depth,
									  out_channels = self.out_channels,
									  kernel_size = 1,
									  stride = 1,
									  padding = 0,
									  bias = False)
			
			self.prelu3 = nn.PReLU()
			
			self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
			self.batchnorm2 = nn.BatchNorm2d(self.out_channels)
			
		def forward(self, x):
			bs = x.size()[0]
			x_copy = x
			
			# Side Branch
			x = self.conv1(x)
			x = self.batchnorm(x)
			x = self.prelu1(x)
			
			x = self.conv21(x)
			x = self.conv22(x)
			x = self.batchnorm(x)
			x = self.prelu2(x)
			
			x = self.conv3(x)
					
			x = self.dropout(x)
			x = self.batchnorm2(x)
			
			# Main Branch
			
			if self.in_channels != self.out_channels:
				out_shape = self.out_channels - self.in_channels
				extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
				if torch.cuda.is_available():
					extras = extras.cuda()
				x_copy = torch.cat((x_copy, extras), dim = 1)
			
			# Sum of main and side branches
			x = x + x_copy
			x = self.prelu3(x)
			
			return x
	
	import cv2
	import sys
	import os
	from tqdm import tqdm
	import matplotlib.pyplot as plt
	from PIL import Image
	
	def create_class_mask(img, color_map, is_normalized_img=True, is_normalized_map=False, show_masks=False):
		"""
		Function to create C matrices from the segmented image, where each of the C matrices is for one class
		with all ones at the pixel positions where that class is present
		img = The segmented image
		color_map = A list with tuples that contains all the RGB values for each color that represents
					some class in that image
		is_normalized_img = Boolean - Whether the image is normalized or not
							If normalized, then the image is multiplied with 255
		is_normalized_map = Boolean - Represents whether the color map is normalized or not, if so
							then the color map values are multiplied with 255
		show_masks = Wherether to show the created masks or not
		"""

		if is_normalized_img and (not is_normalized_map):
			img *= 255

		if is_normalized_map and (not is_normalized_img):
			img = img / 255
		
		mask = []
		hw_tuple = img.shape[:-1]
		for color in color_map:
			color_img = []
			for idx in range(3):
				color_img.append(np.ones(hw_tuple) * color[idx])

			color_img = np.array(color_img, dtype=np.uint8).transpose(1, 2, 0)

			mask.append(np.uint8((color_img == img).sum(axis = -1) == 3))

		return np.array(mask)


	def loader(training_path, segmented_path, batch_size, h=512, w=512):
		"""
		The Loader to generate inputs and labels from the Image and Segmented Directory
		Arguments:
		training_path - str - Path to the directory that contains the training images
		segmented_path - str - Path to the directory that contains the segmented images
		batch_size - int - the batch size
		yields inputs and labels of the batch size
		"""

		filenames_t = os.listdir(training_path)
		total_files_t = len(filenames_t)
		
		filenames_s = os.listdir(segmented_path)
		total_files_s = len(filenames_s)
		
		assert(total_files_t == total_files_s)
		
		if str(batch_size).lower() == 'all':
			batch_size = total_files_s
		
		idx = 0
		while(1):
			batch_idxs = np.random.randint(0, total_files_s, batch_size)
				
			
			inputs = []
			labels = []
			
			for jj in batch_idxs:
				img = plt.imread(training_path + filenames_t[jj])
				#img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
				inputs.append(img)
				
				img = Image.open(segmented_path + filenames_s[jj])
				img = np.array(img)
				#img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
				labels.append(img)
			 
			inputs = np.stack(inputs, axis=2)
			inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)
			
			labels = torch.tensor(labels)
			
			yield inputs, labels


	def loader_cityscapes(txt_path, cityscapes_path, batch_size):
		"""
		The Loader to generate inputs and labels from the txt file
		Arguments:
		txt_path - str - Path to the txt file that contains the training images and segmented images path
		cityscapes_path - str - Cityscapes Path to the directory of Cityscapes image
		batch_size - int - the batch size
		yields inputs and labels of the batch size
		"""

		lines = open(txt_path, 'r').readlines()
		total_files = len(lines)

		images = []
		gts = []
		for line in lines:
			line = line.strip().split(" ")
			images.append(line[0])
			gts.append(line[1])

		if str(batch_size).lower() == 'all':
			batch_size = total_files
			while (1):
				batch_idxs = np.random.randint(0, total_files, batch_size)

				labels = []
				for jj in batch_idxs:
					img = Image.open(cityscapes_path + gts[jj])
					img = np.array(img)
					#img5 = scale_downsample(img, 0.5, 0.5)

					labels.append(img)

				labels = torch.tensor(labels)

				yield labels

		idx = 0
		while (1):
			batch_idxs = np.random.randint(0, total_files, batch_size)

			inputs = []
			labels = []
			for jj in batch_idxs:

				img = plt.imread(cityscapes_path + images[jj])
				#img5 = scale_downsample(img, 0.5, 0.5)
				inputs.append(img)

				img = Image.open(cityscapes_path + gts[jj])
				img = np.array(img)
				#img5 = scale_downsample(img, 0.5, 0.5)
				labels.append(img)

			inputs = np.stack(inputs, axis=2)
			inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)

			labels = torch.tensor(labels)

			yield inputs, labels



	def decode_segmap(image, cityscapes):
		Sky = [128, 128, 128]
		Building = [128, 0, 0]
		Column_Pole = [192, 192, 128]
		Road_marking = [255, 69, 0]
		Road = [128, 64, 128]
		Pavement = [60, 40, 222]
		Tree = [128, 128, 0]
		SignSymbol = [192, 128, 128]
		Fence = [64, 64, 128]
		Car = [64, 0, 128]
		Pedestrain = [64, 64, 0]
		Bicyclist = [0, 128, 192]

		# Json string data
		road = [128,64,128]          #0
		Lsidewalk = [244,35,232]     #1
		building = [70,70,70]        #2
		wall = [102,102,156]         #3
		fence = [190,153,153]        #4
		pole = [153,153,153]         #5
		traffic_light = [250,170,30] #6
		traffic_sign = [220,220,0]   #7
		vegetation = [107,142,35]    #8
		terrain = [152,251,152]      #9
		sky = [70,130,180]           #10
		person = [220,20,60]         #11
		Lrider = [255,0,0]           #12
		car = [0,0,142]              #13
		truck = [0,0,70]             #14
		bus = [0,60,100]             #15
		train = [0,80,100]           #16
		motorcycle = [0,0,230]       #17
		bicycle = [119,11,32]        #18


		if cityscapes:
			label_colors = np.array([road, Lsidewalk, building, wall, fence, pole, traffic_light, traffic_sign,
									 vegetation, terrain, sky, person, Lrider, car, truck, bus, train, motorcycle,
									 bicycle]).astype(np.uint8)
		else:
			label_colors = np.array([Sky, Building, Column_Pole, Road_marking, Road,
								  Pavement, Tree, SignSymbol, Fence, Car,
								  Pedestrain, Bicyclist]).astype(np.uint8)
		buffer = 0
		Bbuffer = 0
		ycount = 0
		xcount = 0
		same_count = 0
		Ccount = 0
		Count_reset = False
		obj_close = False
		
		if obj_close == True:
			for i in image:
				for j in i:
					if ycount == 0:
						ycount = 1
						buffer = j
						Bbuffer = j
					else:    
						if (buffer == j):
							image[xcount][ycount] = buffer
							same_count = same_count + 1
							Ccount = Ccount + 1
						else:
							if Ccount >= 1: #increase to filter out "classification distortions", higher then 20 decreases trust
								image[xcount][ycount - 1] = buffer
								same_count = 0

								if Count_reset == True:
									Ccount = 0
									Bbuffer = j
									Count_reset == False
								Count_reset = True
							else:
								while(same_count > 0):
									image[xcount][ycount - same_count] = Bbuffer                         
									same_count = same_count - 1

								if Bbuffer == j:
									same_count = Ccount
								Count_reset = False
							buffer = j
						ycount = ycount + 1    

				image[xcount][ycount - 1] = Bbuffer
				ycount = 0
				xcount = xcount + 1
		r = np.zeros_like(image).astype(np.uint8)
		g = np.zeros_like(image).astype(np.uint8)
		b = np.zeros_like(image).astype(np.uint8)
		
		for label in range(len(label_colors)):
				
				b[image == label] = label_colors[label, 0]
				g[image == label] = label_colors[label, 1]
				r[image == label] = label_colors[label, 2] 
				
		rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
		rgb[:, :, 0] = b
		rgb[:, :, 1] = g
		rgb[:, :, 2] = r
		
		#image: returns only the altered segmented image, cannot be used with 'cv2.showim()'
		#rgb: returns the colour classified image
		return rgb #Json_ex #rgb  #image

	def show_images(images, in_row=True):
		'''
		Helper function to show 3 images
		'''
		total_images = len(images)

		rc_tuple = (1, total_images)
		if not in_row:
			rc_tuple = (total_images, 1)

		#figure = plt.figure(figsize=(20, 10))
		for ii in range(len(images)):
			plt.subplot(*rc_tuple, ii+1)
			plt.title(images[ii][0])
			plt.axis('off')
			plt.imshow(images[ii][1])
			
		# plt.savefig("./Enet.png")
		Json_ex = json.dumps({"image" : images[1].tolist()})
		print(Json_ex)
		plt.show()
		

	def get_class_weights(loader, num_classes, c=1.02, isCityscapes=False):
		'''
		This class return the class weights for each class
		
		Arguments:
		- loader : The generator object which return all the labels at one iteration
				   Do Note: That this class expects all the labels to be returned in
				   one iteration
		- num_classes : The number of classes
		Return:
		- class_weights : An array equal in length to the number of classes
						  containing the class weights for each class
		'''
		if isCityscapes:
			labels = next(loader)
		else:
			_, labels = next(loader)
		all_labels = labels.flatten()
		all_len = len(all_labels)
		each_class = np.bincount(all_labels, minlength=num_classes)
		if isCityscapes:
			each_class = each_class[0:19]
			num = 0
			for i in each_class:
				num += i
			all_len = num
		prospensity_score = each_class / all_len
		class_weights = 1 / (np.log(c + prospensity_score))
		print("class_weights: ")
		print(class_weights)
		return class_weights


	def scale_downsample(img, kx, ky):
		rows = int(np.round(np.abs(img.shape[0] * kx)))
		cols = int(np.round(np.abs(img.shape[1] * ky)))

		if len(img.shape) == 3 and img.shape[2] >= 3:
			dist = np.zeros((rows, cols, img.shape[2]), img.dtype)
		else:
			dist = np.zeros((rows, cols), img.dtype)

		for y in range(rows):
			for x in range(cols):
				new_y = int((y + 1) / ky + 0.5) - 1
				new_x = int((x + 1) / kx + 0.5) - 1

				dist[y, x] = img[new_y, new_x]

		return dist
	
	def train(FLAGS):

		# Defining the hyperparameters
		device =  FLAGS.cuda
		batch_size = FLAGS.batch_size
		epochs = FLAGS.epochs
		lr = FLAGS.learning_rate
		print_every = FLAGS.print_every
		eval_every = FLAGS.eval_every
		save_every = FLAGS.save_every
		nc = FLAGS.num_classes
		wd = FLAGS.weight_decay
		ip = FLAGS.input_path_train
		lp = FLAGS.label_path_train
		ipv = FLAGS.input_path_val
		lpv = FLAGS.label_path_val


		train_mode = FLAGS.train_mode
		pretrain_model = FLAGS.pretrain_model
		cityscapes_path = FLAGS.cityscapes_path
		resume_model_path = FLAGS.resume_model_path
		print ('[INFO]Defined all the hyperparameters successfully!')

		# Get the class weights
		print ('[INFO]Starting to define the class weights...')
		if len(cityscapes_path):
			pipe = loader_cityscapes(ip, cityscapes_path, batch_size='all')
			class_weights = get_class_weights(pipe, nc, isCityscapes=True)
			#class_weights = np.array([3.03507951, 13.09507946, 4.54913664, 37.64795738, 35.78537802, 31.50943831, 45.88744201, 39.936759,
			#                          6.05101481, 31.85754823, 16.92219283, 32.07766734, 47.35907214, 11.34163794, 44.31105748, 45.81085476,
			#                          45.67260936, 48.3493813, 42.02189188])
		else:
			pipe = loader(ip, lp, batch_size='all')
			class_weights = get_class_weights(pipe, nc)
		print ('[INFO]Fetched all class weights successfully!')

		# Get an instance of the model
		if train_mode.lower() == 'encoder-decoder':
			enet = ENet(nc)
			if len(pretrain_model):
				checkpoint0 = torch.load(pretrain_model)
				pretrain_dict = checkpoint0['state_dict']
				enet_dict = enet.state_dict()
				pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in enet_dict}
				enet_dict.update(pretrain_dict)
				enet.load_state_dict(enet_dict)
				print('[INFO]Previous model Instantiated!')
		else:
			enet = ENet_encoder(nc)

		print ('[INFO]Model Instantiated!')

		enet = enet.to(device)

		# Define the criterion and the optimizer
		if len(cityscapes_path):
			criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device), ignore_index=255)
		else:
			criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

		optimizer = torch.optim.Adam(enet.parameters(), lr=lr, weight_decay=wd)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True, threshold=0.01)
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True,
		#                                                        threshold=0.005)
		print ('[INFO]Defined the loss function and the optimizer')

		# Training Loop starts
		print ('[INFO]Staring Training...')
		print ()

		train_losses = []
		eval_losses = []


		if len(cityscapes_path):
			# Cityscapes Dataset
			bc_train = 2975 // batch_size
			bc_eval = 500 // batch_size

			pipe = loader_cityscapes(ip, cityscapes_path, batch_size)
			eval_pipe = loader_cityscapes(ipv, cityscapes_path, batch_size)
		else:
			# CamVid Dataset
			bc_train = 367 // batch_size
			bc_eval = 101 // batch_size

			pipe = loader(ip, lp, batch_size)
			eval_pipe = loader(ipv, lpv, batch_size)

		epoch = 1
		if len(resume_model_path):
			checkpoint1 = torch.load(resume_model_path)
			epoch = checkpoint1['epochs'] + 1
			enet.load_state_dict(checkpoint1['state_dict'])

		epochs = epochs
				
		for e in range(epoch, epochs+1):
				
			train_loss = 0
			print ('-'*15,'Epoch %d' % e, '-'*15)

			enet.train()
			
			for _ in tqdm(range(bc_train)):
				X_batch, mask_batch = next(pipe)
				
				#assert (X_batch >= 0. and X_batch <= 1.0).all()
				
				X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

				optimizer.zero_grad()

				out = enet(X_batch.float())

				loss = criterion(out, mask_batch.long())
				loss.backward()
				optimizer.step()

				train_loss += loss.item()

				
			print ()
			train_losses.append(train_loss)
			
			if (e+1) % print_every == 0:
				print ('Epoch {}/{}...'.format(e, epochs),
						'Loss {:6f}'.format(train_loss))

			scheduler.step(train_loss)
			
			if e % eval_every == 0:
				with torch.no_grad():
					enet.eval()
					
					eval_loss = 0
					
					for _ in tqdm(range(bc_eval)):
						inputs, labels = next(eval_pipe)

						inputs, labels = inputs.to(device), labels.to(device)
						out = enet(inputs)
						
						loss = criterion(out, labels.long())

						eval_loss += loss.item()

					print ()
					print ('Loss {:6f}'.format(eval_loss))
					
					eval_losses.append(eval_loss)

			if e % save_every == 0:
				checkpoint = {
					'epochs' : e,
					'state_dict' : enet.state_dict()
				}
				if train_mode.lower() == 'encoder-decoder':
					torch.save(checkpoint,
							   './logs/ckpt-enet-{}-{}-{}.pth'.format(e, optimizer.state_dict()['param_groups'][0]['lr'],
																	  train_loss))
				else:
					torch.save(checkpoint,
							   './logs/ckpt-enet_encoder-{}-{}-{}.pth'.format(e, optimizer.state_dict()['param_groups'][0]['lr'],
																			  train_loss))
				print ('Model saved!')

			print ('Epoch {}/{}...'.format(e+1, epochs),
				   'Total Mean Loss: {:6f}'.format(sum(train_losses) / epochs))

		print ('[INFO]Training Process complete!')
	
	def get_extrinsics(src, dst):
		extrinsics = src.get_extrinsics_to(dst)
		R = np.reshape(extrinsics.rotation, [3,3]).T
		T = np.array(extrinsics.translation)
		return (R, T)

	"""
	Returns a camera matrix K from librealsense intrinsics
	"""
	def camera_matrix(intrinsics):
		return np.array([[intrinsics.fx,             0, intrinsics.ppx],
						 [            0, intrinsics.fy, intrinsics.ppy],
						 [            0,             0,              1]])

	"""
	Returns the fisheye distortion from librealsense intrinsics
	"""
	def fisheye_distortion(intrinsics):
		return np.array(intrinsics.coeffs[:4])

	# Set up a mutex to share data between threads 
	from threading import Lock
	frame_mutex = Lock()
	frame_data = {"left"  : None,
				  "right" : None,
				  "timestamp_ms" : None
				  }

	"""
	This callback is called on a separate thread, so we must use a mutex
	to ensure that data is synchronized properly. We should also be
	careful not to do much work on this thread to avoid data backing up in the
	callback queue.
	"""
	def callback(frame):
		global frame_data
		if frame.is_frameset():
			frameset = frame.as_frameset()
			f1 = frameset.get_fisheye_frame(1).as_video_frame()
			f2 = frameset.get_fisheye_frame(2).as_video_frame()
			left_data = np.asanyarray(f1.get_data())
			right_data = np.asanyarray(f2.get_data())
			ts = frameset.get_timestamp()
			frame_mutex.acquire()
			frame_data["left"] = left_data
			frame_data["right"] = right_data
			frame_data["timestamp_ms"] = ts
			frame_mutex.release()
	
	def test(FLAGS):
		# cap = cv2.VideoCapture(0)

		 # Declare RealSense pipeline, encapsulating the actual device and sensors
		pipe = rs.pipeline()

		# Build config object and stream everything
		cfg = rs.config()

		# Start streaming with our callback
		pipe.start(cfg, callback)
		# Set up an OpenCV window to visualize the results
			
		WINDOW_TITLE = 'Realsense'
		cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

			# Configure the OpenCV stereo algorithm. See
			# https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
			# description of the parameters
		window_size = 5
		min_disp = 0
			# must be divisible by 16
		num_disp = 112 - min_disp
		max_disp = min_disp + num_disp
		stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
										numDisparities = num_disp,
										blockSize = 16,
										P1 = 8*3*window_size**2,
										P2 = 32*3*window_size**2,
										disp12MaxDiff = 1,
										uniquenessRatio = 10,
										speckleWindowSize = 100,
										speckleRange = 32)

			# Retreive the stream and intrinsic properties for both cameras
		profiles = pipe.get_active_profile()
		streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
					"right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
		intrinsics = {"left"  : streams["left"].get_intrinsics(),
						"right" : streams["right"].get_intrinsics()}

			# Print information about both cameras
		print("Left camera:",  intrinsics["left"])
		print("Right camera:", intrinsics["right"])

			# Translate the intrinsics from librealsense into OpenCV
		K_left  = camera_matrix(intrinsics["left"])
		D_left  = fisheye_distortion(intrinsics["left"])
		K_right = camera_matrix(intrinsics["right"])
		D_right = fisheye_distortion(intrinsics["right"])
		(width, height) = (intrinsics["left"].width, intrinsics["left"].height)

			# Get the relative extrinsics between the left and right camera
		(R, T) = get_extrinsics(streams["left"], streams["right"])

			# We need to determine what focal length our undistorted images should have
			# in order to set up the camera matrices for initUndistortRectifyMap.  We
			# could use stereoRectify, but here we show how to derive these projection
			# matrices from the calibration and a desired height and field of view

		stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
		stereo_height_px = 300          # 300x300 pixel stereo output
		stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

			# We set the left rotation to identity and the right rotation
			# the rotation between the cameras
		R_left = np.eye(3)
		R_right = R

			# The stereo algorithm needs max_disp extra pixels in order to produce valid
			# disparity on the desired output region. This changes the width, but the
			# center of projection should be on the center of the cropped image
		stereo_width_px = stereo_height_px + max_disp
		stereo_size = (stereo_width_px, stereo_height_px)
		stereo_cx = (stereo_height_px - 1)/2 + max_disp
		stereo_cy = (stereo_height_px - 1)/2

			# Construct the left and right projection matrices, the only difference is
			# that the right projection matrix should have a shift along the x axis of
			# baseline*focal_length
		P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
							[0, stereo_focal_px, stereo_cy, 0],
							[0,               0,         1, 0]])
		P_right = P_left.copy()
		P_right[0][3] = T[0]*stereo_focal_px

			# Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
			# since we will crop the disparity later
		Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
						[0, 1,       0, -stereo_cy],
						[0, 0,       0, stereo_focal_px],
						[0, 0, -1/T[0], 0]])

		# Create an undistortion map for the left and right camera which applies the
		# rectification and undoes the camera distortion. This only has to be done
		# once
		m1type = cv2.CV_32FC1
		(lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
		(rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
		undistort_rectify = {"left"  : (lm1, lm2),
								"right" : (rm1, rm2)}

		mode = "stack"
		
		# Check if the pretrained model is available
		if not FLAGS.m.endswith('.pth'):
			raise RuntimeError('Unknown file passed. Must end with .pth')

		h = FLAGS.resize_height
		w = FLAGS.resize_width
		nc = FLAGS.num_classes
		test_mode = FLAGS.test_mode

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")##dawson add gpu mode
		checkpoint = torch.load(FLAGS.m,  map_location=lambda storage, loc: storage.cpu())#"cpu")

		enet = ENet(nc)
		enet.to(device) ##dawson add gpu mode
		enet.load_state_dict(checkpoint['state_dict'])
		while(True):
			# Check if the camera has acquired any frames
			frame_mutex.acquire()
			valid = frame_data["timestamp_ms"] is not None
			frame_mutex.release()

			# If frames are ready to process

			if valid:
				# Hold the mutex only long enough to copy the stereo frames
				frame_mutex.acquire()
				frame_copy = {"left"  : frame_data["left"].copy(),
									  "right" : frame_data["right"].copy()}
				frame_mutex.release()

				# Undistort and crop the center of the frames
				center_undistorted = {"left" : cv2.remap(src = frame_copy["left"],
												map1 = undistort_rectify["left"][0],
												map2 = undistort_rectify["left"][1],
												interpolation = cv2.INTER_LINEAR),
										"right" : cv2.remap(src = frame_copy["right"],
												map1 = undistort_rectify["right"][0],
												map2 = undistort_rectify["right"][1],
												interpolation = cv2.INTER_LINEAR)}

				# compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
				disparity = stereo.compute(center_undistorted["left"], center_undistorted["right"]).astype(np.float32) / 16.0

				# re-crop just the valid part of the disparity
				disparity = disparity[:,max_disp:]

				# convert disparity to 0-255 and color it
				disp_vis = 255*(disparity - min_disp)/ num_disp
				disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1), cv2.COLORMAP_JET)
				color_image = cv2.cvtColor(center_undistorted["left"][:,max_disp:], cv2.COLOR_GRAY2RGB)
				
				tmg_ = color_image
				tmg_ = cv2.resize(tmg_, (480,360), interpolation= cv2.INTER_NEAREST)
				tmg = torch.tensor(tmg_).unsqueeze(0).float()
				tmg = tmg.transpose(2, 3).transpose(1, 2)

				with torch.no_grad():
					out1 = enet(tmg.cpu().float()).squeeze(0)##dawson add gpu mode


				b_ = out1.data.max(0)[1].cpu().numpy()
				if test_mode.lower() == 'cityscapes':
					decoded_segmap = decode_segmap(b_, True)
				else:
					decoded_segmap = decode_segmap(b_, False)

				cv2.imshow("Realsense", tmg_)
				cv2.imshow("test", decoded_segmap)
					#print(decoded_segmap)
				images = {
					0 : ['Input Image', tmg_],
					1 : ['Predicted Segmentation', b_],
				}
				show_images(images)        
			key = cv2.waitKey(10)
			if key == 27:
				break
	   
		pipe.stop()
		cv2.destroyAllWindows()
	
	import argparse
	if __name__ == '__main__':
		parser = argparse.ArgumentParser()

		parser.add_argument('-m',
							type=str,
							default='F:/EP/city.pth',#F:/EP/vid.pth   F:/EP/city.pth #veranderen naar momentele path
							help='The path to the pretrained enet model')

		parser.add_argument('-i', '--image-path',
							type=str,
							default='F:/EP/road.jpg', #veranderen naar momentele path
							help='The path to the image to perform semantic segmentation')

		parser.add_argument('-rh', '--resize-height',
							type=int,
							default=1280,#1024,
							help='The height for the resized image')

		parser.add_argument('-rw', '--resize-width',
							type=int,
							default=720,#512,
							help='The width for the resized image')

		parser.add_argument('-lr', '--learning-rate',
							type=float,
							default=5e-3,
							help='The learning rate')

		parser.add_argument('-bs', '--batch-size',
							type=int,
							default=10,
							help='The batch size')

		parser.add_argument('-wd', '--weight-decay',
							type=float,
							default=2e-4,
							help='The weight decay')

		parser.add_argument('-c', '--constant',
							type=float,
							default=1.02,
							help='The constant used for calculating the class weights')

		parser.add_argument('-e', '--epochs',
							type=int,
							default=102,
							help='The number of epochs')

		parser.add_argument('-nc', '--num-classes',
							type=int,
							default=19, #12   19
							help='The number of classes')

		parser.add_argument('-se', '--save-every',
							type=int,
							default=10,
							help='The number of epochs after which to save a model')

		parser.add_argument('-iptr', '--input-path-train',
							type=str,
							default='F:/EP/CamVid/train/', #veranderen naar momentele path
							help='The path to the input dataset')

		parser.add_argument('-lptr', '--label-path-train',
							type=str,
							default='F:/EP/CamVid/trainannot/', #veranderen naar momentele path
							help='The path to the label dataset')

		parser.add_argument('-ipv', '--input-path-val',
							type=str,
							default='F:/EP/CamVid/val/', #veranderen naar momentele path
							help='The path to the input dataset')

		parser.add_argument('-lpv', '--label-path-val',
							type=str,
							default='F:/EP/CamVid/valannot/', #veranderen naar momentele path
							help='The path to the label dataset')

		parser.add_argument('-iptt', '--input-path-test',
							type=str,
							default='F:/EP/CamVid/test/', #veranderen naar momentele path
							help='The path to the input dataset')

		parser.add_argument('-lptt', '--label-path-test',
							type=str,
							default='F:/EP/CamVid/testannot/', #veranderen naar momentele path
							help='The path to the label dataset')

		parser.add_argument('-pe', '--print-every',
							type=int,
							default=1,
							help='The number of epochs after which to print the training loss')

		parser.add_argument('-ee', '--eval-every',
							type=int,
							default=10,
							help='The number of epochs after which to print the validation loss')

		parser.add_argument('--cuda',
							type=bool,
							default=False,
							help='Whether to use cuda or not')

		parser.add_argument('--mode',
							choices=['train', 'test'],
							default='test',
							help='Whether to train or test')

		parser.add_argument('--test_mode',
							choices=['cityscapes', 'camvid'],
							default='cityscapes',
							help='Whether to test cityscape model or camvid model')

		parser.add_argument('--train_mode',
							choices=['encoder-decoder', 'encoder'],
							default='encoder-decoder',
							help='Select to train mode of Enet')

		parser.add_argument('--pretrain_model',
							type=str,
							default='',
							help='Import previous train model of encoder ENet')

		parser.add_argument('--cityscapes_path',
							type=str,
							default='',
							help='Cityscapes Path to the directory of Cityscapes image')

		parser.add_argument('--resume_model_path',
							type=str,
							default='',
							help='Model path to resume training')

		
		FLAGS, unparsed = parser.parse_known_args()

		FLAGS.cuda = torch.device('cuda:0' if torch.cuda.is_available() and FLAGS.cuda else 'cpu')

		if FLAGS.mode.lower() == 'train':
			train(FLAGS)
		elif FLAGS.mode.lower() == 'test':
			test(FLAGS)
		else:
			raise RuntimeError('Unknown mode passed. \n Mode passed should be either of "train" or "test"')