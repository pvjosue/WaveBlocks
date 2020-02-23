import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np

class DiffractiveElement(ob.OpticBlock):
	"""Class that simulates a diffractive element such as an apperture, phase mask, coded apperture, etc."""
	def __init__(self, optic_config, members_to_learn, sampling_rate, apperture_size,\
		 function_img, max_phase_shift=5*math.pi, element_y_angle=45.0, center_offset=[0.0,0.0], correction_img=None):
		# apperture_shape is the 2D size of the element in pixels
        # sampling_rate is the pixel size of the diffractive element
		super(DiffractiveElement, self).__init__(optic_config, members_to_learn)
		self.sampling_rate = nn.Parameter(torch.tensor([sampling_rate], dtype=torch.float32, requires_grad=True))
		self.apperture_size = nn.Parameter(torch.tensor(apperture_size, dtype=torch.float32, requires_grad=True))
		self.metric_size = nn.Parameter(sampling_rate * self.apperture_size)
		self.element_y_angle = nn.Parameter(torch.tensor(element_y_angle, dtype=torch.float32, requires_grad=True))
		self.center_offset = nn.Parameter(torch.tensor(center_offset, dtype=torch.float32, requires_grad=True))
		self.max_phase_shift = nn.Parameter(torch.tensor(max_phase_shift, dtype=torch.float32, requires_grad=True))
		self.constant_phase = nn.Parameter(torch.tensor(max_phase_shift/2, dtype=torch.float32, requires_grad=True))
		self.correction_img = nn.Parameter(correction_img)
        # phase function is a lambda function to create in the image based on its x,y coords
		if function_img is not None:
		    self.function_img = nn.Parameter(function_img)
		else:
 			self.function_img = nn.Parameter(self.max_phase_shift/2 - 0.5 + torch.rand((1,1,apperture_size[0], apperture_size[1]), requires_grad=True))
	
	def forward(self, fieldIn, input_sampling):
		# self.function_img = self.function_img.clamp(0,self.max_phase_shift)
		inShape = torch.tensor(fieldIn.shape)

		if self.correction_img.ndim != 1:
			function_img = self.correction_img.detach() + self.function_img
		else:
			function_img = self.function_img #- self.function_img.min()
		# function_img /= function_img.max() * self.max_phase_shift
		

		# translate the image to the center of the SLM 
		if self.center_offset.sum()!=0.0:
			padOffsetSize = 2*[abs(int(self.center_offset[1]))]+2*[abs(int(self.center_offset[0]))]
			function_img_size = function_img.shape
			function_img_padded = F.pad(function_img, padOffsetSize, 'reflect')
			# crop ROI
			function_img = function_img_padded[:,:,(function_img_padded.shape[-2]//2-int(self.center_offset[0])-function_img_size[-2]//2):\
				(function_img_padded.shape[-2]//2-int(self.center_offset[0])-function_img_size[-2]//2+function_img_size[-2]),\
					(function_img_padded.shape[-1]//2-int(self.center_offset[1])-function_img_size[-1]//2):\
				(function_img_padded.shape[-1]//2-int(self.center_offset[1])-function_img_size[-1]//2+function_img_size[-1])]

		self.function_img_with_offset = function_img

		# reshape and mask either input field or local image to match sampling_rate
		resample = input_sampling!=self.sampling_rate or inShape[2]!=function_img.shape[2] or inShape[3]!=function_img.shape[3]
		field = fieldIn
		paddInput = False
		if resample:
			# compute ratio between input and phase image, to have a matching pixels to sampling
			ratio_self_input = self.sampling_rate/input_sampling
			
			# rotate phase mask in the specified angle
			ratio_self_input_y = ratio_self_input*torch.cos(torch.tensor(np.radians(self.element_y_angle.item())))
			ks = [int(1/ratio_self_input),int(1/ratio_self_input_y)]
			function_img = F.avg_pool2d(self.function_img,kernel_size=ks,padding= [ks[0]//2,ks[1]//2])
			# resample funciton image
			# function_img = F.interpolate(function_img, scale_factor=(ratio_self_input_x,ratio_self_input_y), mode='bilinear', align_corners=False)
			newSize = function_img.shape[2:4]
			# pad input to match size of phase mask, in case that the input is smaller
			if field.shape[-3] < function_img.shape[-2] or field.shape[-2] < function_img.shape[-1]:
				paddInput = True
				padSize = list([0, 0, (newSize[1]-inShape[3])//2, (newSize[1]-inShape[3])//2, (newSize[0]-inShape[2])//2, (newSize[0]-inShape[2])//2])
				field = F.pad(fieldIn, tuple(padSize), "constant", 0)
			else:
				padSize = list([(inShape[3]-newSize[1])//2, (inShape[3]-newSize[1])//2, (inShape[2]-newSize[0])//2, (inShape[2]-newSize[0])//2])
				function_img = F.pad(function_img, tuple(padSize), "constant", 0)
			# check if newSize is even, as the padding would be different
			# if newSize[0]%2==0:
			# 	padSize[5] -= 1
			# if newSize[1]%2==0:
			# 	padSize[3] -= 1
			
			
			# todo: avoid next interpolation by computing the size correctly in the first place
			function_img = F.interpolate(function_img, size=field.shape[-3:-1], mode='bilinear', align_corners=False)

		# quantize phase-mask given the posible physical values
		# todo

        # element wise multiplication with phase element
		function_img = function_img.unsqueeze(4).repeat(inShape[0],inShape[1],1,1,2)


		# clamp to 8 bits, posible in Phase-mask
		# min_pm_step = self.max_phase_shift/256.0
		# function_img = ((function_img/min_pm_step).int() * min_pm_step).float()
		
		# set real part to zero, as phase-mask only affects phase
		function_img[:,:,:,:,0] = 0.0
		# function_img = ob.expComplex(function_img)

		# resample = False
		# paddInput = False
		# function_img = torch.cat(2*[self.constant_phase.unsqueeze(0)])
		# function_img[0] = 0.0
		# function_img = ob.expComplex(function_img.unsqueeze(0))
		# field = fieldIn

        # resample might be needed if input_sampling rate is different than sampling_rate (aka phase-mask pixel size)
		output = ob.mulComplex(field, function_img)
		if resample and paddInput:
			output = output[:,:,padSize[4]:output.shape[2]-padSize[5],padSize[2]:output.shape[3]-padSize[3],:]
		# plt.figure()
		# plt.subplot(1,3,1)
		# plt.imshow(function_img[0,0,:,:,1].cpu().detach().numpy())
		# plt.title(str(function_img[0,0,:,:,1].cpu().detach().numpy().sum()))
		# plt.subplot(1,3,2)
		# plt.imshow(fieldIn[0,0,:,:,1].cpu().detach().numpy())
		# plt.title(str(field[0,0,:,:,1].cpu().detach().numpy().sum()))
		# plt.subplot(1,3,3)
		# plt.imshow(output[0,0,:,:,1].cpu().detach().numpy())
		# plt.title(str(output[0,0,:,:,1].cpu().detach().numpy().sum()))
		# plt.show()
		
		return output.contiguous()
