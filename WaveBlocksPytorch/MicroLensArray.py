import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MicroLensArray(ob.OpticBlock):
	"""Class that simulates a micro lens array formed by periodic structures"""
	def __init__(self, optic_config, members_to_learn, focal_length, pixel_size, image_shape, block_shape, block_separation, block_image=None, block_offset_odd_row=0, MLA_image=None):
		super(MicroLensArray, self).__init__(optic_config, members_to_learn)
		self.focal_length = nn.Parameter(torch.tensor(focal_length, dtype=torch.float32, requires_grad=True))
		self.pixel_size = nn.Parameter(torch.tensor(pixel_size, dtype=torch.float32, requires_grad=True))
		self.image_shape = torch.tensor(image_shape, dtype=torch.int32)
		self.block_shape = torch.tensor(block_shape, dtype=torch.int32)
		self.block_separation = torch.tensor(block_separation, dtype=torch.int32)
		self.block_offset_odd_row = torch.tensor(block_offset_odd_row, dtype=torch.int32)
		self.block_image = None
		self.MLA_image = MLA_image
		self.psf_space_variant = nn.Parameter(None)
		
		# check if user provides image for MLA
		if MLA_image != None:
		    self.MLA_image = MLA_image
		else:
			if block_image == None:
				self.block_image = nn.Parameter(self.compute_block_image(), requires_grad=True)
			self.MLA_image = nn.Parameter(self.compute_full_image(), requires_grad=True)

		
		
	def compute_block_image(self):
		halfSize = torch.ceil(self.block_shape.float()/2.0)
		# compute image inside a block, give the transmitance function (trans_function)
		u = np.sort(np.concatenate((np.arange(start = 0, stop = -self.pixel_size*halfSize[0], step = -self.pixel_size, dtype = 'float'),np.arange(start = self.pixel_size, stop = self.pixel_size*halfSize[0], step = self.pixel_size, dtype = 'float'))))
		X, Y = np.meshgrid(u, u)
		X = torch.from_numpy(X).float()
		Y = torch.from_numpy(Y).float()
		xL2norm = torch.mul(X,X) + torch.mul(Y,Y)

		# compute MLA transmintance exp(-1i*k/(2*focalLenght)*xL2norm)
		block_image = ob.expComplex(torch.cat((torch.zeros(xL2norm.shape).unsqueeze(2), -self.optic_config.k * xL2norm.unsqueeze(2) / (2*self.focal_length)), 2))
		return block_image
	
	def compute_full_image(self):
		# replicate the block image such that its larger than the PSF size
		nRepetitions = torch.ceil(torch.div(self.image_shape.float(), self.block_shape.float())) + 2
		nRepetitions = torch.add(nRepetitions, 1-torch.remainder(nRepetitions,2))
		fullMLAImage = self.block_image.repeat(int(nRepetitions[0]), int(nRepetitions[1]), 1)
		return fullMLAImage
	
	def set_space_variant_psf(self, psf):
		self.psf_space_variant = nn.Parameter(psf, requires_grad=psf.requires_grad)
	
	def get_space_variant_psf(self):
		return self.psf_space_variant

	def forward(self, psf, input_sampling):
		# resample might be needed if input_sampling rate is different than sampling_rate (aka phase-mask pixel size)
		assert input_sampling==self.pixel_size, "MLA forward: PSF sampling rate and MLA pixel size should be the same"
		psfShape = torch.tensor(psf.shape[0:4]).int()

		# half psf shape
		halfPSFShape = torch.ceil(psfShape[2:4].float()/2.0).int()
		# half size of the ML
		mlHalfShape = torch.ceil(self.block_shape.float()/2.0).int()
		# half size of MLA image
		mlaHalfShape = torch.ceil(torch.tensor(self.MLA_image.shape[0:2]).float()/2.0).int()

		# define output PSF
		psfOut = torch.zeros((psfShape[0], psfShape[1], mlHalfShape[0], mlHalfShape[1], psfShape[2], psfShape[3], 2), dtype=torch.float32).to(psf.device)

		# iterate positions inside the central ML, as the psf diffracts different when placed at different spots of the MLA
		for x1 in range(mlHalfShape[0]):
			x1Shift = mlHalfShape[0]-x1
			for x2 in range(mlHalfShape[1]):
				# crop translated MLA image, to have the element x,y at the center
				x2Shift = mlHalfShape[1]-x2
				
				transmitanceCurrentXY = self.MLA_image[\
					(mlaHalfShape[0]-halfPSFShape[0]-x1Shift+1):(mlaHalfShape[0]+halfPSFShape[0]-x1Shift), \
					(mlaHalfShape[1]-halfPSFShape[1]-x2Shift+1):(mlaHalfShape[1]+halfPSFShape[1]-x2Shift), :]
				
				# multiply by all depths
				# replicate transmitance by nDepths
				transmitanceCurrentXYZ = transmitanceCurrentXY.unsqueeze(0).unsqueeze(0).repeat(psfShape[0], psfShape[1],1,1,1)
				psfOut[:,:,x1,x2,:,:,:] = ob.mulComplex(transmitanceCurrentXYZ, psf)

		
		# output is ordered as [depths, x, y, Nnum[0], Nnum[1], complex]
		return psfOut