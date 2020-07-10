import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Camera(ob.OpticBlock):
	"""Class that simulates a Camera, and computes the intensity in each of it pixels given an incoming wave-front"""
	def __init__(self, optic_config, members_to_learn, pixel_size):
		# apperture_shape is the 2D size of the element in pixels
        # sampling_rate is the pixel size of the diffractive element
		super(Camera, self).__init__(optic_config, members_to_learn)
		self.pixel_size = nn.Parameter(torch.tensor([pixel_size], dtype=torch.float32))
		    
	def forward(self, input, psf, MLA=None):
		#        """This function either convolves the current PSF with a given object, or propagates the PSF through a periodic_element like a MicroLensArray, returning a 5D PSF"""
		#assert isinstance(psf, ob.PSF), "ob.PSF object needed to compute an image at the camera sensor"
		
		# Check if PSF is in complex mode and convert to intensity
		if psf.shape[-1]==2:
			psfAtSensor = ob.absSquareComplex(psf)
		else:
			psfAtSensor = psf
			
		psfNormalized = self.normalize_psf(psfAtSensor)
		if isinstance(MLA, ob.MicroLensArray):
			output = self.apply_space_variant_psf(input, psfNormalized, MLA)
		else:
			output = self.apply_space_invariant_psf(input, psfNormalized)
		return output,psfNormalized

	def apply_space_variant_psf(self, input, psf, MLA):
		# this function computes the PSF when passed through a MLA or any space variant element
		inputShape = torch.tensor(input.shape, dtype=torch.int32)
		inputHalfShape = inputShape[2:4]//2 + 1
		MLShape = MLA.block_shape
		MLhalfShape = MLShape//2 + 1

		# update PSF in MLA
		#MLA.set_space_variant_psf(psfNormalized)

		nDepths = inputShape[1].item()
		
		# padd input until have a full MLA number inside image and a MLA in the senter
		# padAmount = (torch.mul(MLShape, torch.ceil(torch.div(inputShape[2:4], MLShape.float()))) - inputShape[2:4]).int()
		# padSize = (padAmount[0].item()//2, padAmount[0].item()-padAmount[0].item()//2, padAmount[1].item()//2, padAmount[1].item()-padAmount[1].item()//2)
		
		
		residualsLowerBorder = torch.mul(1+torch.floor((inputHalfShape.float()-MLhalfShape.float()) / MLShape.float()), MLShape.float()) - (inputHalfShape-MLhalfShape).float()
		
		padSize = (int(residualsLowerBorder[1].item()), int(residualsLowerBorder[1].item()), int(residualsLowerBorder[0].item()), int(residualsLowerBorder[0].item()))
		paddedInput = torch.nn.functional.pad(input, padSize, "constant", 0)

		paddedShape = torch.tensor(paddedInput.shape,dtype=torch.int32)
		paddedCenter = paddedShape[2:4]//2 #todo: check this
		# compute views to group toguether all elements with the same position relative to the MLAs centers 
		out_channel = paddedShape[1]*(MLShape[0]**2)
		out_h = paddedShape[2]//MLShape[0]
		out_w = paddedShape[2]//MLShape[1]
		fm_view = paddedInput.contiguous().view(paddedShape[0], paddedShape[1], out_h, MLShape[0], out_w, MLShape[1])

		# pad again to match output size of convolution with input size
		padSize2 = (0,0,1,1,0,0,1,1)
		fm_view = torch.nn.functional.pad(fm_view, padSize2, "constant", 0)
		

		final_image = torch.zeros(input.shape, dtype=torch.float32).to(input.device)
		for x1 in range(MLShape[0]):
			flipX1 = True if x1 >= MLhalfShape[0].item() else False
			x1_quarter = MLShape[0].item() - x1 - 1 if x1 >= MLhalfShape[0].item() else x1
			for x2 in range(MLShape[1]):
				currInput = fm_view[:,:,:,x1,:,x2]
				if currInput.sum() == 0.0:
					continue

				flipX2 = True if x2 >= MLhalfShape[1].item() else False
				x2_quarter = MLShape[1].item() - x2 - 1 if x2 >= MLhalfShape[1].item() else x2
				# fetch correct PSF pattern
				curr_psf = psf[:,:,x1_quarter,x2_quarter,:,:]

				# flip if necesary
				curr_psf = curr_psf if flipX1 == False else curr_psf.flip(2)
				curr_psf = curr_psf if flipX2 == False else curr_psf.flip(3)
				
				# convolve view with corrsponding PSF
				curr_out = nn.functional.conv_transpose2d(currInput, \
					 curr_psf.permute(1,0,2,3), stride=MLShape[0].item(), padding=curr_psf.shape[3]//2 + MLhalfShape[0].item(), groups=nDepths)
				
				
				# crop current sensor response to be centered at the current x1,x2
				offsetLowBoundery = paddedCenter-inputHalfShape - (torch.tensor([x1,x2],dtype=torch.int32)-MLhalfShape)
				offsetHighBoundery = paddedCenter+inputHalfShape - (torch.tensor([x1,x2],dtype=torch.int32)-MLhalfShape) - 1
				curr_out_cropped = curr_out[:,:,offsetLowBoundery[0]:offsetHighBoundery[0], offsetLowBoundery[1]:offsetHighBoundery[1]]
				
				# accumulate output image
				final_image = torch.add(final_image, curr_out_cropped)
				# print(str(x1)+' '+str(x2))

		# sum convolutions per depth
		final_image = final_image.sum(1).unsqueeze(1)
		return final_image

    
	def apply_space_invariant_psf(self, realObject, psf):
		# check for same number of depths
		assert psf.shape[1] == realObject.shape[1], "Different number of depths in PSF and object"
		
		# unsqueeze psf to match (out_channels, groups/in_channels,kH,kW) deffinition
		psf = psf.permute(1,0,2,3)
		# perform group convolution, in order to convolve each depth with the correspondent 
		# object plane     
		volConvolved = torch.nn.functional.conv2d(realObject, psf, padding=psf.shape[2]//2, groups=realObject.shape[1])
		return volConvolved.sum(1).unsqueeze(1)
	

	def normalize_psf(self, psf):
		
		if psf.ndim==6: # LF psf
			for nDepth in range(psf.shape[1]):
				for x1 in range(psf.shape[2]):
					for x2 in range(psf.shape[3]):
						psf[0,nDepth, x1, x2, :, :] /= psf[0,nDepth, x1, x2, :, :].sum()
		else:
			for nDepth in range(psf.shape[1]):
				psf[0,nDepth,:,:] /= psf[0,nDepth,:,:].sum()
		return psf