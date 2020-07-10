import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Lens(ob.OpticBlock):
	"""Class that simulates a wave propagation through a lens, either from object plane to image_plane or from focal plane to back focal plane"""
	def __init__(self, optic_config, members_to_learn, focal_length, sampling_rate, apperture_width, field_size, obj_distance, img_distance):
		
		super(Lens, self).__init__(optic_config, members_to_learn)
		self.focal_length = nn.Parameter(torch.tensor([focal_length], dtype=torch.float32, requires_grad=True))
		self.sampling_rate = nn.Parameter(torch.tensor([sampling_rate], dtype=torch.float32, requires_grad=True))
		self.apperture_width = nn.Parameter(torch.tensor([apperture_width], dtype=torch.float32, requires_grad=True))
		self.field_size = nn.Parameter(torch.tensor([field_size], dtype=torch.float32, requires_grad=True))
		self.obj_distance = nn.Parameter(torch.tensor([obj_distance], dtype=torch.float32, requires_grad=True))
		self.img_distance = nn.Parameter(torch.tensor([img_distance], dtype=torch.float32, requires_grad=True))

		# compute input coefficient (scaling)
		U1C1 = ob.expComplex(torch.tensor([0.,optic_config.k * self.focal_length])) / optic_config.PSF_config.wvl / self.focal_length
		self.coefU1minus = nn.Parameter(ob.mulComplex( U1C1, torch.tensor([0.,-1])))
        
		# compute cuttoff frequency and mask
		M = field_size
		
		#source sample interval
		dx1 = self.field_size/M

		# obs sidelength
		self.L = self.optic_config.PSF_config.wvl*self.focal_length/dx1	

		
		# cutoff frequency
		f0 = self.apperture_width/(self.optic_config.PSF_config.wvl*self.focal_length)
		L = M*self.sampling_rate
		#source sample interval
		Fs = 1/self.sampling_rate
		dFx = Fs/M
		u = np.sort(np.concatenate((np.arange(start = 0, stop = -Fs/2, step = -dFx, dtype = 'float'),np.arange(start = dFx, stop = Fs/2, step = dFx, dtype = 'float'))))

		X, Y = np.meshgrid(u, u)
		X = torch.from_numpy(X).float()
		Y = torch.from_numpy(Y).float()
		rho = torch.sqrt(torch.mul(X,X)+torch.mul(Y,Y))
		mask = ~(rho / f0).ge(torch.tensor([0.5]))
		maskBW = mask.unsqueeze(0).unsqueeze(0).float()

		mask = torch.mul((f0 - rho) , maskBW)
		mask /= mask.max()

		# mask1 = F.pad(mask,2*[mask.shape[-1]//2]+2*[mask.shape[-2]//2])
		# mask2 = F.conv2d(mask1,mask)
		# mask2 = mask2[:,:,mask2.shape[-2]//2-M//2:mask2.shape[-2]//2-M//2+M,mask2.shape[-1]//2-M//2:mask2.shape[-1]//2-M//2+M]
		self.TransferFunctionIncoherent = nn.Parameter(mask.unsqueeze(4).repeat((1,1,1,1,2)).float(), requires_grad=True)
		# if object and img distance are different than the focal_lenght 
        # the wave-front can be propagated until the focal-plane, then 
        # propagate with fourier optics until the back-focalplane, then 
        # propagate again until img plane
		self.wave_prop_obj = None
		self.wave_prop_img = None

		# if self.obj_distance != self.focal_length:
		self.wave_prop_obj = ob.WavePropagation(self.optic_config, [], self.sampling_rate, self.obj_distance-self.focal_length, self.field_size)
		# if self.img_distance != self.focal_length:
		self.wave_prop_img = ob.WavePropagation(self.optic_config, [], self.sampling_rate, self.img_distance-self.focal_length, self.field_size)
		
	    
	def propagate_focal_to_back(self, u1):
        # Based on the function propFF out of the book "Computational Fourier
		# Optics. A MATLAB Tutorial". There you can find more information.

		wavelenght = self.optic_config.PSF_config.wvl
		[M,N] = u1.shape[-3:-1]

		#source sample interval
		dx1 = self.sampling_rate

		# obs sidelength
		L2 = wavelenght*self.focal_length/dx1

		#obs sample interval
		#dx2 = wavelenght*self.focal_length/L1

		# filter input with apperture mask
		# mask = self.mask.unsqueeze(0).unsqueeze(0).repeat((u1.shape[0],u1.shape[1],1,1,1))
		# u1 = torch.mul(u1,self.TransferFunctionIncoherent)

		#output field
		if M % 2==1:
			u2 = torch.mul(self.TransferFunctionIncoherent,ob.batch_fftshift2d(torch.fft(ob.batch_ifftshift2d(u1),2))) * dx1*dx1
		else:
			u2 = torch.mul(self.TransferFunctionIncoherent,ob.batch_ifftshift2d(torch.fft(ob.batch_fftshift2d(u1),2))) * dx1*dx1

		# multiply by precomputed coeff
		u2 = ob.mulComplex(u2, self.coefU1minus)
		return u2, L2

	def forward(self, field):
		# Propagate from obj_distance until focal plane
		if self.wave_prop_obj != None:
			field = self.wave_prop_obj(field)
        # Propagate from focal to back focal plane
		field,_ = self.propagate_focal_to_back(field)
        # Propagate from back focal to img plane
		if self.wave_prop_img != None:
			field = self.wave_prop_img(field)
		return field
            
	def __str__(self):
 		return "sampling_rate: " + str(self.sampling_rate) + " , " + "propagation_distance: " + str(self.propagation_distance) + " , " + "field_length: " + str(self.field_length) + " , " + "method: " + self.method + " , " + "propagation_function: " + self.propagation_function

        
