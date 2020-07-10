import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class WavePropagation(ob.OpticBlock):
	"""Class that finds the correct diffraction approximation (Rayleight-sommerfield, Fresnel or Fraunhofer) for the given distance and field size(apperture), and populates propagation_function"""
	def __init__(self, optic_config, members_to_learn, sampling_rate, shortest_propagation_distance, field_length):
		
		super(WavePropagation, self).__init__(optic_config, members_to_learn)
		self.sampling_rate = nn.Parameter(torch.tensor([sampling_rate], dtype=torch.float32, requires_grad=True))
		self.propagation_distance = nn.Parameter(torch.tensor([shortest_propagation_distance], dtype=torch.float32, requires_grad=True))
		self.field_length = nn.Parameter(torch.tensor([field_length], dtype=torch.float32, requires_grad=True))
		
		self.impulse_response = nn.Parameter(None, requires_grad=True)
		self.h = nn.Parameter(None, requires_grad=True)

		# Field metric size
		L = torch.mul(self.sampling_rate, self.field_length)

		ideal_rate = torch.abs(self.optic_config.PSF_config.wvl  * self.propagation_distance / L)
		# Sampling defines if aliases apear or not, for shorter propagation distances higher sampling is needed
		# here we limit the ammount of sampling posible to 2500, which allow a propagation_distance as small as 400um
		ideal_samples_no = torch.min(torch.tensor([2500.0], dtype=torch.float32), torch.ceil(torch.div(L, ideal_rate)))
		self.ideal_samples_no = nn.Parameter(torch.add(ideal_samples_no, 1-torch.remainder(ideal_samples_no,2)), requires_grad=True)
		self.rate = nn.Parameter(torch.div(L, self.ideal_samples_no), requires_grad=True)

		u = np.sort(np.concatenate((np.arange(start = 0, stop = -L/2, step = -self.rate, dtype = 'float'),np.arange(start = self.rate, stop = L/2, step = self.rate, dtype = 'float'))))

		X, Y = np.meshgrid(u, u)
		X = torch.from_numpy(X).float()
		Y = torch.from_numpy(Y).float()
		self.XY = nn.Parameter(torch.mul(X,X) + torch.mul(Y,Y), requires_grad=True)
		# compute Rayleight-Sommerfield impulse response, which works for any propagation distance
		# self.compute_impulse_response()
			
	def forward(self, fieldIn):
		if self.propagation_distance==0.0:
			return fieldIn
			# self.propagation_distance.data = torch.tensor(0.001, dtype=torch.float32).to(self.propagation_distance.device)

		# check for space variant PSF
		if fieldIn.ndimension()==7:
			field = fieldIn.view(fieldIn.shape[0],-1,fieldIn.shape[4],fieldIn.shape[5],2)
		else:
			field = fieldIn
		fieldOut = field#.clone()
		nDepths = field.shape[1]


		# Pad before fft
		# padSize = 4*[self.ideal_samples_no.item()//2)]
	
		## Rayleight-Sommerfield
		Z = self.propagation_distance
		# compute coordinates
		rho = torch.sqrt(self.XY + torch.mul(Z,Z))
		# h = z./(1i*lambda.*rho.^2).* exp(1i*k.*rho);
		in1 =  ob.divComplex(torch.tensor([Z, 0.0], dtype=torch.float32).to(Z.device), torch.cat((torch.zeros(rho.shape).unsqueeze(2).to(Z.device),self.optic_config.PSF_config.wvl * torch.mul(rho,rho).unsqueeze(2)), 2))
		# exp(1i*k.*rho)
		in2 = ob.expComplex(torch.cat((torch.zeros(rho.shape,dtype=torch.float32).unsqueeze(2).to(Z.device),self.optic_config.k * torch.sign(self.propagation_distance) * rho.unsqueeze(2)), 2))
		h = ob.mulComplex(in2, in1)

		# Pad impulse response
		# h = torch.cat((F.pad(h.unsqueeze(0).unsqueeze(0)[:,:,:,:,0], padSize, 'reflect').unsqueeze(4),\
			# F.pad(h.unsqueeze(0).unsqueeze(0)[:,:,:,:,1], padSize, 'reflect').unsqueeze(4)),4)
		impulse_response_all_depths = self.rate * self.rate * torch.fft(h, 2)
		
		# # Pad impulse response
		# impulse_response_all_depths = torch.cat((F.pad(impulse_response_all_depths[:,:,:,:,0], padSize, 'reflect').unsqueeze(4),\
		# 	F.pad(impulse_response_all_depths[:,:,:,:,1], padSize, 'reflect').unsqueeze(4)),4)

		# iterate depths
		#outAllDepths = field.clone() # maybe consuming memory
		for k in range(nDepths):
			currSlice = field[:,k,:,:,:].unsqueeze(1)
			# Resample input idealy
			inUpReal = F.interpolate(currSlice[:,:,:,:,0],size=(int(self.ideal_samples_no.item()),int(self.ideal_samples_no.item())), mode='bilinear', align_corners=False)
			inUpImag = F.interpolate(currSlice[:,:,:,:,1],size=(int(self.ideal_samples_no.item()),int(self.ideal_samples_no.item())), mode='bilinear', align_corners=False)

			# pad input before fft
			# inUpReal = F.pad(inUpReal, padSize, 'reflect')
			# inUpImag = F.pad(inUpImag, padSize, 'reflect')

			inUp = torch.cat((inUpReal.unsqueeze(4), inUpImag.unsqueeze(4)), 4)

			fft1 = torch.fft(inUp,2)
			
			fft11 = ob.mulComplex(fft1, impulse_response_all_depths)

			out1 = ob.batch_ifftshift2d(torch.ifft(fft11,2))

			# crop padded after both ffts
			# out1 = out1[:,:,padSize[3]:-padSize[2],padSize[1]:-padSize[0],:]

			outSize = list(field.shape[-3:-1])
			out2R = F.interpolate(out1[:,:,:,:,0], outSize, mode='bilinear', align_corners=False)
			out2I = F.interpolate(out1[:,:,:,:,1], outSize, mode='bilinear', align_corners=False)

			outRI = torch.cat((out2R.unsqueeze(4), out2I.unsqueeze(4)), 4)

			fieldOut[:,k,:,:,:] = outRI

		if fieldIn.ndimension()==7:
			fieldOut = fieldOut.view(fieldIn.shape)
		return fieldOut
	
	def compute_impulse_response(self):
			
		# create complex tensors
		# todo eq 4.15 computational fourier optics a matlab tutorial
		## Fresnel
		# # exp(1i*k*z)/(1i*lambda*z)
		# in11 = ob.expComplex(torch.tensor([0.,k * self.propagation_distance]))
		# in12 = torch.tensor([0.,self.optic_config.PSF_config.wvl  * self.propagation_distance])
		# in1 = ob.divComplex(in11,in12)
		# # exp(1i * k/(2*z)*((x).^2+y.^2))
		# inR = 0 * X
		# inI = k / (2*self.propagation_distance) * torch.add(torch.mul(X,X),torch.mul(Y,Y))
		# inRI = torch.cat((inR.unsqueeze(2),inI.unsqueeze(2)),2)
		# in2 = ob.expComplex(inRI)
		

		## Rayleight-Sommerfield
		Z = self.propagation_distance# * torch.ones(X.shape)
		# coordinates
		rho = nn.Parameter(torch.sign(Z) * torch.sqrt(torch.mul(self.X,self.X) + torch.mul(self.Y,self.Y) + torch.mul(Z,Z)), requires_grad=True)
		# h = z./(1i*lambda.*rho.^2).* exp(1i*k.*rho);
		in1 =  ob.divComplex(torch.tensor([Z, 0.0]).to(Z.device), torch.cat((torch.zeros(rho.shape).unsqueeze(2).to(Z.device),self.optic_config.PSF_config.wvl  * torch.mul(rho,rho).unsqueeze(2)), 2))
		# exp(1i*k.*rho)
		in2 = ob.expComplex(torch.cat((torch.zeros(rho.shape).unsqueeze(2).to(Z.device),self.optic_config.k * rho.unsqueeze(2)), 2))
		self.h = nn.Parameter(ob.mulComplex(in2, in1), requires_grad=True)
		# self.rho = nn.Parameter(rho)
		self.impulse_response = nn.Parameter(self.rate * self.rate * torch.fft(self.h, 2), requires_grad=True)

	def __str__(self):
 		return "sampling_rate: " + str(self.sampling_rate) + " , " + "propagation_distance: " + str(self.propagation_distance) + " , " + "field_length: " + str(self.field_length) + " , " + "method: " + self.method + " , " + "propagation_function: " + self.propagation_function
