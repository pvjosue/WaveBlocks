import WaveBlocksPytorch as ob
import torch
import torch.nn as nn

class Noise(ob.OpticBlock):
	"""Class that adds Noise to a signal, either photon noise or read noise"""
	def __init__(self, optic_config, members_to_learn):
		super(Noise, self).__init__(optic_config, members_to_learn)

		# Random generator
		self.generator = torch.Generator()

		if hasattr(optic_config, 'camera_params'):
			# Sensor bit depth
			self.bit_depth = optic_config.camera_params.bit_depth
			# Amounts of posible electrons per pixel
			self.full_well = optic_config.camera_params.full_well
			# Baseline in ADU (analog-to-digital units) (Andor Zyla = 100) added to signal
			self.baseline = 100
			# Ratio of photons converted to electrons or quantum efficiency 
			self.quantum_eff = nn.Parameter(torch.tensor([optic_config.camera_params.quantum_eff], dtype=torch.float32))
			# 𝐾[𝐴𝐷𝑈/𝑒−]  represents the amplification of the voltage in the pixel from the photoelectrons
			self.sensitivity = nn.Parameter(torch.tensor([optic_config.camera_params.sensitivity], dtype=torch.float32))
			# Dark noise is the noise in the pixels when there is no light on the camera.
			# It is comprised of two separate noise sources, i.e. the read noise and the dark current
			self.dark_noise = nn.Parameter(torch.tensor([optic_config.camera_params.dark_noise], dtype=torch.float32))
		else: # Set default values for Hamamatzu orka 2.0 camera
			self.bit_depth = 16
			self.full_well = 30000
			self.baseline = 100
			self.quantum_eff = nn.Parameter(torch.tensor([0.82], dtype=torch.float32))
			self.sensitivity = nn.Parameter(torch.tensor([5.88], dtype=torch.float32))
			self.dark_noise = nn.Parameter(torch.tensor([2.29], dtype=torch.float32))
			
	def forward(self, input_irrad_gray):
		# This function converts the gray values to electrons, applies noise and converts back to gray values
		if self.generator.device != input_irrad_gray.device:
			self.generator = torch.Generator(device=input_irrad_gray.device)
		# todo: store normalization value
		input_irrad_gray *= self.full_well

		# Convert gray to photons
		input_irrad_electrons = input_irrad_gray-self.baseline
		input_irrad_electrons = input_irrad_electrons / self.sensitivity #* self.full_well / (2**self.bit_depth-1)
		input_irrad_electrons[input_irrad_electrons<0] = 0.0

		input_irrad_photons = input_irrad_electrons / self.quantum_eff

		# Add shot noise
		noisy_image =  input_irrad_photons.detach() - torch.poisson(input_irrad_photons.detach(), generator=self.generator)
		photons = input_irrad_photons + noisy_image
	
		# Convert to electrons
		electrons = self.quantum_eff * photons
	
		# Add dark noise
		electrons_out = torch.normal(mean=0.0, std=self.dark_noise, generator=self.generator) + electrons
	
		# Convert to ADU and add baseline
		max_adu	 = torch.tensor([2**self.bit_depth - 1], dtype=torch.float32, device=input_irrad_gray.device)
		adu		 = (electrons_out * self.sensitivity).float() # Convert to discrete numbers
		adu		 += self.baseline
		adu[adu > max_adu] = max_adu # models pixel saturation
	
		return adu / self.full_well
