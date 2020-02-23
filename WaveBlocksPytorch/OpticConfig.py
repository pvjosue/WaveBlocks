import math
import torch.nn as nn

class OpticConfig(nn.Module):
	"""Class containing the global parameters of an optical system:
	Keyword args:
	wavelenght: wavelenght of optical system (in um)
	samplingRate: sampling used when propagating wavefronts through space. Or when using a camera.
	k: wave number
	"""
	def __init__(self, wavelenght, medium_refractive_index):
		super(OpticConfig, self).__init__()
		self.wavelenght = wavelenght # wavelenght of used light (um)
		self.medium_refractive_index = medium_refractive_index # sampling rate or sensor size(um)
		self.k = 2*math.pi*self.medium_refractive_index / self.wavelenght # wave number
		
	def get_wavelenght(self):
		return self.wavelenght

	def get_medium_refractive_index(self):
		return self.medium_refractive_index

	def get_k(self):
		return self.k

	def __str__():
 		return "wavelenght: " + self.wavelenght + " , " + "medium_refractive_index: " + self.medium_refractive_index + "k: " + self.k
