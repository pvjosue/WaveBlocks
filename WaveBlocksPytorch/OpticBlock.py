import math
import torch.nn as nn
import abc

# Base class of WaveBlocks library containing functionality to fetch the parameters to learn
# in a WaveBlocks microscope

class OpticBlock(nn.Module): # pure virtual class
	"""Base class containing all the basic functionality of an optic block"""
	def __init__(self, opticConfig=None, members_to_learn=[]):
		super(OpticBlock, self).__init__()
		self.optic_config = opticConfig
		self.members_to_learn = members_to_learn# list of strings containing which members should be optimized
        # if None is provided then all members are created without gradients
	
	def get_trainable_variables(self):
		trainable_vars = []
		for name,param in self.named_parameters():
			if name in self.get_members_to_learn():
				trainable_vars.append(param)
		return list(trainable_vars)

	def get_optic_config(self):
		return self.optic_config

	def get_members_to_learn(self):
		return self.members_to_learn