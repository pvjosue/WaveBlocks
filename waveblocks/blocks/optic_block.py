# Third party libraries imports
import torch.nn as nn


class OpticBlock(nn.Module):  # pure virtual class
    """Base class containing all the basic functionality of an optic block"""

    def __init__(
        self, optic_config=None, members_to_learn=None,
    ):  # Contains a list of members which should be optimized (In case none are provided members are created without gradients)
        super(OpticBlock, self).__init__()
        self.optic_config = optic_config
        self.members_to_learn = [] if members_to_learn is None else members_to_learn


    def get_trainable_variables(self):
        trainable_vars = []
        for name, param in self.named_parameters():
            if name in self.members_to_learn:
                trainable_vars.append(param)
        return list(trainable_vars)
