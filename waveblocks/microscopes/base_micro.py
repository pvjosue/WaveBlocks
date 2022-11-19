from abc import ABC, abstractmethod

# Third party libraries
import torch.nn as nn
import torch

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock


class BaseMicroscope(OpticBlock, ABC):
    """Abstract base class for a microscope.
    Contains all necessary methods and variables of a microscope"""

    def __init__(
        self, optic_config, members_to_learn,  psf_in, space_variant_psf
    ):
        # Initialize OpticBlock
        super(BaseMicroscope, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )
        # Initializes microscope specific values
        self.psf_in = nn.Parameter(psf_in, requires_grad=True)
        self.space_variant_psf = space_variant_psf
        self.sampling_rate = optic_config.sensor_pitch
        self.field_length = psf_in.shape[2]
        self.use_relay = (
            optic_config.use_relay if hasattr(optic_config, "use_relay") else False
        )
        self.use_mla = (
            optic_config.use_mla if hasattr(optic_config, "use_mla") else False
        )
        self.mla_type = optic_config.mla_type
        self.use_pm = optic_config.use_pm if hasattr(optic_config, "use_pm") else False

        self.dummy_device_param = nn.Parameter(torch.empty(0))

    @abstractmethod
    def forward(self, real_image):
        pass
    
    def get_device(self):
        return self.dummy_device_param.device

