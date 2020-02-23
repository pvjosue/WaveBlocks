import WaveBlocksPytorch as ob
import torch
import torch.nn as nn

# Microscope Class derived from Optical block, simulates the behavior of a microscope
class Microscope(ob.OpticBlock):
    def __init__(self, psfIn, vars_to_learn, optical_config):
        super(Microscope, self).__init__(optical_config, vars_to_learn)

        # Normalize PSF
        self.psf = nn.Parameter(psfIn/psfIn.sum(), requires_grad=True)
        
        # Fetch pixel size and sampling rate in general
        self.sampling_rate = optical_config.sensor_pitch
        # Lateral size of PSF
        field_length = psfIn.shape[2]

        # Create Wave-Propagation block, to defocus the PSF to a given depth
        self.wave_prop = ob.WavePropagation(self.optic_config, [], self.sampling_rate, optical_config.minDefocus, field_length)
        
        # Create Camera Block
        self.camera = ob.Camera(self.optic_config, [], self.sampling_rate)

    def forward(self, realObject):
        # Fetch PSF
        psf = self.psf

        # Defocus PSF given wave_prop.propagation_distance
        psf = self.wave_prop(psf)
        
        # Compute PSF irradiance and convolve with object
        finalImg,psf = self.camera(realObject, psf)
        return finalImg
