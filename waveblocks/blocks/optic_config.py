# Python imports
import math

# Third party libraries imports
import torch.nn as nn


class PSFConfig:
    pass

class scattering:
    pass

class OpticConfig(nn.Module):
    """Class containing the global parameters of an optical system:
    Keyword args:
    wavelenght: wavelenght of optical system (in um)
    samplingRate: sampling used when propagating wavefronts through space. Or when using a camera.
    k: wave number
    """

    @staticmethod
    def get_default_PSF_config():
        psf_config = PSFConfig()
        psf_config.M = 40  # Magnification
        psf_config.NA = 0.9  # Numerical Aperture
        psf_config.Ftl = 165000  # Tube lens focal length
        psf_config.ns = 1.33  # Specimen refractive index (RI)
        psf_config.ng0 = 1.515  # Coverslip RI design value
        psf_config.ng = 1.515  # Coverslip RI experimental value
        psf_config.ni0 = 1  # Immersion medium RI design value
        psf_config.ni = 1  # Immersion medium RI experimental value
        psf_config.ti0 = (
            150  # Microns, working distance (immersion medium thickness) design value
        )
        psf_config.tg0 = 170  # Microns, coverslip thickness design value
        psf_config.tg = 170  # Microns, coverslip thickness experimental value
        psf_config.zv = (
            0  # Offset of focal plane to coverslip, negative is closer to objective
        )
        psf_config.wvl = 0.63  # Wavelength of emission

        return psf_config

    def __init__(self, PSF_config=None):
        super(OpticConfig, self).__init__()
        if PSF_config is None:
            self.PSF_config = self.get_default_PSF_config()
        else:
            self.PSF_config = PSF_config
        self.set_k()

    def get_wavelenght(self):
        return self.PSF_config.wvl

    def get_medium_refractive_index(self):
        return self.PSF_config.ni

    def set_k(self):
        # Wave Number
        self.k = 2 * math.pi * self.PSF_config.ni / self.PSF_config.wvl  # wave number
        
    def get_k(self):
        return self.k
