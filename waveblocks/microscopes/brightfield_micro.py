# Waveblocks imports
from waveblocks.microscopes import BaseMicroscope
from waveblocks.blocks import Camera
from waveblocks.blocks import WavePropagation


# Microscope Class derived from Optical block, simulates the behavior of a microscope
class BrightFieldMicroscope(BaseMicroscope):
    def __init__(self, optic_config, members_to_learn, psf_in):
        super(BrightFieldMicroscope, self).__init__(
            optic_config=optic_config,
            members_to_learn=members_to_learn,
            psf_in=psf_in,
            space_variant_psf=False,
        )

        # Create Camera Block
        self.camera = Camera(
            optic_config=self.optic_config,
            members_to_learn=[],
            pixel_size=self.sampling_rate,
            space_variant_psf=False,
        )

    def forward(self, real_object):
        # Fetch PSF
        psf = self.psf_in

        # Compute PSF irradiance and convolve with object
        finalImg, psf, _ = self.camera(real_object, psf)
        return finalImg
