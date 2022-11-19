# Waveblocks imports
from waveblocks.microscopes import BaseMicroscope
from waveblocks.blocks import Camera
from waveblocks.blocks import WavePropagation


# Microscope Class derived from Optical block, simulates the behavior of a microscope
class Microscope(BaseMicroscope):
    def __init__(self, optic_config, members_to_learn, psf_in):
        super(Microscope, self).__init__(
            optic_config=optic_config,
            members_to_learn=members_to_learn,
            psf_in=psf_in,
            space_variant_psf=False,
        )

        # Create Wave-Propagation block, to defocus the PSF to a given depth
        self.wave_prop = WavePropagation(
            optic_config=self.optic_config,
            members_to_learn=[],
            sampling_rate=self.sampling_rate,
            shortest_propagation_distance=optic_config.minDefocus,
            field_length=self.field_length,
        )

        # Create Camera Block
        self.camera = Camera(
            optic_config=self.optic_config,
            members_to_learn=[],
            pixel_size=self.sampling_rate,
            space_variant_psf=self.space_variant_psf,
        )

    def forward(self, real_object):
        # Fetch PSF
        psf = self.psf_in.clone()

        # Defocus PSF given wave_prop.propagation_distance
        psf = self.wave_prop(psf)

        # Compute PSF irradiance and convolve with object
        finalImg, psf, _ = self.camera(real_object, psf, full_psf_graph=True)
        return finalImg
