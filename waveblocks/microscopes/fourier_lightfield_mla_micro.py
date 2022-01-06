# Waveblocks imports
from waveblocks.microscopes import BaseMicroscope
from waveblocks.blocks import Lens
from waveblocks.blocks import Camera
from waveblocks.blocks import WavePropagation
from waveblocks.blocks.microlens_arrays import MLAType
from waveblocks.blocks.microlens_arrays import CoordinateMLA
from waveblocks.blocks.microlens_arrays import PeriodicMLA
from waveblocks.blocks.phase_mask import PhaseMask

from matplotlib import pyplot as plt
import torch


class Microscope(BaseMicroscope):
    """Implements and extends the abstract base class for a microscope.
    Contains all necessary methods and variables of a furier light field microscope"""

    def __init__(
        self,
        optic_config,
        members_to_learn,
        psf_in,
        mla_coordinates=None,
        mla_shape=None
    ):
        # Initializes the BaseMicroscope
        super(Microscope, self).__init__(
            optic_config=optic_config,
            members_to_learn=members_to_learn,
            psf_in=psf_in,
            space_variant_psf=False,
        )
        # Initializes microscope specific values
        self.mla_coordinates = mla_coordinates
        self.mla_shape = mla_shape

        # Setup relay
        self.lens1 = Lens(
            optic_config=self.optic_config,
            members_to_learn=members_to_learn,
            focal_length=optic_config.relay_focal_length,
            sampling_rate=self.sampling_rate,
            aperture_width=optic_config.relay_aperture_width,
            field_length=self.field_length,
            object_distance=optic_config.relay_focal_length,
            image_distance=optic_config.relay_focal_length,
        )
        


        # Setup MLA
        if self.mla_coordinates is None or self.mla_shape is None:
            raise Exception("MLA coordinates or mla shape missing!")

        self.mla = CoordinateMLA(
            optic_config=self.optic_config,
            members_to_learn=members_to_learn,
            focal_length=optic_config.fm,
            pixel_size=self.sampling_rate,
            image_shape=self.psf_in.shape[2:4],
            block_shape=optic_config.Nnum,
            space_variant_psf=self.space_variant_psf,
            mla_coordinates=self.mla_coordinates,
            mla_shape=self.mla_shape,
        )

        # Propagation
        self.mla2sensor = WavePropagation(
            optic_config=self.optic_config,
            members_to_learn=members_to_learn,
            sampling_rate=self.sampling_rate,
            shortest_propagation_distance=optic_config.mla2sensor,
            field_length=self.field_length,
        )
        
        # Setup camera
        self.camera = Camera(
            optic_config=self.optic_config,
            members_to_learn=members_to_learn,
            pixel_size=self.sampling_rate,
            space_variant_psf=self.space_variant_psf,
        )

    def forward(self, real_object, compute_psf=True, full_psf_graph=False, is_backwards=False):
        """
        (Calculates) and applies PSF to an object to simulate resulting image

        real_object: volume to simulate
        compute_psf: if the PSF needs to be recomputed (e.g. if PM has changed)
        full_psf_graph: if you wish to train a part of the PSF, you will need pytorch to compute the full PSF graph
        """

        if compute_psf:
            psf = self.psf_in
        else:
            psf = self.psf

        # If we need pytorch to have
        if full_psf_graph:
            self.camera.fft_paddings_ready = False

        if compute_psf:

            # Relay image from image plane to fourier space
            psf = self.lens1(psf)
    

            # Check whether a mla is used for this microscope
            if self.use_mla:
                # Create PSF of MLA
                psf_mla = self.mla(psf, self.sampling_rate)

                # Propagate MLA to sensor
                psf = self.mla2sensor(psf_mla)

        if is_backwards:
            real_object = real_object.repeat(1, psf.shape[1], 1, 1)
        # Compute final PSF and convolve with object
        convolved_object, psf, _ = self.camera(real_object, psf, self.mla, full_psf_graph=full_psf_graph, is_backwards=is_backwards)
        
        self.psf = psf

        return convolved_object
