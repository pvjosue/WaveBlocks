# Third party library imports
import torch

# Waveblocks imports
from waveblocks.microscopes import BaseMicroscope
from waveblocks.blocks import Noise
from waveblocks.blocks import Camera
from waveblocks.blocks import WavePropagation
from waveblocks.blocks.microlens_arrays import PeriodicMLA


class Microscope(BaseMicroscope):
    def __init__(self, optic_config, members_to_learn, psf_in, precompute=True):
        super(Microscope, self).__init__(
            optic_config=optic_config,
            members_to_learn=members_to_learn,
            psf_in=psf_in,
            space_variant_psf=True,
        )
        self.space_variant_psf = self.use_mla
        if self.use_mla:
            # MLA
            self.mla = PeriodicMLA(
                optic_config=self.optic_config,
                members_to_learn=[],
                focal_length=optic_config.fm,
                pixel_size=self.sampling_rate,
                image_shape=self.psf_in.shape[2:4],
                block_shape=optic_config.Nnum,
                space_variant_psf=self.space_variant_psf,
                block_separation=optic_config.Nnum,
                block_offset=0,
            )

            # Propagation
            self.mla2sensor = WavePropagation(
                optic_config=self.optic_config,
                members_to_learn=[],
                sampling_rate=self.sampling_rate,
                shortest_propagation_distance=optic_config.mla2sensor,
                field_length=self.field_length,
            )

            if precompute:
                if self.use_mla:
                    # MLA
                    with torch.no_grad():
                        psf_5d = self.mla(self.psf_in, self.sampling_rate)
                        # Propagate from MLA to sensor
                        self.psf_at_sensor = torch.nn.Parameter(self.mla2sensor(psf_5d))

            else:
                self.psf_at_sensor = self.psf_in

        # Camera
        self.camera = Camera(
            optic_config=self.optic_config,
            members_to_learn=[],
            pixel_size=self.sampling_rate,
            space_variant_psf=self.space_variant_psf,
        )

            # System Noise
            # self.noise = Noise(optic_config=self.optic_config, members_to_learn=[])

    def forward(self, real_object):
        if self.use_mla:
            # compute final PSF and convolve with object
            convolved_obj, psf, _ = self.camera(
                real_object, self.psf_at_sensor, self.mla, full_psf_graph=True
            )
        else:
            convolved_obj, psf, _ = self.camera(real_object, self.psf_at_sensor, full_psf_graph=True)
        # convolved_obj = self.noise(convolved_obj)
        return convolved_obj


class MicroscopeTrainable(BaseMicroscope):
    def __init__(self, optic_config, members_to_learn, psf_in):
        super(MicroscopeTrainable, self).__init__(
            optic_config=optic_config,
            members_to_learn=members_to_learn,
            psf_in=psf_in,
            space_variant_psf=self.space_variant_psf,
        )

        if self.use_mla:
            # MLA
            self.mla = PeriodicMLA(
                optic_config=self.optic_config,
                members_to_learn=[],
                focal_length=optic_config.fm,
                pixel_size=self.sampling_rate,
                image_shape=self.psf_in.shape[2:4],
                block_shape=[optic_config.Nnum, optic_config.Nnum],
                space_variant_psf=self.space_variant_psf,
                block_separation=optic_config.block_separation,
                block_offset=0,
            )

            # Propagation
            self.mla2sensor = WavePropagation(
                optic_config=self.optic_config,
                members_to_learn=[],
                sampling_rate=self.sampling_rate,
                shortest_propagation_distance=optic_config.mla2sensor,
                field_length=self.field_length,
            )

        # Camera
        self.camera = Camera(
            optic_config=self.optic_config,
            members_to_learn=[],
            pixel_size=self.sampling_rate,
            space_variant_psf=self.space_variant_psf,
        )

        # System Noise
        self.noise = Noise(optic_config=self.optic_config, members_to_learn=[])

    def forward(self, real_object):

        psf_5d = self.mla(self.psf_in, self.sampling_rate)
        # propagate from MLA to sensor
        psf_at_sensor = self.mla2sensor(psf_5d)

        if self.useMLA:
            # compute final PSF and convolve with object
            convolved_obj, psf, _ = self.camera(real_object, psf_at_sensor, self.MLA)
        else:
            convolved_obj, psf, _ = self.camera(real_object, psf_at_sensor)

        # convolved_obj = self.noise(convolved_obj)
        return convolved_obj
