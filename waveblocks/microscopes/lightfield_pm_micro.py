# Waveblocks imports
from waveblocks.microscopes import BaseMicroscope
from waveblocks.blocks import Lens
from waveblocks.blocks import Camera
from waveblocks.blocks import PhaseMask
from waveblocks.blocks import WavePropagation
from waveblocks.blocks.microlens_arrays import PeriodicMLA


class Microscope(BaseMicroscope):
    def __init__(self, optic_config, members_to_learn, psf_in):
        super(Microscope, self).__init__(
            optic_config=optic_config,
            members_to_learn=members_to_learn,
            psf_in=psf_in,
            space_variant_psf=True,
        )

        # Setup relay
        if self.use_relay:
            # Relays
            self.lens1 = Lens(
                optic_config=self.optic_config,
                members_to_learn=[],
                focal_length=optic_config.relay_focal_length,
                sampling_rate=self.sampling_rate,
                aperture_width=optic_config.relay_aperture_width,
                field_length=self.field_length,
                object_distance=optic_config.relay_focal_length,
                image_distance=optic_config.relay_focal_length,
            )

            self.lens2 = Lens(
                optic_config=self.optic_config,
                members_to_learn=[],
                focal_length=optic_config.relay_focal_length,
                sampling_rate=self.sampling_rate,
                aperture_width=optic_config.relay_aperture_width,
                field_length=self.field_length,
                object_distance=optic_config.relay_focal_length,
                image_distance=optic_config.relay_focal_length,
            )

            # Setup Spatial Light Modulator
            pm_sampling_rate = optic_config.pm_sampling
            pm_shape = optic_config.pm_shape
            pm_image = None

            if hasattr(optic_config, "pm_image"):
                pm_image = optic_config.pm_image
            if pm_sampling_rate is None:
                # Compute pixel size at fourier plane of first lens
                fs = 1.0 / self.sampling_rate
                cycles_perum = fs / self.field_length
                # Incoherent resolution limit
                resolution_limit = optic_config.PSF_config.wvl / (
                    optic_config.PSF_config.NA * optic_config.PSF_config.M
                )
                n_pix_in_fourier = resolution_limit / cycles_perum
                # Diameter of the objective back focal plane, which acts as the entrance pupil for our system
                d_obj = optic_config.PSF_config.fobj * optic_config.PSF_config.NA
                # Sampling size on the fourier domain
                self.fourierMetricSampling = d_obj / n_pix_in_fourier
                pm_sampling_rate = self.fourierMetricSampling
            else:
                self.fourierMetricSampling = pm_sampling_rate

            pm_correction_img = None
            if hasattr(optic_config, "pm_zero"):
                pm_correction_img = optic_config.pm_zero

            self.phaseMask = PhaseMask(
                optic_config=self.optic_config,
                members_to_learn=[],
                sampling_rate=pm_sampling_rate,
                apperture_size=pm_shape,
                function_img=pm_image,
                max_phase_shift=self.optic_config.max_phase_shift,
                element_y_angle=45,
                center_offset=[0, 0],
                correction_img=pm_correction_img,
            )

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

            # propagation
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

    def forward(self, real_object):

        # propagate a certain distance
        psf = self.psf_in.clone()

        if self.use_relay:
            # First lens of relay
            psf = self.lens1(psf)
            # interact with phase mask at fourier plane
            psf = self.phaseMask(psf, self.fourierMetricSampling)
            # Second lens of relay
            psf = self.lens2(psf)

        if self.use_mla:
            # MLA
            psf_5d = self.mla(psf, self.sampling_rate)
            # propagate from MLA to sensor
            psf_at_sensor = self.mla2sensor(psf_5d)
            # compute final PSF and convolve with object
            convolved_obj, psf, _ = self.camera(real_object, psf_at_sensor, self.mla)
        else:
            convolved_obj, psf, _ = self.camera(real_object, psf)
        return convolved_obj
