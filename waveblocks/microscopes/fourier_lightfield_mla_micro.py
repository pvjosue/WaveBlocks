# Waveblocks imports
from waveblocks.microscopes import BaseMicroscope
from waveblocks.blocks import Lens
from waveblocks.blocks import Camera
from waveblocks.blocks import WavePropagation
from waveblocks.blocks.microlens_arrays import MLAType
from waveblocks.blocks.microlens_arrays import CoordinateMLA
from waveblocks.blocks.microlens_arrays import PeriodicMLA
from waveblocks.blocks.phase_mask import PhaseMask
from waveblocks.blocks.optic_config import OpticConfig


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
        mla_shape=None,
        pm_image=None,
        pm_correction_image=None,
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
        if self.use_relay:
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
        else:
            raise Exception("Microscope is missing relay information")

        # Setup PhaseMask
        if self.use_pm:
            self.phase_mask = PhaseMask(
                optic_config=self.optic_config,
                members_to_learn=members_to_learn,
                sampling_rate=self.optic_config.pm_sampling,
                apperture_size=pm_image.shape[0],
                function_img=pm_image,
                max_phase_shift=self.optic_config.pm_max_phase_shift,
                element_y_angle=45,
                center_offset=[0, 0],
                correction_img=pm_correction_image,
            )

        # Setup MLA
        if self.use_mla:
            # MLA
            if self.mla_type == MLAType.coordinate:

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

            elif self.mla_type == MLAType.periodic:
                self.mla = PeriodicMLA(
                    optic_config=self.optic_config,
                    members_to_learn=members_to_learn,
                    focal_length=optic_config.fm,
                    pixel_size=self.sampling_rate,
                    image_shape=self.psf_in.shape[2:4],
                    block_shape=optic_config.Nnum,
                    space_variant_psf=self.space_variant_psf,
                    block_separation=optic_config.Nnum,
                    block_offset=0,
                )

            else:
                raise Exception("Microscope is missing mla type information")

            # Propagation
            self.mla2sensor = WavePropagation(
                optic_config=self.optic_config,
                members_to_learn=members_to_learn,
                sampling_rate=self.sampling_rate,
                shortest_propagation_distance=optic_config.mla2sensor,
                field_length=self.field_length,
            )
        else:
            raise Exception("Microscope is missing mla information")

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

            # Check whether relays are used for this microscope
            if self.use_relay:
                # First lens of relay
                psf = self.lens1(psf)
            # else:
            #     raise Exception("Microscope is missing relay information")

            # Check whether pm are used for this microscope
            if self.use_pm:
                # plt.title("PSF before PM")
                # plt.imshow(torch.imag(psf).cpu().detach().numpy()[0, 1, :, :])
                # plt.show()

                psf = self.phase_mask(psf, self.optic_config.pm_sampling)

            # Check whether a mla is used for this microscope
            if self.use_mla:
                # Create PSF of MLA
                if self.mla_type == MLAType.coordinate:
                    psf_mla = self.mla(psf, self.sampling_rate)

                elif self.mla_type == MLAType.periodic:
                    psf_mla = self.mla(psf, self.sampling_rate)

                else:
                    raise Exception("Microscope is missing mla type information")

                # Propagate MLA to sensor
                psf = self.mla2sensor(psf_mla)

            # Compute final PSF and convolve with object

        # if self.use_mla:
        #     convolved_object, psf, _ = self.camera(real_object, psf, self.mla, full_psf_graph=full_psf_graph)
        # else:
        #     convolved_object, psf, _ = self.camera(real_object, psf, full_psf_graph=full_psf_graph)

        if is_backwards:
            real_object = real_object.repeat(1, psf.shape[1], 1, 1)

        # Compute final PSF and convolve with object
        convolved_object, psf, _ = self.camera(real_object, psf, self.mla, full_psf_graph=full_psf_graph, is_backwards=is_backwards)
        


        self.psf = psf

        return convolved_object

def preset1():
    """
    New preset fixing uncentered PSF 
    """
    # Create opticalConfig object with the information from the microscope
    optic_config = OpticConfig()


    # Microscope numerical aperture
    optic_config.PSF_config.NA = 0.8
    # Microscope magnification
    optic_config.PSF_config.M = 16
    # Microscope tube-lens focal length
    optic_config.PSF_config.Ftl = 180000
    # Objective focal length = Ftl/M
    optic_config.PSF_config.fobj = optic_config.PSF_config.Ftl / optic_config.PSF_config.M
    # Emission wavelength
    optic_config.PSF_config.wvl = 0.63
    # Immersion refractive index
    optic_config.PSF_config.ni = 1.344
    optic_config.PSF_config.ni0 = 1.344

    # Camera
    optic_config.sensor_pitch = 6.45
    optic_config.use_relay = False


    # MLA
    optic_config.use_mla = True
    optic_config.mla_type = MLAType.coordinate
    # Distance between micro lenses centers
    optic_config.MLAPitch = 1000
    optic_config.MLAPitch_pixels = optic_config.MLAPitch // optic_config.sensor_pitch
    # Number of pixels behind a single lens
    optic_config.Nnum = 2 * [optic_config.MLAPitch // optic_config.sensor_pitch]
    optic_config.Nnum = [int(n + (1 if (n % 2 == 0) else 0))
                        for n in optic_config.Nnum]
    # Distance between the mla and the sensor
    optic_config.mla2sensor = 36100
    # MLA focal length
    optic_config.fm = 36100

    # Define phase_mask to initialize
    optic_config.use_relay = True
    optic_config.relay_focal_length = 125000
    optic_config.relay_aperture_width = 38100

    return optic_config

def preset_old():
    """
    Old preset with uncentered PSF.
    """
    # Create opticalConfig object with the information from the microscope
    optic_config = OpticConfig()

    # Microscope numerical aperture
    optic_config.PSF_config.NA = 0.45
    # Microscope magnification
    optic_config.PSF_config.M = 6
    # Microscope tube-lens focal length
    optic_config.PSF_config.Ftl = 165000
    # Objective focal length = Ftl/M
    optic_config.PSF_config.fobj = optic_config.PSF_config.Ftl / optic_config.PSF_config.M
    # Emission wavelength
    optic_config.PSF_config.wvl = 0.63
    # Immersion refractive index
    optic_config.PSF_config.ni = 1

    # Camera
    optic_config.sensor_pitch = 3.9
    optic_config.use_relay = False

    # MLA
    optic_config.use_mla = True
    optic_config.mla_type = MLAType.coordinate
    # Distance between micro lenses centers
    optic_config.MLAPitch = 250

    # Number of pixels behind a single lens
    optic_config.Nnum = 2 * [optic_config.MLAPitch // optic_config.sensor_pitch]
    optic_config.Nnum = [int(n + (1 if (n % 2 == 0) else 0)) for n in optic_config.Nnum]

    # Distance between the mla and the sensor
    optic_config.mla2sensor = 2500

    # MLA focal length
    optic_config.fm = 2500

    # Define phase_mask to initialize
    optic_config.use_relay = True
    optic_config.relay_focal_length = 150000
    optic_config.relay_separation = optic_config.relay_focal_length * 2
    optic_config.relay_aperture_width = 50800

