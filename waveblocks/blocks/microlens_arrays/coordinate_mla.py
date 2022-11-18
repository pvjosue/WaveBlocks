# Third party libraries imports
import numpy as np
import torch
import torch.nn as nn
import logging

# Waveblocks imports
from waveblocks.blocks.microlens_arrays import BaseMLA
from waveblocks.utils.debug import debug_mla

logger = logging.getLogger("Waveblocks")


class CoordinateMLA(BaseMLA):
    """
    Implements and extends the abstract base class for a microlens array.
    Contains all necessary methods and variables of a coordinate micro lens array
    
    Example:
    
    self.mla = CoordinateMLA(
    optic_config=self.optic_config, # Forwards the optic config
    members_to_learn=members_to_learn, # Forwards the members to learn during the optimization process
    focal_length=optic_config.fm, # Extracts the focal length from the optic config
    pixel_size=self.sampling_rate, # Specifies the sampling rate
    image_shape=self.psf_in.shape[2:4], # Defines the output image shape
    block_shape=optic_config.Nnum, # Defines the amount of lenselet blocks
    space_variant_psf=self.space_variant_psf, # Specifies if it is a space variant psf
    mla_coordinates=self.mla_coordinates, # Allows to manually specify the coordinates of the microlens array
    mla_shape=self.mla_shape, # Defines the shape of the microlense array
    )
    
    """

    def __init__(
        self,
        optic_config,
        members_to_learn,
        focal_length,
        pixel_size,
        image_shape,
        block_shape,
        space_variant_psf,
        mla_coordinates,
        mla_shape,
        block_image=None,
        mla_image=None,
    ):
        # Initializes the BaseMLA
        super(CoordinateMLA, self).__init__(
            optic_config=optic_config,
            members_to_learn=members_to_learn,
            focal_length=focal_length,
            pixel_size=pixel_size,
            image_shape=image_shape,
            block_shape=block_shape,
            space_variant_psf=space_variant_psf,
            block_image=block_image,
            mla_image=mla_image,
        )
        # Initializes microscope specific values
        self.mla_coordinates = mla_coordinates
        self.mla_shape = mla_shape

        # Check if predefined mla exists otherwise compute it
        if self.mla_image is None:
            # Check if predefined block image exists otherwise compute it
            if self.block_image is None:
                self.block_image = nn.Parameter(
                    self.compute_block_image(), requires_grad=True
                )

            if logger.level == logging.DEBUG or logger.debug_mla:
                debug_mla(self.block_image)

            self.mla_image = nn.Parameter(self.compute_full_image(), requires_grad=True)

        logger.info("Initialized CoordinateMLA")

        d = {
            "MLA Coordinates": self.mla_coordinates,
            "MLA Shape": self.mla_shape,
        }
        
        if logger.level == logging.DEBUG or logger.debug_mla:
            for key in d:
                logger.debug_override("Output: {0}: {1}".format(key, d[key]))

        if logger.level == logging.DEBUG or logger.debug_mla:
            debug_mla(self.mla_image)

    def compute_full_image(self):

        logger.info("Computing Full Image ...")

        # Construct a empty MLA using the mla_shape to which all the lenses get added to
        mla = torch.zeros(
            [self.mla_shape[0], self.mla_shape[1]],
            dtype=torch.complex64,
        )

        # Cycle through coordinate list and place lenses
        for c in self.mla_coordinates:
            # Calculate left corner x,y = (self.block_image.x, self.block_image.y)
            block_shape = self.block_image.shape

            # Extract the first coordinate of the element in the coordinate list
            x_tuple = c[0]
            x_block_coordinates = np.ceil(block_shape[0] / 2)
            x = int(x_tuple - x_block_coordinates)
            # Check whether the lens is placed correctly
            if x < 0:
                logger.error(
                    "Lens cannot be placed outside of MLA! X-Coordinate is: " + str(x)
                )
                raise Exception(
                    "Lens cannot be placed outside of MLA! X-Coordinate is: " + str(x)
                )

            # Extract the second coordinate of the element in the coordinate list
            y_tuple = c[1]
            y_block_coordinates = np.ceil(block_shape[1] / 2)
            y = int(y_tuple - y_block_coordinates)
            # Check whether the lens is placed correctly
            if y < 0:
                logger.error(
                    "Lens cannot be placed outside of MLA! Y-Coordinate is: " + str(y)
                )
                raise Exception(
                    "Lens cannot be placed outside of MLA! Y-Coordinate is: " + str(y)
                )

            x_end = int(x + block_shape[0])
            # Check whether the lens is placed correctly
            if x_end > self.mla_shape[0]:
                logger.error(
                    "Lens cannot be placed outside of MLA! X-Coordinate is: "
                    + str(x_end)
                )
                raise Exception(
                    "Lens cannot be placed outside of MLA! X-Coordinate is: "
                    + str(x_end)
                )

            y_end = int(y + block_shape[1])
            # Check whether the lens is placed correctly
            if y_end > self.mla_shape[1]:
                logger.error(
                    "Lens cannot be placed outside of MLA! Y-Coordinate is: "
                    + str(y_end)
                )
                raise Exception(
                    "Lens cannot be placed outside of MLA! Y-Coordinate is: "
                    + str(y_end)
                )

            mla[x:x_end, y:y_end] = self.block_image

        logger.info("Successfully Computed Full Image!")

        return mla

    def forward(self, psf, input_sampling):
        if self.space_variant_psf:
            # resample might be needed if input_sampling rate is different than sampling_rate (aka phase-mask pixel size)
            # TODO: addapt to Pytorch 1.8 FFT standart, where complex64 is used instead of additional dimmension
            assert (
                input_sampling == self.pixel_size
            ), "MLA forward: PSF sampling rate and MLA pixel size should be the same"
            psf_shape = torch.tensor(psf.shape[0:4]).int()

            # half psf shape
            half_psf_shape = torch.ceil(psf_shape[2:4].float() / 2.0).int()
            # half size of the ML
            ml_half_shape = torch.ceil(self.block_shape.float() / 2.0).int()
            # half size of mla image
            mla_half_shape = torch.ceil(
                torch.tensor(self.mla_image.shape[0:2]).float() / 2.0
            ).int()

            # define output PSF
            psf_out = torch.zeros(
                (
                    psf_shape[0],
                    psf_shape[1],
                    ml_half_shape[0],
                    ml_half_shape[1],
                    psf_shape[2],
                    psf_shape[3]
                ),
                dtype=torch.float32,
            ).to(psf.device)

            # iterate positions inside the central ML,
            # as the psf diffracts different when placed at different spots of the mla
            for x1 in range(ml_half_shape[0]):
                x1_shift = ml_half_shape[0] - x1
                for x2 in range(ml_half_shape[1]):
                    # crop translated mla image, to have the element x,y at the center
                    x2_shift = ml_half_shape[1] - x2

                    transmittance_current_xy = self.mla_image[
                        (mla_half_shape[0] - half_psf_shape[0] - x1_shift + 1) : (
                            mla_half_shape[0] + half_psf_shape[0] - x1_shift
                        ),
                        (mla_half_shape[1] - half_psf_shape[1] - x2_shift + 1) : (
                            mla_half_shape[1] + half_psf_shape[1] - x2_shift
                        )
                    ]

                    # multiply by all depths
                    # replicate transmittance by nDepths
                    transmittance_current_xyz = (
                        transmittance_current_xy.unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(psf_shape[0], psf_shape[1], 1, 1)
                    )
                    psf_out[:, :, x1, x2, ...] = transmittance_current_xyz * psf

            # output is ordered as [depths, x, y, Nnum[0], Nnum[1], complex]
            return psf_out

        else:
            # Adjust shape of psf and add padding if necessary
            psf, mla = self.adjust_shape_psf_and_mla(psf, self.mla_image)

            # Cycle through depth of psf
            for z in range(psf.shape[1]):
                # Multiply PSF with MLA
                # TODO: check the .clone() maybe this breaks the graph
                psf[0, z, :, :] = psf[0, z, :, :].clone() * mla

            return psf
