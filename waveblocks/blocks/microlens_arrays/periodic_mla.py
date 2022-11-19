# Python imports
import math

# Third party libraries imports
import torch
import torch.nn as nn
import logging

# Waveblocks imports
from waveblocks.blocks.microlens_arrays import BaseMLA
from waveblocks.utils.debug import debug_mla

logger = logging.getLogger("Waveblocks")


class PeriodicMLA(BaseMLA):
    """
    Implements and extends the abstract base class for a micro lens array.
    Contains all necessary methods and variables of a coordinate micro lens array
    
    Example:
    self.mla = PeriodicMLA(
    optic_config=self.optic_config, # Forwards the optic config
    members_to_learn=members_to_learn, # Forwards the members to learn during the optimization process
    focal_length=optic_config.fm, # Extracts the focal length from the optic config
    pixel_size=self.sampling_rate, # Specifies the sampling rate
    image_shape=self.psf_in.shape[2:4], # Defines the output image shape
    block_shape=optic_config.Nnum, # Defines the amount of lenselet blocks
    space_variant_psf=self.space_variant_psf, # Specifies if it is a space variant psf
    block_separation=optic_config.Nnum, # Defines the space separation between blocks
    block_offset=0,
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
        block_separation,
        block_offset,
        block_image=None,
        mla_image=None,
        debug=False,
    ):
        # Initializes the BaseMLA
        super(PeriodicMLA, self).__init__(
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

        # Initialize specific mla values
        self.block_separation = torch.tensor(block_separation, dtype=torch.int32)
        self.block_offset = torch.tensor(block_offset, dtype=torch.int32)

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

        logger.info("Initialized PeriodicMLA")

        d = {
            "Block Separatoion": self.block_separation,
            "Block Offset": self.block_offset,
        }

        if logger.level == logging.DEBUG or logger.debug_mla:
            for key in d:
                logger.debug_override("Output: {0}: {1}".format(key, d[key]))


        if logger.level == logging.DEBUG or logger.debug_mla:
            debug_mla(self.mla_image)

    def compute_full_image(self):

        logger.info("Computing Full Image ...")

        # replicate the block image such that its larger than the PSF size
        n_repetitions = (
            torch.ceil(torch.div(self.image_shape.float(), self.block_shape.float()))
            + 2
        )
        n_repetitions = torch.add(n_repetitions, 1 - torch.remainder(n_repetitions, 2))
        full_mla_image = self.block_image.repeat(
            int(n_repetitions[0]), int(n_repetitions[1])
        )

        # Calculate the center points of the
        center_x = math.floor(full_mla_image.shape[0] / 2)
        center_y = math.floor(full_mla_image.shape[1] / 2)

        a = center_x - self.image_shape[0] // 2
        b = center_x + self.image_shape[0] // 2 + 1
        c = center_y - self.image_shape[1] // 2
        d = center_y + self.image_shape[1] // 2 + 1

        cropped_mla_image = full_mla_image[a:b, c:d]

        assert (
            cropped_mla_image.shape[0] == self.image_shape[0]
            and cropped_mla_image.shape[1] == self.image_shape[1]
        ), "Cropped mla image has to have the same size as the image shape"

        logger.info("Successfully Computed Full Image")

        return cropped_mla_image

    def forward(self, psf, input_sampling):
        if self.space_variant_psf:
            # resample might be needed if input_sampling rate is different than sampling_rate (aka phase-mask pixel size)
            # TODO
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
                dtype=torch.complex64,
            ).to(psf.device)

            # iterate positions inside the central ML,
            # as the psf diffracts different when placed at different spots of the mla
            for x1 in range(ml_half_shape[0]):
                x1_shift = ml_half_shape[0] - x1
                bottom_corner_1 = max((mla_half_shape[0] - half_psf_shape[0] - x1_shift + 1).item(), 0)
                top_corner_1 = min((mla_half_shape[0] + half_psf_shape[0] - x1_shift).item(), self.mla_image.shape[0])
                for x2 in range(ml_half_shape[1]):
                    # crop translated mla image, to have the element x,y at the center
                    x2_shift = ml_half_shape[1] - x2

                    bottom_corner_2 = max((mla_half_shape[1] - half_psf_shape[1] - x2_shift + 1).item(), 0)
                    top_corner_2 = min((mla_half_shape[1] + half_psf_shape[1] - x2_shift).item(), self.mla_image.shape[1])

                    transmittance_current_xy = torch.zeros_like(self.mla_image)

                    transmittance_current_xy[bottom_corner_1 : top_corner_1,
                        bottom_corner_2 : top_corner_2] = self.mla_image[
                        bottom_corner_1 : top_corner_1,
                        bottom_corner_2 : top_corner_2,
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
                psf[0, z, ...] = psf[0, z, ...].clone() * mla

            return psf
