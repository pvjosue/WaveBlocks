# Python imports
from abc import ABC, abstractmethod
from enum import Enum

# Third party libraries imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import logging

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock

logger = logging.getLogger("Waveblocks")


class MLAType(Enum):
    coordinate = 1
    periodic = 2


class BaseMLA(OpticBlock, ABC):
    """Abstract base class for a micro lens array.
    Contains all necessary methods and variables of a micro lens array"""

    def __init__(
            self,
            optic_config,
            members_to_learn,
            focal_length,
            pixel_size,
            image_shape,
            block_shape,
            space_variant_psf,
            block_image=None,
            mla_image=None,
    ):
        super(BaseMLA, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )
        self.focal_length = nn.Parameter(
            torch.tensor(focal_length, dtype=torch.float32, requires_grad=True)
        )
        self.pixel_size = nn.Parameter(
            torch.tensor(pixel_size, dtype=torch.float32, requires_grad=True)
        )
        self.image_shape = torch.tensor(image_shape, dtype=torch.int32)
        self.block_shape = torch.tensor(block_shape, dtype=torch.int32)
        self.space_variant_psf = space_variant_psf
        self.block_image = block_image
        self.mla_image = mla_image

        logger.info("Initialized BaseMLA")

        d = {
            "Focal Length": self.focal_length,
            "Pixel Size": self.pixel_size,
            "Image Shape": self.image_shape,
            "Block Shape": self.block_shape,
        }

        if logger.level == logging.DEBUG or logger.debug_mla:
            for key in d:
                logger.debug_override("Output: {0}: {1}".format(key, d[key]))

    # Methods which are used by all child classes

    def compute_block_image(self):

        logger.info("Computing Block Image ... ")

        half_size = torch.ceil(self.block_shape.float() / 2.0)
        # Compute image inside a block, give the transmittance function (trans_function)
        u = np.sort(
            np.concatenate(
                (
                    np.arange(
                        start=0,
                        stop=-self.pixel_size * half_size[0],
                        step=-self.pixel_size,
                        dtype="float",
                    ),
                    np.arange(
                        start=self.pixel_size.data,
                        stop=self.pixel_size * half_size[0],
                        step=self.pixel_size,
                        dtype="float",
                    ),
                )
            )
        )
        x, y = np.meshgrid(u, u)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        xl2norm = torch.mul(x, x) + torch.mul(y, y)

        # compute MLA transmittance exp(-1i*k/(2*FocalLength)*xL2norm)
        block_image = torch.exp(
            -1j * self.optic_config.k * xl2norm / (2 * self.focal_length)
        )

        logger.info("Successfully Computed Block Image!")

        return block_image

    @staticmethod
    def adjust_shape_psf_and_mla(psf, mla):

        #logger.info("Adjusting PSF and MLA Shape ... ")

        # Get shape of psf and mla
        psf_x, psf_y = psf.shape[2], psf.shape[3]
        mla_x, mla_y = mla.shape[0], mla.shape[1]

        # Check whether the psf or the mla has the bigger x dimension
        if psf_x != mla_x:
            if psf_x > mla_x:
                # Pad the mla_x to get the same dimension as the psf_x
                total_padding = psf_x - mla_x
                single_padding = int(np.floor(total_padding / 2))
                rest = total_padding % 2

                mla = f.pad(
                    input=mla,
                    pad=[0, 0, single_padding, int(single_padding + rest)],
                    mode="constant",
                    value=0,
                )
            else:
                # Pad the psf_x to get the same dimension as the mla_x
                total_padding = mla_x - psf_x
                single_padding = int(np.floor(total_padding / 2))
                rest = total_padding % 2

                psf = f.pad(
                    input=psf,
                    pad=[0, 0, single_padding, int(single_padding + rest), 0, 0, 0, 0],
                    mode="constant",
                    value=0,
                )

        # Check whether the psf or the mla has the bigger y dimension
        if psf_y != mla_y:
            if psf_y > mla_y:
                # Pad the mla_y to get the same dimension as the psf_y
                total_padding = psf_y - mla_y
                single_padding = int(np.floor(total_padding / 2))
                rest = total_padding % 2

                mla = f.pad(
                    input=mla,
                    pad=[single_padding, int(single_padding + rest), 0, 0],
                    mode="constant",
                    value=0,
                )
            else:
                # Pad the psf_y to get the same dimension as the mla_y
                total_padding = mla_y - psf_y
                single_padding = int(np.floor(total_padding / 2))
                rest = total_padding % 2

                psf = f.pad(
                    input=psf,
                    pad=[single_padding, int(single_padding + rest), 0, 0, 0, 0, 0, 0,
                         ],
                    mode="constant",
                    value=0,
                )

        #logger.info("Successfully Adjusted PSF and MLA Shape!")

        return psf, mla

    # Abstract methods which are implemented in all child classes which inherit this base class
    @abstractmethod
    def compute_full_image(self):
        pass

    @abstractmethod
    def forward(self, psf):
        pass
