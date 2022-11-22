# This file provides functions for reconstructing volumes from an image taken by a FLF microscope using Adam optimizer

# Erik Riedel & Josef Kamysek
# erik.riedel@tum.de & josef@kamysek.com
# 5/11/2020, Munich, Germany

# Third party library imports
from PIL.Image import init
from waveblocks import microscopes
import torch
import matplotlib.pyplot as plt
import math
import logging
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger("Waveblocks")


class Optimizer_Reconstruction:
    def __init__(self, microscope, GT_FLF_img, volume_shape, optimizer, lr, crit, init_volume=None):
        """
        Initializes all variables required for reconstructing a volume using torch optimizers.
        params:
            - microscope: Configured microscope object
            - GT_FLF_img: Image captured by microscope
            - volume_shape: Shape of volume to reconstruct
            - optimizer: torch optimizer to use e.g. torch.optim.adam
            - lr: learning rate
            - crit: loss function e.g.torch.nn.L1Loss()
        """

        # Setup volume, optimizer and loss function
        self.microscope = microscope
        if(init_volume == None):
            self.microscope.volume_pred = nn.Parameter(
                torch.ones(volume_shape, device=self.microscope.psf.device) * 0.5,
                requires_grad=True,
            )
        else:
            self.microscope.volume_pred = nn.Parameter(
                init_volume,
                requires_grad=True,
            )

        self.GT_FLF_img = GT_FLF_img

        if optimizer == torch.optim.SGD:
            self.optimizer = torch.optim.SGD([{"params": self.microscope.get_trainable_variables(), "lr": lr}], lr=1e-2, momentum=0.9)
        else:
            self.optimizer = optimizer([{"params": self.microscope.get_trainable_variables(), "lr": lr}], lr=lr)

        self.crit = crit

    def train(self, num_epochs):
        """
        Reconstruct volume using torch optimizer in given epochs
        """
        loss = 0
        loss_plot = []

        display_modulo = math.floor(num_epochs / 20)
        if display_modulo == 0:
            display_modulo = 1

        for epoch in tqdm(range(num_epochs), "Reconstructing volume"):

            self.optimizer.zero_grad()
            self.microscope.zero_grad()

            # If we are learning the phase Mask then we need to recompute psf and generate full pytorch graph
            compute_psf = False
            full_psf_graph = False
            if "phase_mask.function_img" in self.microscope.members_to_learn:
                compute_psf = True
                full_psf_graph = True

            img_pred = self.microscope(
                self.microscope.volume_pred,
                compute_psf=compute_psf,
                full_psf_graph=full_psf_graph,
            )

            loss = self.crit(img_pred, self.GT_FLF_img)

            loss.backward()    
            self.optimizer.step()

            if logger.level == logging.DEBUG or logger.debug_optimizer:
                if epoch % display_modulo == 0:
                    logger.debug_override(
                        "loss = {} - {:.2f} %% done".format(
                            str(loss.sum()),
                            (100 * epoch / num_epochs),
                        )
                    )

            loss_plot.append(loss.item())
            
        return self.microscope.volume_pred, loss_plot
