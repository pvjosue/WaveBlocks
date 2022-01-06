# Third party libraries imports
import numpy as np
import torch
import torch.nn as nn
import torch.fft

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock
from waveblocks.blocks.wave_propagation import WavePropagation
import waveblocks.utils.complex_operations as co


class Lens(OpticBlock):
    """Class that simulates a wave propagation through a lens, either from object plane to image_plane or from focal
    plane to back focal plane"""

    def __init__(
        self,
        optic_config,
        members_to_learn,
        focal_length,
        sampling_rate,
        aperture_width,
        field_length,
        object_distance,
        image_distance,
    ):

        super(Lens, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )
        self.focal_length = nn.Parameter(
            torch.tensor([focal_length], dtype=torch.float32, requires_grad=True)
        )
        self.sampling_rate = nn.Parameter(
            torch.tensor([sampling_rate], dtype=torch.float32, requires_grad=True)
        )
        self.aperture_width = nn.Parameter(
            torch.tensor([aperture_width], dtype=torch.float32, requires_grad=True)
        )
        self.field_size = nn.Parameter(
            torch.tensor([field_length], dtype=torch.float32, requires_grad=True)
        )
        self.obj_distance = nn.Parameter(
            torch.tensor([object_distance], dtype=torch.float32, requires_grad=True)
        )
        self.img_distance = nn.Parameter(
            torch.tensor([image_distance], dtype=torch.float32, requires_grad=True)
        )

        # compute input coefficient (scaling)
        U1C1 = (
            torch.exp(1j * optic_config.k * self.focal_length) 
            / optic_config.PSF_config.wvl 
            / self.focal_length
        )
        self.coefU1minus = nn.Parameter(U1C1 * (-1j))

        # compute cut off frequency and mask
        M = field_length

        # source sample interval
        dx1 = self.field_size / M

        # obs side length
        self.L = self.optic_config.PSF_config.wvl * self.focal_length / dx1

        # cutoff frequency
        f0 = self.aperture_width / (
            self.optic_config.PSF_config.wvl * self.focal_length
        )
        L = M * self.sampling_rate
        # source sample interval
        Fs = 1 / self.sampling_rate
        dFx = Fs / M
        u = np.sort(
            np.concatenate(
                (
                    np.arange(start=0, stop=-Fs / 2, step=-dFx, dtype="float"),
                    np.arange(start=dFx, stop=Fs / 2, step=dFx, dtype="float"),
                )
            )
        )

        X, Y = np.meshgrid(u, u)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        rho = torch.sqrt(torch.mul(X, X) + torch.mul(Y, Y))
        mask = ~(rho / f0).ge(torch.tensor([0.5]))
        maskBW = mask.unsqueeze(0).unsqueeze(0).float()

        mask = torch.mul((f0 - rho), maskBW)
        mask /= mask.max()

        # mask1 = F.pad(mask,2*[mask.shape[-1]//2]+2*[mask.shape[-2]//2])
        # mask2 = F.conv2d(mask1,mask)
        # mask2 = mask2[:,:,mask2.shape[-2]//2-M//2:mask2.shape[-2]//2-M//2+M,mask2.shape[-1]//2-M//2:mask2.shape[-1]//2-M//2+M]
        self.TransferFunctionIncoherent = nn.Parameter(
            mask.float(), requires_grad=True
        )
        # if object and img distance are different than the focal_lenght
        # the wave-front can be propagated until the focal-plane, then
        # propagate with fourier optics until the back-focalplane, then
        # propagate again until img plane
        self.wave_prop_obj = None
        self.wave_prop_img = None

        # if self.obj_distance != self.focal_length:
        self.wave_prop_obj = WavePropagation(
            optic_config=self.optic_config,
            members_to_learn=[],
            sampling_rate=self.sampling_rate,
            shortest_propagation_distance=self.obj_distance - self.focal_length,
            field_length=self.field_size,
        )
        # if self.img_distance != self.focal_length:
        self.wave_prop_img = WavePropagation(
            optic_config=self.optic_config,
            members_to_learn=[],
            sampling_rate=self.sampling_rate,
            shortest_propagation_distance=self.img_distance - self.focal_length,
            field_length=self.field_size,
        )

    def propagate_focal_to_back(self, u1):
        # Based on the function propFF out of the book "Computational Fourier
        # Optics. A MATLAB Tutorial". There you can find more information.

        wavelenght = self.optic_config.get_wavelenght()
        [M, N] = u1.shape[-3:-1]

        # source sample interval
        dx1 = self.sampling_rate

        # obs sidelength
        L2 = wavelenght * self.focal_length / dx1

        # obs sample interval
        # dx2 = wavelength*self.focal_length/L1

        # filter input with aperture mask
        # mask = self.mask.unsqueeze(0).unsqueeze(0).repeat((u1.shape[0],u1.shape[1],1,1,1))
        # u1 = torch.mul(u1,self.TransferFunctionIncoherent)

        # output field
        if M % 2 == 1:
            u2 = (
                torch.mul(
                    self.TransferFunctionIncoherent,
                    co.batch_fftshift2d(torch.fft.fft2(co.batch_ifftshift2d(u1))),
                )
                * dx1
                * dx1
            )
        else:
            u2 = (
                torch.mul(
                    self.TransferFunctionIncoherent,
                    co.batch_ifftshift2d(torch.fft.fft2(co.batch_fftshift2d(u1))),
                )
                * dx1
                * dx1
            )

        # multiply by precomputed coeff
        u2 = u2 * self.coefU1minus
        return u2, L2

    def forward(self, field):
        # Propagate from obj_distance until focal plane
        if self.wave_prop_obj is not None:
            field = self.wave_prop_obj(field)
        # Propagate from focal to back focal plane
        field, _ = self.propagate_focal_to_back(field)
        # Propagate from back focal to img plane
        if self.wave_prop_img is not None:
            field = self.wave_prop_img(field)
        return field

    def __str__(self):
        return (
            "sampling_rate: "
            + str(self.sampling_rate)
            + " , "
            + "propagation_distance: "
            + str(self.propagation_distance)
            + " , "
            + "field_length: "
            + str(self.field_length)
            + " , "
            + "method: "
            + self.method
            + " , "
            + "propagation_function: "
            + self.propagation_function
        )
