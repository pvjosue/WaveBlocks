# Python imports
import math

# Third party libraries imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from matplotlib import pyplot as plt

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock


class PhaseMask(OpticBlock):
    """Class that simulates a diffractive element such as an apperture, phase mask, coded apperture, etc."""

    def __init__(
        self,
        optic_config,
        members_to_learn,
        sampling_rate,
        apperture_size,
        function_img,
        max_phase_shift=5 * math.pi,
        element_y_angle=45.0,
        center_offset=[0.0, 0.0],
        correction_img=None,
    ):
        # aperture_shape is the 2D size of the element in pixels
        # sampling_rate is the pixel size of the diffraction element
        super(PhaseMask, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )
        self.sampling_rate = nn.Parameter(
            torch.tensor([sampling_rate], dtype=torch.float32, requires_grad=True)
        )
        self.apperture_size = nn.Parameter(
            torch.tensor(apperture_size, dtype=torch.float32, requires_grad=True)
        )
        self.metric_size = nn.Parameter(sampling_rate * self.apperture_size)
        self.element_y_angle = nn.Parameter(
            torch.tensor(element_y_angle, dtype=torch.float32, requires_grad=True)
        )
        self.center_offset = nn.Parameter(
            torch.tensor(center_offset, dtype=torch.float32, requires_grad=True)
        )
        self.max_phase_shift = nn.Parameter(
            torch.tensor(max_phase_shift, dtype=torch.float32, requires_grad=True)
        )
        self.constant_phase = nn.Parameter(
            torch.tensor(max_phase_shift / 2, dtype=torch.float32, requires_grad=True)
        )
        self.correction_img = nn.Parameter(correction_img)
        # phase function is a lambda function to create in the image based on its x,y coords
        if function_img is not None:
            self.function_img = nn.Parameter(function_img, requires_grad=True)
        else:
            self.function_img = nn.Parameter(
                self.max_phase_shift / 2
                - 0.5
                + torch.rand(
                    (1, 1, apperture_size[0], apperture_size[1]), requires_grad=True
                )
            )

    def forward(self, fieldIn, input_sampling):
        # self.function_img = self.function_img.clamp(0,self.max_phase_shift)
        inShape = torch.tensor(fieldIn.shape)

        if self.correction_img.ndim != 1:
            function_img = self.correction_img.detach() + self.function_img
        else:
            function_img = self.function_img  # - self.function_img.min()
            
        function_img = function_img * self.max_phase_shift
        function_img = function_img / function_img.max() * self.max_phase_shift

        # translate the image to the center of the SLM
        if self.center_offset.sum() != 0.0:
            padOffsetSize = 2 * [abs(int(self.center_offset[1]))] + 2 * [
                abs(int(self.center_offset[0]))
            ]
            function_img_size = function_img.shape
            function_img_padded = f.pad(function_img, padOffsetSize, "reflect")
            # crop ROI
            function_img = function_img_padded[
                :,
                :,
                (
                    function_img_padded.shape[-2] // 2
                    - int(self.center_offset[0])
                    - function_img_size[-2] // 2
                ) : (
                    function_img_padded.shape[-2] // 2
                    - int(self.center_offset[0])
                    - function_img_size[-2] // 2
                    + function_img_size[-2]
                ),
                (
                    function_img_padded.shape[-1] // 2
                    - int(self.center_offset[1])
                    - function_img_size[-1] // 2
                ) : (
                    function_img_padded.shape[-1] // 2
                    - int(self.center_offset[1])
                    - function_img_size[-1] // 2
                    + function_img_size[-1]
                ),
            ]

        self.function_img_with_offset = function_img.clone()

        # reshape and mask either input field or local image to match sampling_rate
        resample = (
            input_sampling != self.sampling_rate
            or inShape[2] != function_img.shape[2]
            or inShape[3] != function_img.shape[3]
        )
        field = fieldIn
        paddInput = False
        if resample:
            # compute ratio between input and phase image, to have a matching pixels to sampling
            ratio_self_input = self.sampling_rate / input_sampling

            # rotate phase mask in the specified angle
            ratio_self_input_y = ratio_self_input * torch.cos(
                torch.tensor(np.radians(self.element_y_angle.item()))
            )
            ks = [int(1 / ratio_self_input), int(1 / ratio_self_input_y)]
            function_img = f.avg_pool2d(
                self.function_img, kernel_size=ks, padding=[ks[0] // 2, ks[1] // 2]
            )
            # resample funciton image
            # function_img = F.interpolate(function_img, scale_factor=(ratio_self_input_x,ratio_self_input_y), mode='bilinear', align_corners=False)
            newSize = function_img.shape[2:4]
            # pad input to match size of phase mask, in case that the input is smaller
            if (
                field.shape[-3] < function_img.shape[-2]
                or field.shape[-2] < function_img.shape[-1]
            ):
                paddInput = True
                padSize = list(
                    [
                        (newSize[1] - inShape[3]) // 2,
                        (newSize[1] - inShape[3]) // 2,
                        (newSize[0] - inShape[2]) // 2,
                        (newSize[0] - inShape[2]) // 2,
                    ]
                )
                field = f.pad(fieldIn, tuple(padSize), "constant", 0)
            else:
                padSize = list(
                    [
                        (inShape[3] - newSize[1]) // 2,
                        (inShape[3] - newSize[1]) // 2,
                        (inShape[2] - newSize[0]) // 2,
                        (inShape[2] - newSize[0]) // 2,
                    ]
                )
                function_img = f.pad(function_img, tuple(padSize), "constant", 0)
            # check if newSize is even, as the padding would be different
            # if newSize[0]%2==0:
            # 	padSize[5] -= 1
            # if newSize[1]%2==0:
            # 	padSize[3] -= 1

            # todo: avoid next interpolation by computing the size correctly in the first place
            function_img = f.interpolate(
                function_img,
                size=field.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # element wise multiplication with phase element
        function_img = function_img.repeat(inShape[0], inShape[1], 1, 1)

        # clamp to 8 bits, posible in Phase-mask
        # min_pm_step = self.max_phase_shift/256.0
        # function_img = ((function_img/min_pm_step).int() * min_pm_step).float()

        # set real part to zero, as phase-mask only affects phase
        function_img = torch.exp(1j * function_img)

        # resample might be needed if input_sampling rate is different than sampling_rate (aka phase-mask pixel size)
        # print(field.shape)
        # print(function_img.shape)
        # plt.title("Input Field")
        # plt.imshow(torch.imag(field[0,5,:,:]).detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.title("PM image")
        # plt.imshow(torch.imag(function_img[0,5,:,:]).detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
        output = field * function_img
        # plt.title("Output Field")
        # plt.imshow(torch.imag(output[0,5,:,:]).detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()

        if resample and paddInput:
            output = output[
                :,
                :,
                padSize[2] : output.shape[2] - padSize[3],
                padSize[0] : output.shape[3] - padSize[1],
            ]

        return output.contiguous()
