# Third party libraries imports
import torch
import torch.nn as nn
import torch.nn.functional as f

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock
import waveblocks.utils.complex_operations as co


class Camera(OpticBlock):
    """Class that simulates a Camera, and computes the intensity in each of it pixels given an incoming wave-front"""

    def __init__(
        self, optic_config, members_to_learn, pixel_size, space_variant_psf
    ):
        # aperture_shape is the 2D size of the element in pixels
        # sampling_rate is the pixel size of the diffractive element
        super(Camera, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )
        self.space_variant_psf = space_variant_psf
        self.pixel_size = nn.Parameter(torch.tensor([pixel_size], dtype=torch.float32))
        self.fft_paddings_ready = False
        if hasattr(optic_config, "use_fft_conv"):
            self.use_fft_conv = optic_config.use_fft_conv
        else:
            self.use_fft_conv = True

    def forward(self, input, psf, mla=None, is_backwards=False, full_psf_graph=False):
        # This function either convolves the current PSF with a given object, or propagates the PSF through a
        # periodic_element like a MicroLensArray, returning a 5D PSF assert isinstance(psf, ob.PSF),
        # "ob.PSF object needed to compute an image at the camera sensor"
        
        # Check if PSF is in complex mode and convert to intensity
        if 'ComplexFloatTensor' in psf.type():
            psf_at_sensor = psf.abs()**2
        else:
            psf_at_sensor = psf

        psf_normalized = self.normalize_psf(psf_at_sensor)

        output = None
        if self.space_variant_psf:
            if is_backwards:
                raise NotImplementedError
            image = self.apply_space_variant_psf(input, psf_normalized, mla)
        else:
            if self.use_fft_conv:
                output = self.apply_space_invariant_psf_fourier(
                    input, psf_normalized, is_backwards=is_backwards, full_psf_graph=full_psf_graph
                )
            else:
                output = self.apply_space_invariant_psf(
                    input, psf_normalized, is_backwards=is_backwards,
                )

            # sum convolutions per depth
            if not is_backwards:
                output = output.sum(1).unsqueeze(1)

        return output, psf_normalized, output

    def apply_space_variant_psf(self, input, psf, mla):
        # this function computes the PSF when passed through a MLA or any space variant element
        input_shape = torch.tensor(input.shape, dtype=torch.int32)
        input_half_shape = input_shape[2:4] // 2 + 1
        ml_shape = mla.block_shape
        ml_half_shape = ml_shape // 2 + 1

        # update PSF in MLA
        # MLA.set_space_variant_psf(psfNormalized)

        n_depths = input_shape[1].item()

        # padd input until have a full MLA number inside image and a MLA in the senter
        # padAmount = (torch.mul(ml_shape, torch.ceil(torch.div(input_shape[2:4], ml_shape.float()))) - input_shape[2:4]).int()
        # pad_size = (padAmount[0].item()//2, padAmount[0].item()-padAmount[0].item()//2, padAmount[1].item()//2, padAmount[1].item()-padAmount[1].item()//2)

        residuals_lower_border = (
            torch.mul(
                1
                + torch.floor(
                    (input_half_shape.float() - ml_half_shape.float())
                    / ml_shape.float()
                ),
                ml_shape.float(),
            )
            - (input_half_shape - ml_half_shape).float()
        )

        pad_size = (
            int(residuals_lower_border[1].item()),
            int(residuals_lower_border[1].item()),
            int(residuals_lower_border[0].item()),
            int(residuals_lower_border[0].item()),
        )
        padded_input = torch.nn.functional.pad(input, pad_size, "constant", 0)

        padded_shape = torch.tensor(padded_input.shape, dtype=torch.int32)
        padded_center = padded_shape[2:4] // 2  # todo: check this
        # compute views to group together all elements with the same position relative to the MLAs centers
        out_channel = padded_shape[1] * (ml_shape[0] ** 2)
        out_h = padded_shape[2] // ml_shape[0]
        out_w = padded_shape[2] // ml_shape[1]
        fm_view = padded_input.contiguous().view(
            padded_shape[0], padded_shape[1], out_h, ml_shape[0], out_w, ml_shape[1]
        )

        # pad again to match output size of convolution with input size
        pad_size2 = (0, 0, 1, 1, 0, 0, 1, 1)
        fm_view = torch.nn.functional.pad(fm_view, pad_size2, "constant", 0)

        final_image = torch.zeros(input.shape, dtype=torch.float32).to(input.device)
        for x1 in range(ml_shape[0]):
            flip_x1 = True if x1 >= ml_half_shape[0].item() else False
            x1_quarter = (
                ml_shape[0].item() - x1 - 1 if x1 >= ml_half_shape[0].item() else x1
            )
            for x2 in range(ml_shape[1]):
                curr_input = fm_view[:, :, :, x1, :, x2]
                if curr_input.sum() == 0.0:
                    continue

                flip_x2 = True if x2 >= ml_half_shape[1].item() else False
                x2_quarter = (
                    ml_shape[1].item() - x2 - 1 if x2 >= ml_half_shape[1].item() else x2
                )
                # fetch correct PSF pattern
                curr_psf = psf[:, :, x1_quarter, x2_quarter, :, :]

                # flip if necesary
                curr_psf = curr_psf if flip_x1 == False else curr_psf.flip(2)
                curr_psf = curr_psf if flip_x2 == False else curr_psf.flip(3)

                # convolve view with corrsponding PSF
                curr_out = nn.functional.conv_transpose2d(
                    curr_input,
                    curr_psf.permute(1, 0, 2, 3),
                    stride=ml_shape[0].item(),
                    padding=curr_psf.shape[3] // 2 + ml_half_shape[0].item(),
                    groups=n_depths,
                )

                # crop current sensor response to be centered at the current x1,x2
                offset_low_boundery = (
                    padded_center
                    - input_half_shape
                    - (torch.tensor([x1, x2], dtype=torch.int32) - ml_half_shape)
                )
                offset_high_boundery = (
                    padded_center
                    + input_half_shape
                    - (torch.tensor([x1, x2], dtype=torch.int32) - ml_half_shape)
                    - 1
                )
                curr_out_cropped = curr_out[
                    :,
                    :,
                    offset_low_boundery[0] : offset_high_boundery[0],
                    offset_low_boundery[1] : offset_high_boundery[1],
                ]

                # accumulate output image
                final_image = torch.add(final_image, curr_out_cropped)

        # sum convolutions per depth
        final_image = final_image.sum(1).unsqueeze(1)
        return final_image

    def apply_space_invariant_psf(self, real_object, psf, is_backwards=False):
        # check for same number of depths
        assert (
            psf.shape[1] == real_object.shape[1]
        ), "Different number of depths in PSF and object"

        # unsqueeze psf to match (out_channels, groups/in_channels,kH,kW) deffinition
        psf = psf.permute(1, 0, 2, 3)
        # perform group convolution, in order to convolve each depth with the correspondent
        # object plane

        # Compute amount of padding such that the output image is the size of the PSF
        object_pad = [psf.shape[n] - real_object.shape[n] // 2 - 1 for n in range(2, 4)] 
        real_object = f.pad(
            real_object,
            [object_pad[-1], object_pad[-1], object_pad[-2], object_pad[-2]],
        )

        if not is_backwards:
            psf = psf.flip(2).flip(3)
        # Pytorch conv2d is actually coorrelation, hence we need to fil the psf along x and y axis
        vol_convolved = torch.nn.functional.conv2d(
            real_object, psf.float(), padding=0, groups=real_object.shape[1]
        )

        return vol_convolved

    def apply_space_invariant_psf_fourier(self, real_object, psf, is_backwards=False, full_psf_graph=False):
        # If the real_object is an image, replicate it to match the psf number of depths. This only for a forward projection
        if real_object.shape[1]==1 and is_backwards==True:
            real_object = real_object.repeat(1, psf.shape[1], 1, 1)

        assert (    
            psf.shape[1] == real_object.shape[1]
        ), "Different number of depths in PSF and object"
        # assert (
        #     real_object.shape[-1] % 2 == 1 and real_object.shape[-2] % 2 == 1 and\
        #     psf.shape[-1] % 2 == 1 and psf.shape[-2] % 2 == 1
        # ), "Odd size PSF and Volume needed for Fourier convolution"
        # Compute OTF, this happens only once even though this function is called multiple times
        # Padding is performed, so the fourier convolution is equivalent to regular convolution
        self.padSizesVol, self.padSizesPSF, OTF = self.setup_fft_conv(psf, real_object, full_psf_graph=full_psf_graph)

        if is_backwards:
            # Transpose the kernel
            OTF = torch.real(OTF) - 1j * torch.imag(OTF)

        real_object = f.pad(real_object, self.padSizesVol)
        real_object_fft = torch.fft.rfft2(real_object)

        conv_res = co.batch_fftshift2d_real(
            torch.fft.irfft2(real_object_fft * OTF)
        )

        # Unpad result to be the same size as the PSF
        conv_res = f.pad(conv_res, [-k for k in self.padSizesPSF])

        return conv_res

    def normalize_psf(self, psf):
        if psf.ndim == 6:  # LF psf
            for n_depth in range(psf.shape[1]):
                for x1 in range(psf.shape[2]):
                    for x2 in range(psf.shape[3]):
                        psf[0, n_depth, x1, x2, :, :] /= psf[
                            0, n_depth, x1, x2, :, :
                        ].sum()
        else:
            for n_depth in range(psf.shape[1]):
                psf[0, n_depth, :, :] /= psf[0, n_depth, :, :].sum()
        return psf

    def setup_fft_conv(self, PSF, volume, full_psf_graph=False):
        if not self.fft_paddings_ready:
            volume_shape = volume.shape
            # Compute proper padding
            fullSize = torch.tensor(volume_shape[2:]) + torch.tensor(PSF.shape[2:])
            padSizeA = fullSize - torch.tensor(volume_shape[2:])
            padSizesVol = torch.zeros(4, dtype=int)
            padSizesVol[0::2] = torch.floor(padSizeA / 2.0)
            padSizesVol[1::2] = torch.ceil(padSizeA / 2.0)
            self.padSizesVol = list(padSizesVol.numpy()[::-1])

            padSizeB = fullSize - torch.tensor(PSF.shape[2:])
            padSizesPSF = torch.zeros(4, dtype=int)
            padSizesPSF[0::2] = torch.floor(padSizeB / 2.0)
            padSizesPSF[1::2] = torch.ceil(padSizeB / 2.0)
            self.padSizesPSF = list(padSizesPSF.numpy()[::-1])
            # Pad PSF
            OTF_tmp = f.pad(PSF, self.padSizesPSF)
            OFT_tmp_second = torch.fft.rfft2(OTF_tmp)
            # TODO: Why do we need a detach()??
            if full_psf_graph:
                self.OTF = OFT_tmp_second
            else:
                self.OTF = OFT_tmp_second.detach()
            # Set flag indicating that it's ready
            self.fft_paddings_ready = True

        return self.padSizesVol, self.padSizesPSF, self.OTF.clone()
