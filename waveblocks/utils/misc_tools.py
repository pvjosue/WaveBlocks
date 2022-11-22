# Python imports
from math import exp
import argparse
import os
import sys
import h5py
import numpy as np
import os,glob
from PIL import Image

# Third party libraries imports
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as f


class Dataset(object):
    def __init__(
        self,
        fname,
        randomSeed=None,
        img_indices=None,
        fov=None,
        neighShape=1,
        keep_imgs=False,
        random_imgs=False,
        center_region=None,
        get_full_imgs=False,
        signal_power=0,
    ):
        self.fname = fname
        self.tables = h5py.File(self.fname, "r")  # , dri   ver="H5FD_CORE")
        self.neighShape = neighShape
        self.LFShape = self.tables.root.LFData.shape
        self.volShape = self.tables.root.volData.shape
        self.tilesPerImage = self.LFShape[2:4]
        self.keep_imgs = keep_imgs
        self.nPatchesPerImg = self.tilesPerImage[0] * self.tilesPerImage[1]
        self.randomSeed = randomSeed
        self.nImagesInDB = self.LFShape[-1]
        self.getFullImgs = get_full_imgs
        self.fov = fov
        self.nDepths = self.volShape[2]
        if randomSeed is not None:
            torch.manual_seed(randomSeed)
        if fov is None:
            fov = 9
        self.centerRegion = center_region

        # Defines ranges to use accross each dimension
        LF_ix_y = list(range(0, self.LFShape[2]))
        LF_ix_x = list(range(0, self.LFShape[3]))
        vol_ix_y = list(range(0, self.volShape[0]))
        vol_ix_x = list(range(0, self.volShape[1]))

        # If a center region of the images is desired, crop
        if center_region is not None:
            LF_ix_y = list(
                range(
                    self.LFShape[2] // 2 - center_region // 2,
                    self.LFShape[2] // 2 + center_region // 2,
                )
            )
            LF_ix_x = list(
                range(
                    self.LFShape[3] // 2 - center_region // 2,
                    self.LFShape[3] // 2 + center_region // 2,
                )
            )
            vol_ix_y = list(
                range(
                    self.volShape[0] // 2 - (center_region * self.LFShape[0]) // 2,
                    self.volShape[0] // 2 + (center_region * self.LFShape[0]) // 2,
                )
            )
            vol_ix_x = list(
                range(
                    self.volShape[1] // 2 - (center_region * self.LFShape[1]) // 2,
                    self.volShape[1] // 2 + (center_region * self.LFShape[1]) // 2,
                )
            )
            self.tilesPerImage = [center_region, center_region]
            self.nPatchesPerImg = self.tilesPerImage[0] * self.tilesPerImage[1]
            self.LFShape = tuple(
                [
                    self.LFShape[0],
                    self.LFShape[1],
                    center_region,
                    center_region,
                    self.LFShape[-1],
                ]
            )
            self.volShape = tuple(
                [
                    self.LFShape[0] * center_region,
                    self.LFShape[1] * center_region,
                    self.volShape[-2],
                    self.volShape[-1],
                ]
            )

        self.LFSideLenght = fov + neighShape - 1
        self.VolSideLength = self.neighShape * self.LFShape[0]

        # Use images either suggested by user or all images
        if img_indices is None:
            self.img_indices = range(0, self.nImagesInDB)
        else:
            self.img_indices = img_indices

        # Randomize images
        if random_imgs:
            self.img_indices = torch.randperm(int(self.nImagesInDB))

        self.nImagesToUse = len(self.img_indices)

        self.nPatches = self.nPatchesPerImg * self.nImagesToUse

        # Compute padding
        fov_half = self.fov // 2
        if self.getFullImgs:
            neighShapeHalf = self.neighShape // 2
            startOffset = fov_half
            paddedLFSize = self.LFShape[:2] + tuple(
                [self.LFShape[2] + 2 * startOffset, self.LFShape[3] + 2 * startOffset]
            )
            paddedVolSize = self.volShape[0:3]
        else:
            neighShapeHalf = self.neighShape // 2
            startOffset = fov_half + neighShapeHalf
            paddedLFSize = self.LFShape[:2] + tuple(
                [self.LFShape[2] + 2 * startOffset, self.LFShape[3] + 2 * startOffset]
            )
            paddedVolSize = tuple(
                [
                    self.volShape[0] + 2 * neighShapeHalf * self.LFShape[0],
                    self.volShape[1] + 2 * neighShapeHalf * self.LFShape[1],
                    self.volShape[2],
                ]
            )

        self.LFFull = torch.zeros(
            paddedLFSize + tuple([self.nImagesToUse]), dtype=torch.uint8
        )
        self.VolFull = torch.zeros(
            paddedVolSize + tuple([self.nImagesToUse]), dtype=torch.uint8
        )

        print("Loading img:  ", end=" ")
        for nImg, imgIx in enumerate(self.img_indices):
            print(str(imgIx), end=" ")
            # Load data from database
            currLF = torch.tensor(
                self.tables.root.LFData[:, :, :, :, imgIx], dtype=torch.uint8
            )
            currVol = torch.tensor(
                self.tables.root.volData[:, :, :, imgIx], dtype=torch.uint8
            )
            currLF = currLF[:, :, LF_ix_y, :]
            currLF = currLF[:, :, :, LF_ix_x]
            currVol = currVol[vol_ix_y, :, :]
            currVol = currVol[:, vol_ix_x, :]
            # Pad with zeros borders
            currLF = f.pad(
                currLF, (startOffset, startOffset, startOffset, startOffset, 0, 0, 0, 0)
            )
            if not self.getFullImgs:
                currVol = f.pad(
                    currVol,
                    (
                        0,
                        0,
                        neighShapeHalf * self.LFShape[1],
                        neighShapeHalf * self.LFShape[1],
                        neighShapeHalf * self.LFShape[0],
                        neighShapeHalf * self.LFShape[0],
                    ),
                )

            if signal_power != 0:
                maxCurr = currLF.max().float()
                currLF = currLF.float()
                currLF = torch.abs(torch.poisson(signal_power * currLF / currLF.max()))
                # inputGPU += torch.FloatTensor(inputGPU.shape).uniform_(0, args.signal_power).to(device)
                currLF /= currLF.max()
                currLF = (currLF * maxCurr).type(torch.uint8)

                maxCurr = currVol.max().float()
                currVol = currVol.float()
                currVol = torch.abs(
                    torch.poisson(signal_power * currVol / currVol.max())
                )

                currVol /= currVol.max()
                currVol = (currVol * maxCurr).type(torch.uint8)

            self.LFFull[:, :, :, :, nImg] = currLF
            self.VolFull[:, :, :, nImg] = currVol

        self.volMax = self.VolFull.max()
        self.LFMax = self.LFFull.max()
        self.VolDims = [
            neighShape * self.LFShape[0],
            neighShape * self.LFShape[1],
            self.volShape[2],
        ]
        self.LFDims = [
            self.LFShape[0],
            self.LFShape[1],
            self.LFSideLenght,
            self.LFSideLenght,
        ]

        if self.getFullImgs:
            self.VolDims = self.volShape[0:3]
            self.LFDims = self.LFShape[0:4]
            self.nPatches = len(self.img_indices)
        self.tables.close()

    def __getitem__(self, index):
        # Fetch full image or patches
        if self.getFullImgs:
            currLFPatch = self.LFFull[:, :, :, :, index].unsqueeze(0)
            currVolPatch = self.VolFull[:, :, :, index].unsqueeze(0)
        else:
            nImg = index // self.nPatchesPerImg
            nPatch = index - nImg * self.nPatchesPerImg
            yLF = nPatch // self.LFShape[3]
            xLF = nPatch % self.LFShape[3]
            yVol = yLF * self.LFShape[0]
            xVol = xLF * self.LFShape[1]

            # Crop current patch
            currLFPatch = self.LFFull[
                :, :, yLF : yLF + self.LFSideLenght, xLF : xLF + self.LFSideLenght, nImg
            ].unsqueeze(0)
            currVolPatch = self.VolFull[
                yVol : yVol + self.VolSideLength,
                xVol : xVol + self.VolSideLength,
                :,
                nImg,
            ].unsqueeze(0)

        return currLFPatch, currVolPatch

    def __len__(self):
        return self.nPatches

    def get_n_depths(self):
        return self.VolDims[-1]

    def get_max(self):
        return self.LFMax, self.volMax

    def __shape__(self):
        return self.VolDims, self.LFDims

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def resize_volume(volume, volume_new_size):
    # Check if the volume is a 3D tensor or a 2D with depths in channels
    if volume.shape[1]>1:
        volume = volume.permute(0,2,3,1).unsqueeze(1)
    assert len(volume_new_size)==3
    volume = torch.nn.functional.interpolate(
        volume, volume_new_size, mode='nearest'
    )
    volume = volume[:,0,...].permute(0, 3, 1, 2).contiguous()
    return volume

def read_tiff_stack(filename, out_datatype=np.float16):
        try:
            max_val = np.iinfo(out_datatype).max
        except:
            max_val = np.finfo(out_datatype).max

        dataset = Image.open(filename)
        h,w = np.shape(dataset)
        tiffarray = np.zeros([h,w,dataset.n_frames], dtype=out_datatype)
        for i in range(dataset.n_frames):
            dataset.seek(i)
            img =  np.array(dataset)
            img = np.nan_to_num(img)
            # img[img>=max_val/2] = max_val/2
            tiffarray[:,:,i] = img.astype(out_datatype)
        return torch.from_numpy(tiffarray)


def load_process_volume(data_path, volume_new_size=[], dark_current_ths=0.15, norm='max', channel_order='zxy', device="cpu"):
    # Check if data_path is a volume or a path
    if isinstance(data_path, str):
        if 'h5' in data_path or 'hdf5' in data_path:
            vol_file = tables.open_file(data_path, "r", driver="H5FD_CORE")
            gt_volume = (
                torch.tensor(vol_file.root.fish_phantom)
            )
            vol_file.close()
        elif 'tif' in data_path:
            gt_volume = read_tiff_stack(data_path, np.float32).to(device)#imread(data_path).squeeze().astype(np.float32)

            # gt_volume = torch.from_numpy(gt_volume).to(device)
        else:
            raise NotImplementedError
    else: # is a tensor
        gt_volume = data_path

    if gt_volume.ndim == 3:
        if channel_order=='xyz':
            gt_volume = gt_volume.permute(2, 1, 0)
        if channel_order=='yxz':
            gt_volume = gt_volume.permute(2, 0, 1)
        gt_volume = gt_volume.unsqueeze(0).to(device)
            
    out_volume = resize_volume(gt_volume.float(), volume_new_size)
    
    out_volume /= out_volume.max()
    out_volume[out_volume < dark_current_ths] = 0
    if norm=='std':
        mean,std = torch.std_mean(out_volume)
        out_volume = (out_volume-mean)/std

    return out_volume.type_as(gt_volume)


class MicroscopeSimulatorDataset(torch.utils.data.Dataset):
    def __init__(self, microscope, vols_path, volumes_to_load=[], volume_reshape_size=[], volume_ths=0.15, vol_norm='max', pre_simulate=True):
        self.microscope = microscope
        self.device = microscope.get_device()
        self.simulated = pre_simulate

        if len(volumes_to_load) == 0:
            self.all_files = sorted(glob.glob(vols_path)) 
        else:
            self.all_files = [sorted(glob.glob(vols_path))[volumes_to_load[i]] for i in range(len(volumes_to_load))]

        # How many samples?
        self.n_samples = len(self.all_files)

        # Load a single volume and create a proper storage
        vol_sample = load_process_volume(self.all_files[0], volume_reshape_size, volume_ths, norm=vol_norm)
        # Create storage
        self.volumes = torch.zeros(self.n_samples,*list(vol_sample.shape[1:]))

        # Create image storage
        img_sample = microscope(vol_sample.to(self.device))
        self.images = torch.zeros(self.n_samples,*list(img_sample.shape[1:]))

        for n_sample in range(self.n_samples):
            self.volumes[n_sample,...] = load_process_volume(self.all_files[0], volume_reshape_size, volume_ths, norm=vol_norm, channel_order='yxz')[0,...]
            if pre_simulate:
                self.images[n_sample,...] = microscope(self.volumes[n_sample,...].unsqueeze(0).to(self.device))[0,...]

        return
    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples
    def get_n_depths(self):
        return self.volumes.shape[1]

    def __getitem__(self, index):
        if not self.simulated:
            image = self.microscope(self.volumes[index,...].unsqueeze(0).to(self.device))[0,...]
        else:
            image = self.images[index,...]
        
        return image, self.volumes[index,...],0


def convert3Dto2DTiles(x, lateralTile):
    nDepths = x.shape[-1]
    volSides = x.shape[-3:-1]
    nChans = x.shape[1]
    verticalTile = (
        x.permute(0, 1, 4, 2, 3)
        .contiguous()
        .view(-1, nChans, volSides[0] * nDepths, volSides[1])
    )
    currPred = verticalTile[:, :, 0 : volSides[0] * lateralTile[0], :]
    for k in range(1, lateralTile[1]):
        currPred = torch.cat(
            (
                currPred,
                verticalTile[
                    :,
                    :,
                    (lateralTile[0] * volSides[0] * k) : (
                        lateralTile[0] * volSides[0] * (k + 1)
                    ),
                    :,
                ],
            ),
            dim=3,
        )
    return currPred


def convert4Dto3DTiles(x, lateralTile):
    nDepths = x.shape[-1]
    volSide = x.shape[-2]
    nSamples = x.shape[0]
    verticalTile = (
        x.permute(1, 0, 2, 3).contiguous().view(volSide, volSide * nSamples, nDepths)
    )
    currPred = verticalTile[:, 0 : volSide * lateralTile[0], :]
    for k in range(1, lateralTile[1]):
        currPred = torch.cat(
            (
                currPred,
                verticalTile[
                    :,
                    (lateralTile[0] * volSide * k) : (
                        lateralTile[0] * volSide * (k + 1)
                    ),
                    :,
                ],
            ),
            dim=0,
        )
    return currPred


def LF2Spatial(xIn, LFSize):
    xShape = xIn.shape
    x = xIn
    if xIn.ndimension() == 6:
        x = (
            xIn.permute((0, 1, 4, 2, 5, 3))
            .contiguous()
            .view(xShape[0], xShape[1], LFSize[0] * LFSize[2], LFSize[1] * LFSize[3])
        )
    if xIn.ndimension() == 4:
        x = (
            xIn.view(xShape[0], xShape[1], LFSize[2], LFSize[0], LFSize[3], LFSize[1])
            .permute((0, 1, 3, 5, 2, 4))
            .contiguous()
        )
    return x


def LF2Angular(xIn, LFSize):
    xShape = xIn.shape
    x = xIn
    if xIn.ndimension() == 6:
        x = (
            xIn.permute((0, 1, 2, 4, 3, 5))
            .contiguous()
            .view(xShape[0], xShape[1], LFSize[0] * LFSize[2], LFSize[1] * LFSize[3])
        )
    if xIn.ndimension() == 4:
        x = (
            xIn.view(xShape[0], xShape[1], LFSize[0], LFSize[2], LFSize[1], LFSize[3])
            .permute((0, 1, 2, 4, 3, 5))
            .contiguous()
        )
    return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight.data, 1 / len(m.weight.data))
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.constant_(m.weight.data, 1 / len(m.weight.data))


def getThreads():
    if sys.platform == "win32":
        return int(os.environ["NUMBER_OF_PROCESSORS"])
    else:
        return int(os.popen("grep -c cores /proc/cpuinfo").read())


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def imadjust(x, a, b, c, d, gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    mask = (y > 0).float()
    y = torch.mul(y, mask)
    return y


######## SSIM


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = f.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = f.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        f.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        f.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        f.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def volume_2_projections(vol, proj_type=torch.max, depths_in_ch=False):
    # vol = vol.detach()
    if depths_in_ch:
        vol = vol.permute(0,2,3,1).unsqueeze(1)
    vol_size = vol.shape
    if proj_type is torch.max or proj_type is torch.min:
        x_projection, _ = proj_type(vol.float().cpu(), dim=2)
        y_projection, _ = proj_type(vol.float().cpu(), dim=3)
        z_projection, _ = proj_type(vol.float().cpu(), dim=4)
    elif proj_type is torch.sum:
        x_projection = proj_type(vol.float().cpu(), dim=2)
        y_projection = proj_type(vol.float().cpu(), dim=3)
        z_projection = proj_type(vol.float().cpu(), dim=4)

    out_img = torch.zeros(
        vol_size[0], vol_size[1], vol_size[2] + vol_size[4], vol_size[3] + vol_size[4]
    )

    out_img[:, :, : vol_size[2], : vol_size[3]] = z_projection
    out_img[:, :, vol_size[2] :, : vol_size[3]] = x_projection.permute(0, 1, 3, 2)
    out_img[:, :, : vol_size[2], vol_size[3] :] = y_projection

    # Draw white lines
    out_img[:, :, vol_size[2], ...] = z_projection.max()
    out_img[:, :, :, vol_size[3], ...] = z_projection.max()

    return out_img

# Loss function to noramlize two point spread functions to each other, depthwhise
def normalize_PSF_pair(psf_in, psf_gt, norm_to_use=torch.sum, chans_to_norm=[]):
    # psf_in: the psf to normalize
    # psf_gt: the psf of reference
    # chans_to_norm: which channels to norm
    # if no psf_gt is provided, every channel will be normalized to norm_to_use(ch)=1

    n_chans = psf_in.shape[1]
    if len(chans_to_norm)==0:
        chans_to_norm = [0,]+list(range(2, psf_in.dim()))

    channelwise_norm = torch.ones([psf_in.shape[0]])

    # if a reference was provided, update the norm per channel
    if torch.is_tensor(psf_gt):
        channelwise_norm = norm_to_use(psf_gt,chans_to_norm, keepdim=True)
    
    channelwise_norm_psf_in = norm_to_use(psf_in,chans_to_norm, keepdim=True)

    # normalize psf_in
    psf_out = psf_in / channelwise_norm_psf_in

    # apply new normalization
    psf_out = psf_out * channelwise_norm

    return psf_out