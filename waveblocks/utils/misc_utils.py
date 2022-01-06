import torch
import torchvision as tv
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as TF
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu as otsu
import h5py
import gc
import re
import numpy as np
from sklearn.decomposition import PCA
import findpeaks
import pickle
import math 
# from tsne_torch import TorchTSNE as TSNE
from sklearn.manifold import TSNE

"""
Extracted the shpere implemention from the package pymrt since the package had errors

Link: https://pypi.org/project/pymrt/
Repository: https://bitbucket.org/norok2/pymrt/src/master/

"""



def combine_iter_len(items, combine=max):
    """
    Determine the maximum length of an item within a collection of items.

    Args:
        items (iterable): The collection of items to inspect.
        combine (callable): The combination method.

    Returns:
        num (int): The combined length of the collection.
            If none of the items are iterable, the result is `1`.

    Examples:
        >>> a = list(range(10))
        >>> b = tuple(range(5))
        >>> c = set(range(20))
        >>> combine_iter_len((a, b, c))
        20
        >>> combine_iter_len((a, b, c), min)
        5
        >>> combine_iter_len((1, a))
        10
    """
    num = None
    for val in items:
        try:
            iter(val)
        except TypeError:
            pass
        else:
            if num is None:
                num = len(val)
            else:
                num = combine(len(val), num)
    if num is None:
        num = 1
    return num


# ======================================================================
def auto_repeat(obj, n, force=False, check=False):
    """
    Automatically repeat the specified object n times.

    If the object is not iterable, a tuple with the specified size is returned.
    If the object is iterable, the object is left untouched.

    Args:
        obj: The object to operate with.
        n (int): The length of the output object.
        force (bool): Force the repetition, even if the object is iterable.
        check (bool): Ensure that the object has length n.

    Returns:
        val (tuple): Returns obj repeated n times.

    Raises:
        AssertionError: If force is True and the object does not have length n.

    Examples:
        >>> auto_repeat(1, 3)
        (1, 1, 1)
        >>> auto_repeat([1], 3)
        [1]
        >>> auto_repeat([1, 3], 2)
        [1, 3]
        >>> auto_repeat([1, 3], 2, True)
        ([1, 3], [1, 3])
        >>> auto_repeat([1, 2, 3], 2, True, True)
        ([1, 2, 3], [1, 2, 3])
        >>> auto_repeat([1, 2, 3], 2, False, True)
        Traceback (most recent call last):
            ...
        AssertionError
    """
    try:
        iter(obj)
    except TypeError:
        force = True
    finally:
        if force:
            obj = (obj,) * n
    if check:
        assert len(obj) == n
    return obj


# ======================================================================
def grid_coord(shape, position=0.5, is_relative=True, use_int=True, dense=False):
    """
    Calculate the generic x_i coordinates for N-dim operations.

    Args:
        shape (iterable[int]): The shape of the mask in px.
        position (float|iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool): Interpret origin as relative.
        dense (bool): Determine the shape of the mesh-grid arrays.
        use_int (bool): Force interger values for the coordinates.

    Returns:
        coord (list[np.ndarray]): mesh-grid ndarrays.
            The shape is identical if dense is True, otherwise only one
            dimension is larger than 1.

    Examples:
        >>> grid_coord((4, 4))
        [array([[-2],
               [-1],
               [ 0],
               [ 1]]), array([[-2, -1,  0,  1]])]
        >>> grid_coord((5, 5))
        [array([[-2],
               [-1],
               [ 0],
               [ 1],
               [ 2]]), array([[-2, -1,  0,  1,  2]])]
        >>> grid_coord((2, 2))
        [array([[-1],
               [ 0]]), array([[-1,  0]])]
        >>> grid_coord((2, 2), dense=True)
        array([[[-1, -1],
                [ 0,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]])
        >>> grid_coord((2, 3), position=(0.0, 0.5))
        [array([[0],
               [1]]), array([[-1,  0,  1]])]
        >>> grid_coord((3, 9), position=(1, 4), is_relative=False)
        [array([[-1],
               [ 0],
               [ 1]]), array([[-4, -3, -2, -1,  0,  1,  2,  3,  4]])]
        >>> grid_coord((3, 9), position=0.2, is_relative=True)
        [array([[0],
               [1],
               [2]]), array([[-1,  0,  1,  2,  3,  4,  5,  6,  7]])]
        >>> grid_coord((4, 4), use_int=False)
        [array([[-1.5],
               [-0.5],
               [ 0.5],
               [ 1.5]]), array([[-1.5, -0.5,  0.5,  1.5]])]
        >>> grid_coord((5, 5), use_int=False)
        [array([[-2.],
               [-1.],
               [ 0.],
               [ 1.],
               [ 2.]]), array([[-2., -1.,  0.,  1.,  2.]])]
        >>> grid_coord((2, 3), position=(0.0, 0.0), use_int=False)
        [array([[ 0.],
               [ 1.]]), array([[ 0.,  1.,  2.]])]
    """
    position = coord(shape, position, is_relative, use_int)
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    return np.ogrid[grid] if not dense else np.mgrid[grid]


# ======================================================================
def coord(shape, position=0.5, is_relative=True, use_int=True):
    """
    Calculate the coordinate in a given shape for a specified position.

    Args:
        shape (iterable[int]): The shape of the mask in px.
        position (float|iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool): Interpret origin as relative.
        use_int (bool): Force interger values for the coordinates.

    Returns:
        position (list): The coordinate in the shape.

    Examples:
        >>> coord((5, 5))
        (2, 2)
        >>> coord((4, 4))
        (2, 2)
        >>> coord((5, 5), 3, False)
        (3, 3)
    """
    position = auto_repeat(position, len(shape), check=True)
    if is_relative:
        if use_int:
            position = tuple(int(scale(x, (0, dim))) for x, dim in zip(position, shape))
        else:
            position = tuple(scale(x, (0, dim - 1)) for x, dim in zip(position, shape))
    elif any([not isinstance(x, int) for x in position]) and use_int:
        raise TypeError("Absolute origin must be integer.")
    return position


# ======================================================================
def scale(val, out_interval=None, in_interval=None):
    """
    Linear convert the value from input interval to output interval

    Args:
        val (float|np.ndarray): Value(s) to convert.
        out_interval (float,float): Interval of the output value(s).
            If None, set to: (0, 1).
        in_interval (float,float): Interval of the input value(s).
            If None, and val is iterable, it is calculated as:
            (min(val), max(val)), otherwise set to: (0, 1).

    Returns:
        val (float|np.ndarray): The converted value(s).

    Examples:
        >>> scale(100, (0, 1000), (0, 100))
        1000.0
        >>> scale(50, (0, 1000), (-100, 100))
        750.0
        >>> scale(50, (0, 10), (0, 1))
        500.0
        >>> scale(0.5, (-10, 10))
        0.0
        >>> scale(np.pi / 3, (0, 180), (0, np.pi))
        60.0
        >>> scale(np.arange(5), (0, 1))
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
        >>> scale(np.arange(6), (0, 10))
        array([  0.,   2.,   4.,   6.,   8.,  10.])
        >>> scale(np.arange(6), (0, 10), (0, 2))
        array([  0.,   5.,  10.,  15.,  20.,  25.])
    """
    if in_interval:
        in_min, in_max = sorted(in_interval)
    elif isinstance(val, np.ndarray):
        in_min, in_max = minmax(val)
    else:
        in_min, in_max = (0, 1)
    if out_interval:
        out_min, out_max = sorted(out_interval)
    else:
        out_min, out_max = (0, 1)
    return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


# ======================================================================
def minmax(arr):
    """
    Calculate the minimum and maximum of an array: (min, max).

    Args:
        arr (np.ndarray): The input array.

    Returns:
        min (float): the minimum value of the array
        max (float): the maximum value of the array

    Examples:
        >>> minmax(np.arange(10))
        (0, 9)
    """
    return np.min(arr), np.max(arr)



def pad_img_to_min(image):
    min_size = min(image.shape[-2:])
    img_pad = [min_size-image.shape[-1], min_size-image.shape[-2]]
    img_pad = [img_pad[0]//2, img_pad[0]//2, img_pad[1],img_pad[1]]
    image = F.pad(image.unsqueeze(0).unsqueeze(0), img_pad)[0,0]
    return image

# Prepare a volume to be shown in tensorboard as an image
def volume_2_tensorboard(vol, batch_index=0, z_scaling=2):
    vol = vol.detach()
    # expecting dims to be [batch, depth, xDim, yDim]
    xyProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(1).cpu().data, normalize=True, scale_each=True)
    
    # interpolate z in case that there are not many depths
    vol = torch.nn.functional.interpolate(vol.permute(0,2,3,1).unsqueeze(1), (vol.shape[2], vol.shape[3], vol.shape[1]*z_scaling))
    yzProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(3).cpu().data, normalize=True, scale_each=True)
    xzProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(2).cpu().data, normalize=True, scale_each=True)

    return xzProj, yzProj, xyProj

# Convert volume to single 2D MIP image, input [batch,1,xDim,yDim,zDim]
def volume_2_projections(vol, proj_type=torch.max):
    # vol = vol.detach()
    vol_size = vol.shape
    if proj_type is torch.max or proj_type is torch.min:
        x_projection,_ = proj_type(vol.float().cpu(), dim=2)
        y_projection,_ = proj_type(vol.float().cpu(), dim=3)
        z_projection,_ = proj_type(vol.float().cpu(), dim=4)
    elif proj_type is torch.sum:
        x_projection = proj_type(vol.float().cpu(), dim=2)
        y_projection = proj_type(vol.float().cpu(), dim=3)
        z_projection = proj_type(vol.float().cpu(), dim=4)

    out_img = torch.zeros(vol_size[0], vol_size[1], vol_size[2] + vol_size[4], vol_size[3] + vol_size[4])

    out_img[:,:,:vol_size[2], :vol_size[3]] = z_projection
    out_img[:,:,vol_size[2]:, :vol_size[3]] = x_projection.permute(0,1,3,2)
    out_img[:,:,:vol_size[2], vol_size[3]:] = y_projection

    # Draw white lines
    out_img[:,:,vol_size[2],...] = z_projection.max()/2
    out_img[:,:,:,vol_size[3],...] = z_projection.max()/2

    return out_img


# Aid functions for shiftfft2
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
    
def batch_fftshift2d_real(x):
    out = x
    for dim in range(2, len(out.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        out = roll_n(out, axis=dim, n=n_shift)
    return out  

# FFT convolution, the kernel fft can be precomputed
def fft_conv(A,B, fullSize, Bshape=[],B_precomputed=False):
    import torch.fft
    nDims = A.ndim-2
    # fullSize = torch.tensor(A.shape[2:]) + Bshape
    # fullSize = torch.pow(2, torch.ceil(torch.log(fullSize.float())/torch.log(torch.tensor(2.0)))-1)
    padSizeA = (fullSize - torch.tensor(A.shape[2:]))
    padSizesA = torch.zeros(2*nDims,dtype=int)
    padSizesA[0::2] = torch.floor(padSizeA/2.0)
    padSizesA[1::2] = torch.ceil(padSizeA/2.0)
    padSizesA = list(padSizesA.numpy()[::-1])

    A_padded = F.pad(A,padSizesA)
    Afft = torch.fft.rfft2(A_padded)
    if B_precomputed:
        return batch_fftshift2d_real(torch.fft.irfft2( Afft * B.detach()))
    else:
        padSizeB = (fullSize - torch.tensor(B.shape[2:]))
        padSizesB = torch.zeros(2*nDims,dtype=int)
        padSizesB[0::2] = torch.floor(padSizeB/2.0)
        padSizesB[1::2] = torch.ceil(padSizeB/2.0)
        padSizesB = list(padSizesB.numpy()[::-1])
        B_padded = F.pad(B,padSizesB)
        Bfft = torch.fft.rfft2(B_padded)
        return batch_fftshift2d_real(torch.fft.irfft2( Afft * Bfft.detach())), Bfft.detach()


def reprojection_loss_camera(gt_imgs, prediction, PSF, camera, dataset, device="cpu"):
    out_type = gt_imgs.type()
    camera = camera.to(device)
    reprojection = camera(prediction.to(device), PSF.to(device))
    reprojection_views = dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[0,0,...]
    loss = F.mse_loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

    return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)

def reprojection_loss(gt_imgs, prediction, OTF, psf_shape, dataset, n_split=20, device="cpu", loss=F.mse_loss):
    out_type = gt_imgs.type()
    batch_size = prediction.shape[0]
    reprojection = fft_conv_split(prediction[0,...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True, device=device)

    reprojection_views = torch.zeros_like(gt_imgs)
    reprojection_views[0,...] = reprojection #dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[0,0,...]

    # full_reprojection = reprojection.detach()
    # reprojection_views = reprojection_views.unsqueeze(0).repeat(batch_size,1,1,1)
    for nSample in range(1,batch_size):
        reprojection = fft_conv_split(prediction[nSample,...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True, device=device)
        reprojection_views[nSample,...] = reprojection#dataset.extract_views(reprojection, dataset.lenslet_coords, dataset.subimage_shape)[0,0,...]
        # full_reprojection += reprojection.detach()

    # gt_imgs /= gt_imgs.float().max()
    # reprojection_views /= reprojection_views.float().max()
    # loss = F.mse_loss(gt_imgs[gt_imgs!=0].to(device), reprojection_views[gt_imgs!=0])
    #loss = (1-gt_imgs[reprojection_views!=0]/reprojection_views[reprojection_views!=0]).abs().mean()
    loss = loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

    return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)

# Split an fft convolution into batches containing different depths
def fft_conv_split(A, B, psf_shape, n_split, B_precomputed=False, device = "cpu"):
    n_depths = A.shape[1]
    
    split_conv = n_depths//n_split
    depths = list(range(n_depths))
    depths = [depths[i:i + split_conv] for i in range(0, n_depths, split_conv)]

    fullSize = torch.tensor(A.shape[2:]) + torch.tensor(psf_shape)
    
    crop_pad = [(psf_shape[i] - fullSize[i])//2 for i in range(0,2)]
    crop_pad = (crop_pad[1], (psf_shape[-1]- fullSize[-1])-crop_pad[1], crop_pad[0], (psf_shape[-2] - fullSize[-2])-crop_pad[0])
    # Crop convolved image to match size of PSF
    img_new = torch.zeros(A.shape[0], 1, psf_shape[0], psf_shape[1], device=device)
    if B_precomputed == False:
        OTF_out = torch.zeros(1, n_depths, fullSize[0], fullSize[1]//2+1, requires_grad=False, dtype=torch.complex64, device=device)
    for n in range(n_split):
        # print(n)
        curr_psf = B[:,depths[n],...].to(device)
        img_curr = fft_conv(A[:,depths[n],...].to(device), curr_psf, fullSize, psf_shape, B_precomputed)
        if B_precomputed == False:
            OTF_out[:,depths[n],...] = img_curr[1]
            img_curr = img_curr[0]
        img_curr = F.pad(img_curr, crop_pad)
        img_new += img_curr[:,:,:psf_shape[0],:psf_shape[1]].sum(1).unsqueeze(1).abs()
    
    if B_precomputed == False:
        return img_new, OTF_out
    return img_new


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    mask = (y>0).float()
    y = torch.mul(y,mask)
    return y


# Apply different normalizations to volumes and images
def normalize_type(LF_views, vols, id=0, mean_imgs=0, std_imgs=1, mean_vols=0, std_vols=1, max_imgs=1, max_vols=1, inverse=False):
    if inverse:
        if id==-1: # No normalization
            return LF_views, vols
        if id==0: # baseline normlization
            return (LF_views) * (2*std_imgs), vols * std_vols + mean_vols
        if id==1: # Standarization of images and volume normalization
            return LF_views * std_imgs + mean_imgs, vols * std_vols + mean_vols
        if id==2: # normalization of both
            return LF_views * max_imgs, vols * max_vols
        if id==3: # normalization of both
            return LF_views * std_imgs, vols * std_vols
    else:
        if id==-1: # No normalization
            return LF_views, vols
        if id==0: # baseline normlization
            return (LF_views) / (2*std_imgs), (vols - mean_vols) / std_vols
        if id==1: # Standarization of images and volume normalization
            return (LF_views - mean_imgs) / std_imgs, (vols - mean_vols) / std_vols
        if id==2: # normalization of both
            return LF_views / max_imgs, vols / max_vols
        if id==3: # normalization of both
            return LF_views / std_imgs, vols / std_vols

class PercentileNormalizer:
    def __init__(self, training_stage_tags_and_n_imgs = {}):

        # Pre-alocate space for min and max percentile
        self.percentile_storage = {}
        for key, value in training_stage_tags_and_n_imgs.items():
            self.percentile_storage[key] = -1*torch.ones(value, 2)
    
    def compute_and_store_percentiles(self, x, stage_id='train', n_sample=[0], percentiles=[0.1,99.9], border_to_remove=10):
        device = x.device
        if x.ndim==4: #is 2D
            x = x[:,:,border_to_remove:-border_to_remove,border_to_remove:-border_to_remove].cpu().detach().numpy()
        else:
            x = x.cpu().detach().numpy()
        pmin,pmax = percentiles
        mi = torch.from_numpy(np.percentile(x, pmin,axis=list(range(1,x.ndim)),keepdims=True)).to(device)
        ma = torch.from_numpy(np.percentile(x, pmax,axis=list(range(1,x.ndim)),keepdims=True)).to(device)

        self.percentile_storage[stage_id][n_sample,0] = mi.squeeze().float().cpu()
        self.percentile_storage[stage_id][n_sample,1] = ma.squeeze().float().cpu()

    def normalize_sample(self, x, stage_id='train', n_sample=[0], percentiles=[0.0,99.9999], clamp=False, inverse=False):
        device = x.device
        if self.percentile_storage[stage_id][n_sample[0],0] == -1:
            self.compute_and_store_percentiles(x, stage_id, n_sample, percentiles)
        mi = self.percentile_storage[stage_id][n_sample,0].to(device).type(x.type())
        ma = self.percentile_storage[stage_id][n_sample,1].to(device).type(x.type())

        # Set dimensions so each batch sample is normalized independently
        mi = mi.reshape((ma.shape[0],) + (len(x.shape)-1)*(1,))#.to(device)
        ma = ma.reshape((ma.shape[0],) + (len(x.shape)-1)*(1,))#.to(device)
        if inverse:
            alpha = ma - mi
            beta  = mi
            return alpha * x + beta
        else:
            if clamp:
                for sampleIx in range(x.shape[0]):
                    curr_sample = x[sampleIx,...]
                    curr_sample[curr_sample>ma[sampleIx,...].squeeze()] = ma[sampleIx,...].squeeze().type(x.type())
            return (x - mi) / ( ma - mi + torch.finfo(x.dtype).tiny )


def norm_percentile(input, original_input, percentiles=[0.1,99.9], inverse=False):
    x = original_input.cpu().detach().numpy()
    pmin,pmax = percentiles
    mi = torch.from_numpy(np.percentile(x, pmin,axis=list(range(1,original_input.ndim)),keepdims=True)).to(input.device)
    ma = torch.from_numpy(np.percentile(x, pmax,axis=list(range(1,original_input.ndim)),keepdims=True)).to(input.device)

    if inverse:
        alpha = ma - mi
        beta  = mi
        return alpha * input + beta
    else:
        return (input - mi) / ( ma - mi + 1e-10 )


def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.flatten(),target.flatten())
    alpha = cov[0,1] / (cov[0,0]+1e-10)
    beta = target.mean() - alpha*x.mean()
    return alpha*x + beta

def loss_laplace(y_true, y_pred):
    """Laplace loss function, where the first half of the channels are the mean and the second half the scales."""
    C = torch.log(2.0)
    n = y_pred.shape[1]//2
    mu    = y_pred[:,:n,...]
    sigma = y_pred[:,n:,...]
    return torch.mean(torch.abs((mu-y_true)/sigma) + torch.log(sigma) + C)

# Random transformation of volume, for augmentation
def transform_volume(currVol, transformParams=None, maxZRoll=180):
    # vol format [B,Z,X,Y]
    if transformParams==None:
        angle, transl, scale, shear = TF.RandomAffine.get_params((-180,180), (0.1,0.1), (0.9,1.1), (0,0), currVol.shape[2:4])
        zRoll = int(maxZRoll*torch.rand(1)-maxZRoll//2)
        transformParams = {'angle':angle, 'transl':transl, 'scale':scale, 'shear':shear, 'zRoll':zRoll}
    
    zRoll = transformParams['zRoll']
    for nVol in range(currVol.shape[0]):
        for nDepth in range(currVol.shape[1]):
            currDepth = TF.functional.to_pil_image(currVol[nVol,nDepth,...].float())
            currDepth = TF.functional.affine(currDepth, transformParams['angle'], transformParams['transl'], transformParams['scale'], transformParams['shear'])
            currVol[nVol,nDepth,...] = TF.functional.to_tensor(currDepth)
    currVol = currVol.roll(zRoll, 1)
    if zRoll>=0:
        currVol[:,0:zRoll,...] = 0
    else:
        currVol[:,zRoll:,...] = 0
    return currVol, transformParams

def plot_param_grads(writer, net, curr_it, prefix=""):
    grads_sum = 0
    for tag, parm in net.named_parameters():
        if parm.grad is not None:
            if torch.isnan(parm.grad.data).any():
                print(F"param: {tag} is nan")
                continue
            if torch.isinf(parm.grad.data).any():
                print(F"param: {tag} is inf")
                continue
            writer.add_histogram(prefix+tag, parm.grad.data.cpu().numpy(), curr_it)
            assert not torch.isnan(parm.grad.sum()), print("NAN in: " + str(tag) + "\t\t")
            grads_sum += parm.grad.sum()
    return grads_sum

def compute_histograms(gt, pred, input_img, n_bins=1000):
    volGTHist = torch.histc(gt, bins=n_bins, max=gt.max().item())
    volPredHist = torch.histc(pred, bins=n_bins, max=pred.max().item())
    inputHist = torch.histc(input_img, bins=n_bins, max=input_img.max().item())
    return volGTHist,volPredHist,inputHist


def match_histogram(source, reference):
    isTorch = False
    source = source / source.max() * reference.max()
    if isinstance(source, torch.Tensor):
        source = source.cpu().numpy()
        isTorch = True
    if isinstance(reference, torch.Tensor):
        reference = reference[:source.shape[0],...].cpu().numpy()

    matched = match_histograms(source, reference, multichannel=False)
    if isTorch:
        matched = torch.from_numpy(matched)
    return matched

def load_PSF(filename, n_depths=120):
    # Load PSF
    try:
        # Check permute
        psfIn = torch.from_numpy(loadmat(filename)['PSF']).permute(2,0,1).unsqueeze(0)
    except:
        psfFile = h5py.File(filename,'r')
        psfIn = torch.from_numpy(psfFile.get('PSF')[:]).permute(0,2,1).unsqueeze(0)

    # Make a square PSF
    psfIn = pad_img_to_min(psfIn)

    # Grab only needed depths
    depths_to_use = torch.linspace(0, psfIn.shape[1], n_depths+2).long()[1:-1]
    # psfIn = psfIn[:, psfIn.shape[1]//2- n_depths//2+1 : psfIn.shape[1]//2- n_depths//2+1 + n_depths, ...]
    psfIn = psfIn[:, depths_to_use, ...]
    # Normalize psfIn such that each depth sum is equal to 1
    for nD in range(psfIn.shape[1]):
        psfIn[:,nD,...] = psfIn[:,nD,...] / psfIn[:,nD,...].sum()
    
    return psfIn

def load_PSF_OTF(filename, vol_size, n_split=20, n_depths=120, downS=1, device="cpu",
                 dark_current=106, calc_max=False, psfIn=None, compute_transpose=False,
                 n_lenslets=29, lenslet_centers_file_out='lenslet_centers_python.txt'):
    # Load PSF
    if psfIn is None:
        psfIn = load_PSF(filename, n_depths)

    if len(lenslet_centers_file_out)==0:
        find_lenslet_centers(psfIn[0,n_depths//2,...].numpy(), n_lenslets=n_lenslets, file_out_name=lenslet_centers_file_out)
    if calc_max:
        psfMaxCoeffs = torch.amax(psfIn, dim=[0,2,3])

    psf_shape = torch.tensor(psfIn.shape[2:])
    vol = torch.rand(1,psfIn.shape[1], vol_size[0], vol_size[1], device=device)
    img, OTF = fft_conv_split(vol, psfIn.float().detach().to(device), psf_shape, n_split=n_split, device=device)
    
    OTF = OTF.detach()

    if compute_transpose:
        OTFt = torch.real(OTF) - 1j * torch.imag(OTF)
        OTF = torch.cat((OTF.unsqueeze(-1), OTFt.unsqueeze(-1)), 4)
    if calc_max:
        return OTF, psf_shape, psfMaxCoeffs
    else:
        return OTF,psf_shape


def find_lenslet_centers(img, n_lenslets=29, file_out_name='lenslet_centers_python.txt'):
    fp2 = findpeaks.findpeaks()
    
    image_divisor = 4 # To find the centers faster
    img = findpeaks.stats.resize(img, size=(img.shape[0]//image_divisor,img.shape[1]//image_divisor))
    results_2 = fp2.fit(img)
    limit_min = fp2.results['persistence'][0:n_lenslets+1]['score'].min()

    # Initialize topology
    fp = findpeaks.findpeaks(method='topology', limit=limit_min)
    # make the fit
    results = fp.fit(img)
    # Make plot
    fp.plot_persistence()
    # fp.plot()
    results = np.ndarray([n_lenslets,2], dtype=int)
    for ix,data in enumerate(fp.results['groups0']):
        results[ix] = np.array(data[0], dtype=int) * image_divisor
    if len(file_out_name) > 0:
        np.savetxt(file_out_name, results, fmt='%d', delimiter='\t')

    return results

# Aid functions for getting information out of directory names
def get_intensity_scale_from_name(name):
    intensity_scale_sparse = re.match(r"^.*_(\d*)outScaleSp",name)
    if intensity_scale_sparse is not None:
        intensity_scale_sparse = int(intensity_scale_sparse.groups()[0])
    else:
        intensity_scale_sparse = 1

    intensity_scale_dense = re.match(r"^.*_(\d*)outScaleD",name)
    if intensity_scale_dense is not None:
        intensity_scale_dense = int(intensity_scale_dense.groups()[0])
    else:
        intensity_scale_dense = 1
    return intensity_scale_dense,intensity_scale_sparse

def get_number_of_frames(name):
    n_frames = re.match(r"^.*_(\d*)timeF",name)
    if n_frames is not None:
        n_frames = int(n_frames.groups()[0])
    else:
        n_frames = 1
    return n_frames


def net_get_params(net):
    if hasattr(net, 'module'):
        return net.module
    else:
        return net

def plot_to_tensorboard(fig):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape((3,) + fig.canvas.get_width_height() )

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    return img

def imshow2D(img, blocking=False):
    plt.imshow(img[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()
def imshow3D(vol, blocking=False):
    plt.imshow(volume_2_projections(vol.permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()
def imshowComplex(vol, blocking=False):
    plt.subplot(1,2,1)
    plt.imshow(volume_2_projections(torch.real(vol).permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    plt.subplot(1,2,2)
    plt.imshow(volume_2_projections(torch.imag(vol).permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()


def log_likelehood(gt_img, forward_img, reg=True):
    if reg:
        reg = forward_img.abs()
    else:
        reg = 0
    return (forward_img - gt_img * (1e-8 + forward_img).log() + reg).mean()


def channel_loss(input, gt):
    channelwise_norm = torch.sum(gt.view(gt.shape[0], gt.shape[1],...))
    channelwise_norm_in = torch.sum(input.view(gt.shape[0], gt.shape[1],...))
    return (channelwise_norm - channelwise_norm_in).abs().mean()

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





# Compute PCA
def fit_pca_to_latent(X, pca=None, n_components=2, pca_type=1):
    '''pca_type referes to: 1: PCA, 2: T-sne'''
    if len(X.shape) != 2:
        X = X.view(X.shape[0],-1)
    X = X.detach().float().cpu().numpy()
    if pca is None:
        if pca_type==1:
            pca = PCA(n_components=n_components)
            pca.fit(X)
        elif pca_type==2:
            pca = TSNE(n_components=n_components)#, perplexity=30, n_iter=1000, verbose=True)
            # pca.fit_transform(X)
        return 0,0,0,pca

    reproj = X
    # Apply dimensionality reduction
    if pca_type==1:
        forward = pca.transform(X)# Project back
        reproj = pca.inverse_transform(forward)
    elif pca_type==2:
        forward = pca.fit_transform(X)# Project back
    
    # Measure distance between projection and GT
    distance = np.mean((X-reproj)**2,axis=1)

    return forward, reproj, distance, pca


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.max_values = None
    def __getitem__(self, i):
        n_dataset = 0
        for d in self.datasets:
            if i>=len(d):
                n_dataset += 1
                i -= len(d)
        return tuple(self.datasets[n_dataset][i])

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def get_statistics(self):
        if len(self.datasets[0].stacked_views.shape)==4: # It has sparse images as well
            all_images = torch.cat(tuple([d.stacked_views[...,0].float().unsqueeze(-1) for d in self.datasets]), dim=-1)
            all_images_s = torch.cat(tuple([d.stacked_views[...,1].float().unsqueeze(-1) for d in self.datasets]), dim=-1)
        else:
            all_images = torch.cat(tuple([d.stacked_views.float().unsqueeze(-1) for d in self.datasets]), dim=-1)
            all_images_s = all_images

        mean_imgs = all_images.mean().type(self.datasets[0].stacked_views.type())
        std_imgs = all_images.std().type(self.datasets[0].stacked_views.type())

        mean_imgs_s = all_images_s.mean().type(self.datasets[0].stacked_views.type())
        std_imgs_s = all_images_s.std().type(self.datasets[0].stacked_views.type())

        all_vols = torch.cat(tuple([d.vols.float().unsqueeze(-1) for d in self.datasets]), dim=-1)
        mean_vols = all_vols.mean().type(self.datasets[0].vols.type())
        std_vols = all_vols.std().type(self.datasets[0].vols.type())

        return mean_imgs, std_imgs, mean_imgs_s, std_imgs_s, mean_vols, std_vols

    def get_max(self):
        if self.max_values is None:
            if len(self.datasets[0].stacked_views.shape)==4: # It has sparse images as well
                all_images = torch.cat(tuple([d.stacked_views[...,0].float().unsqueeze(-1) for d in self.datasets]), dim=-1)
                all_images_s = torch.cat(tuple([d.stacked_views[...,1].float().unsqueeze(-1) for d in self.datasets]), dim=-1)
            else:
                all_images = torch.cat(tuple([d.stacked_views.float().unsqueeze(-1) for d in self.datasets]), dim=-1)
                all_images_s = all_images
            
            all_vols = torch.cat(tuple([d.vols.float().unsqueeze(-1) for d in self.datasets]), dim=-1)

            # Store for later use
            self.max_values = [all_images.max(), all_images_s.max(), all_vols.max()]
        return self.max_values
    
    def normalize_datasets(self):
        if self.max_values is None:
            self.max_values = self.get_max()
        has_sparse = len(self.datasets[0].stacked_views.shape)==4 # It has sparse images as well

        # Normalize datasets
        for d in self.datasets:
            if has_sparse:
                d.stacked_views[...,0] = (d.stacked_views[...,0]/d.stacked_views[...,0].max() * self.max_values[1]).type(d.stacked_views.type())
                d.stacked_views[...,1] = (d.stacked_views[...,1]/d.stacked_views[...,1].max() * self.max_values[0]).type(d.stacked_views.type())
            else:
                d.stacked_views = (d.stacked_views.float()/d.stacked_views.float().max() * self.max_values[0]).type(d.stacked_views.type())
            d.vols = (d.vols.float() / d.vols.float().max() * self.max_values[2]).type(d.vols.type())
        
    def standarize_datasets(self,stats=None):
        if stats is None:
            stats = self.get_statistics()
        
        # Normalize datasets
        for d in self.datasets:
            d.standarize(stats)


def _otsu(data: torch.Tensor, **kwargs) -> float:
    """
    Returns an intensity threshold for an image that separates it
    into backgorund and foreground pixels.

    The implementation uses Otsu's method, which assumes a GMM with
    2 components but uses some heuristic to maximize the variance
    differences. The input data is shaped into a 2D image for the
    purpose of evaluating the threshold value.

    Args:
        data: Pytorch tensor containing the data.

    Returns:
        Threshold value determined via Otsu's method.
    """
    h = 2 ** int(1 + math.log2(data.shape[2]) / 2)
    fake_img = data.view(h, -1).cpu().numpy()
    return otsu(fake_img, h)

def enable_test_dropout(model):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def compute_certainity(predictions, dropout_val=0.25, weight_decay=0.0, length_scale=1):
    mean = torch.mean(predictions, dim=0)
    # predictions /= mean.unsqueeze(0).repeat(predictions.shape[0],1,1,1)
    
    var = torch.var(predictions, dim=0, unbiased=True)
    var /= mean
    var[torch.isnan(var)] = 0
    tau = 0
    if weight_decay != 0.0:
        tau = length_scale**2 * (1 - dropout_val) / (2 * predictions.shape[0] * weight_decay)
    
    var += tau
    return mean.unsqueeze(0),var.unsqueeze(0)


def psnr(img1, img2):
    mse = torch.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))



def plot_distributions_from_INN(INN_lambda, initial_block, final_block, ignore_blocks):
    all_outs,_ = INN_lambda()
    blocks_to_use = []
    plt.figure(figsize=(7,10))
    plt.clf()
    # Gather all needed blocks taking into account that the blocks are in inverted order
    for key,val in all_outs.items():
        if all([ib not in key[0].name for ib in ignore_blocks]) and (final_block in key[0].name or len(blocks_to_use)>0):
            blocks_to_use.append([key[0],val])
        if initial_block in key[0].name:
            break
    
    # Iterate backwards and plot histograms
    for nBlock,curr_block in enumerate(blocks_to_use[::-1]):
        curr_stats = torch.std_mean(curr_block[1].cpu().detach())
        ax = plt.subplot(len(blocks_to_use), 1, nBlock+1)
        plt.plot(np.histogram(curr_block[1].cpu().detach().numpy(), bins=1000, density=True)[0], label=curr_block[0].name)
        textstr = '\n'.join((
        r'$\mu=%.2f$' % (curr_stats[1].item(), ),
        r'$\sigma=%.2f$' % (curr_stats[0].item(), )))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.show()

    return plt.gcf()