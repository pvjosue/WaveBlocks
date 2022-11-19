# Example script of WaveBlocks framework.
# This script creates a defocused image in front of the microscope and predicts the defocusing
# distance, by minimizing the MSE between the image generated with the GT defocus, vs the one
# with the current defocus. For this the file microWithPropagation uses a wave-propagation module
# and a Camera module for rendering.

# Josue Page Vizcaino
# pv.josue@gmail.com
# 02/08/2020, Bern Switzerland

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import tables
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv
from datetime import datetime
import pathlib
import os 
from tqdm import tqdm

# Waveblocks imports
from waveblocks.microscopes.lightfield_micro import Microscope
from waveblocks.blocks.optic_config import OpticConfig
from waveblocks.blocks.microlens_arrays import MLAType
import waveblocks.blocks.point_spread_function as psf

torch.set_num_threads(16)


work_dir = os.getcwd()
save_img_path = f"{work_dir}/outputs/{os.path.basename(__file__)[:-3]}/"
os.makedirs(save_img_path, exist_ok=True)

# Optical Parameters
depth_range = [-50, 50]
depth_step = 5
depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
nDepths = len(depths)
vol_xy_size = 251

# Configuration parameters
lr = 1e-2
lrNet = 5e-4
net_weight = 5e-6
maxVolume = 1
nEpochs = 20
file_path = pathlib.Path(__file__).parent.absolute()
output_path = str(file_path.parent.joinpath("runs/runs_ResBlock_newPSF/")) + "/"
data_path = file_path.parent.joinpath("data")
prefix = "_ResBlock3DeepConv7_"
plot = True
# Fetch Device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable plotting
if plot:
    fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor="w", edgecolor="k")
    plt.ion()
    plt.show()

# # Load PSF and optics information from file
# psfFile = tables.open_file('config_files/psfLF_20_M__-5_5_5_depths__0.01_ths.h5', "r", driver="H5FD_CORE")
# # Load PSF and arrange it as [1,nDepths,x,y,2], the last dimension stores the complex data
# psfIn = torch.tensor(psfFile.root.PSFWaveStack, dtype=torch.float32, requires_grad=True).permute(1,3,2,0).unsqueeze(0).contiguous()

# Load volume to use as our object in front of the microscope
vol_file = tables.open_file(
    data_path.joinpath("fish_phantom_251_251_51.h5"), "r", driver="H5FD_CORE"
)
GT_volume = (
    torch.tensor(vol_file.root.fish_phantom)
    .permute(2, 1, 0)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)
GT_volume = torch.nn.functional.interpolate(
    GT_volume, [vol_xy_size, vol_xy_size, nDepths]
)
GT_volume = GT_volume[:, 0, ...].permute(0, 3, 1, 2).contiguous()
GT_volume = torch.nn.functional.pad(GT_volume, [2, 2, 2, 2])
GT_volume /= GT_volume.max()
GT_volume *= maxVolume

# Create opticalConfig object with the information from the microscope
opticalConfig = OpticConfig()

# Update optical config from input PSF
psf_size = 17 * 5
opticalConfig.PSF_config.NA = 0.45
opticalConfig.PSF_config.M = 20
opticalConfig.PSF_config.Ftl = 165000
opticalConfig.PSF_config.wvl = 0.63
opticalConfig.PSF_config.ni = 1
opticalConfig.PSF_config.fobj = (
    opticalConfig.PSF_config.Ftl / opticalConfig.PSF_config.M
)

# Camera
opticalConfig.sensor_pitch = 6.9
opticalConfig.use_relay = False

# MLA
opticalConfig.use_mla = True
opticalConfig.MLAPitch = 112
opticalConfig.Nnum = 2 * [opticalConfig.MLAPitch // opticalConfig.sensor_pitch]
opticalConfig.Nnum = [int(n + (1 if (n % 2 == 0) else 0)) for n in opticalConfig.Nnum]
opticalConfig.mla2sensor = 2500
opticalConfig.fm = 2500
opticalConfig.mla_type = MLAType.periodic

# Define relay   to initialize
opticalConfig.use_relay = True
opticalConfig.relay_focal_length = 150000
opticalConfig.relay_separation = opticalConfig.relay_focal_length * 2
opticalConfig.relay_aperture_width = 50800

# Define PSF
PSF = psf.PSF(opticalConfig, [])
_, psf_in = PSF.forward(
    opticalConfig.sensor_pitch / opticalConfig.PSF_config.M, psf_size, depths
)

# Create a Microscope
WBMicro = Microscope(optic_config=opticalConfig, members_to_learn=[], psf_in=psf_in).to(
    device
)
# WBMicro.mla2sensor = WBMicro.mla2sensor.to("cuda:1")
WBMicro.eval()

# Load GT LF image
GT_LF_img = WBMicro(GT_volume.detach())
GT_LF_img = GT_LF_img.detach()
GT_LF_img.requires_grad = False
# Initial Volume
curr_volume = torch.autograd.Variable(
    maxVolume * torch.ones(GT_volume.shape, requires_grad=True, device=device),
    requires_grad=True,
)


# Refinement net
# @autocast()
class ResidualBlock(torch.nn.Module):
    def __init__(self, n_depths):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Conv3d(1, 4, kernel_size=(7, 7, 3), padding=(3, 3, 1))
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(4, 1, kernel_size=(7, 7, 3), padding=(3, 3, 1))

    def forward(self, input):
        x = input.permute(0, 2, 3, 1).unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.relu1(x1)
        x3 = self.conv2(x2)
        x4 = x + x3
        return x4[:, 0, :].permute(0, 3, 1, 2)


# refine_net = torch.nn.Conv2d(nDepths, nDepths, kernel_size=5, padding=2, padding_mode='reflect').to(device)

refine_net = ResidualBlock(nDepths).to(device)

params = [
    {"params": refine_net.parameters(), "lr": lrNet},
    {"params": curr_volume, "lr": lr},
]

crit = torch.nn.MSELoss()  # poissonLoss
optimizer = optim.Adam(params, lr=lr)

save_folder = (
    output_path
    + datetime.now().strftime("%Y_%m_%d__%H:%M:%S")
    + "_"
    + str(nDepths)
    + "_nD__"
    + str(lr)
    + "_lr__"
    + str(lrNet)
    + "_lrNet__"
    + str(net_weight)
    + "_reg"
    + prefix
)

# Create summary writer to log stuff
writer = SummaryWriter(log_dir=save_folder)

# Arrays for storing statistics
errors = []
predictions = []

# Optimize
for ep in tqdm(range(nEpochs)):
    plt.clf()
    optimizer.zero_grad()
    WBMicro.zero_grad()
    # with autocast(enabled=False):
    # ReLU as non negativity constraing
    currImg = WBMicro(curr_volume)

    volume_loss = crit(currImg, GT_LF_img)
    if net_weight != 0:
        # Predict defocused image with current defocus
        regularization = refine_net(curr_volume)

        diff = volume_loss + net_weight * torch.abs(regularization).sum()
    else:
        regularization = torch.zeros_like(curr_volume)
        diff = volume_loss

    # Compute error
    curr_error = volume_loss.detach().item() / nDepths
    # Propagate gradients back
    diff.backward()
    # Update wave_prop
    optimizer.step()

    # curr_volume.data -= lr * curr_volume.grad.data
    # curr_volume.grad.data.zero_()
    # WBMicro.zero_grad()

    # # Store errors
    errors.append(curr_error)

    # print(str(ep) + " MSE: " + str(curr_error))

    # writer.add_scalar("error", curr_error, ep)
    # writer.add_scalar("full error", diff.item(), ep)
    # # local_volumes[0,...].unsqueeze(0).sum(3).cpu().data.detach(), normalize=True, scale_each=True)
    # writer.add_image(
    #     "GT_vol",
    #     tv.utils.make_grid(GT_volume.sum(1).cpu(), normalize=True, scale_each=True),
    #     ep,
    # )
    # currProj = GT_volume.sum(2).detach().cpu().unsqueeze(0)
    # currProj = torch.nn.functional.interpolate(
    #     currProj, [currProj.shape[2] * 15, currProj.shape[3]], mode="nearest"
    # )
    # writer.add_image(
    #     "GT_vol_xz", tv.utils.make_grid(currProj, normalize=True, scale_each=True), ep
    # )
    # writer.add_image(
    #     "prediction",
    #     tv.utils.make_grid(
    #         curr_volume.sum(1).detach().cpu(), normalize=True, scale_each=True
    #     ),
    #     ep,
    # )
    # currProj = curr_volume.sum(2).detach().cpu().unsqueeze(0)
    # currProj = torch.nn.functional.interpolate(
    #     currProj, [currProj.shape[2] * 15, currProj.shape[3]], mode="nearest"
    # )
    # writer.add_image(
    #     "prediction_xz",
    #     tv.utils.make_grid(currProj[0], normalize=True, scale_each=True),
    #     ep,
    # )
    # currProj = regularization.sum(2).detach().cpu().unsqueeze(0)
    # currProj = torch.nn.functional.interpolate(
    #     currProj, [currProj.shape[2] * 15, currProj.shape[3]], mode="nearest"
    # )
    # writer.add_image(
    #     "regularization_xz",
    #     tv.utils.make_grid(currProj[0], normalize=True, scale_each=True),
    #     ep,
    # )
    # writer.add_image(
    #     "regularization",
    #     tv.utils.make_grid(
    #         regularization.sum(1).detach().cpu(), normalize=True, scale_each=True
    #     ),
    #     ep,
    # )
    # writer.add_image(
    #     "GT_img",
    #     tv.utils.make_grid(
    #         GT_LF_img.sum(1).detach().cpu(), normalize=True, scale_each=True
    #     ),
    #     ep,
    # )
    # writer.add_image(
    #     "predicted_img",
    #     tv.utils.make_grid(
    #         currImg.sum(1).detach().cpu(), normalize=True, scale_each=True
    #     ),
    #     ep,
    # )

    # if ep % 25 == 0:
    #     torch.save(
    #         {
    #             "epoch": ep,
    #             "model_state_dict": refine_net.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #             "loss": curr_error,
    #         },
    #         save_folder + "/model_" + str(ep),
    #     )
    if plot:
        # Display results
        plt.subplot(3, 2, 2)
        plt.imshow(currImg[0, 0, :, :].detach().cpu().numpy())
        plt.title("Current Guess")
        plt.gray()
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        plt.subplot(3, 2, 1)
        plt.imshow(GT_LF_img[0, 0, :, :].detach().cpu().numpy())
        plt.title("GT image")
        plt.gray()
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        ax = plt.subplot(3, 2, 3)
        ax.plot(errors, alpha=0.9, color="b", label="Image")
        # ax.plot(errorsPM, alpha=0.9, color="r" , label='PM')
        ax.hlines(0, 0, len(errors) + 1, linewidth=1, color="k")
        ax.legend(loc="upper right")
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.title("L2 loss image: " + "{:06.3f}".format(curr_error))

        plt.subplot(3, 2, 4)
        plt.imshow(regularization[0, :, :, :].sum(0).detach().cpu().numpy())
        plt.title("refined")
        plt.gray()
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        plt.subplot(3, 2, 5)
        plt.imshow(GT_volume[0, :, :, :].sum(0).detach().cpu().numpy())
        plt.title("GT volume")
        plt.gray()
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        plt.subplot(3, 2, 6)
        plt.imshow(curr_volume[0, :, :, :].sum(0).detach().cpu().numpy())
        plt.title("current_volume")
        plt.gray()
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        plt.pause(0.1)
        plt.savefig(f'{save_img_path}/output_{ep}.png')
        plt.show()
