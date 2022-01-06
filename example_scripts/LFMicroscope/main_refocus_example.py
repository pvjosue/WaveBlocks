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
import torch
import torch.nn as nn
import tables
import torch.optim as optim
from PIL import Image
import torchvision.transforms.functional as TF
import pathlib

# Waveblocks imports
from waveblocks.microscopes.fourier_lightfield_mla_micro import Microscope
from waveblocks.blocks.microlens_arrays import MLAType
from waveblocks.blocks.optic_config import OpticConfig

# Configuration parameters
lr = 1e2
nEpochs = 2
# Defocus distance in front of the objective
GT_defocus = -20

file_path = pathlib.Path(__file__).parent.absolute()
data_path = file_path.parent.joinpath("data")

# Fetch Device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable plotting
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor="w", edgecolor="k")
plt.ion()
plt.show()

# Load PSF and optics information from file
psfFile = tables.open_file(
    data_path.joinpath("psf_20x_0.45NA.h5"), "r", driver="H5FD_CORE"
)
# Load PSF and arrange it as [1,nDepths,x,y,2], the last dimension stores the complex data
psf_in = (
    torch.tensor(psfFile.root.PSFWaveStack, dtype=torch.float32, requires_grad=True)
    .permute(1, 3, 2, 0)
    .unsqueeze(0)
    .contiguous()
)

# Load image to use as our object in front of the microscope
obj_image = (
    TF.to_tensor(Image.open(data_path.joinpath("GT.tif"))).unsqueeze(0).to(device)
)
obj_image /= obj_image.max()

# Create opticalConfig object with the information from the microscope
opticalConfig = OpticConfig()
# Update optical config from input PSF
opticalConfig.PSF_config.wvl = psfFile.root.wavelenght[0]
opticalConfig.PSF_config.ni = 1

# Load info from PSF file
opticalConfig.sensor_pitch = psfFile.root.sensorRes[0]
opticalConfig.NA = psfFile.root.NA[0]
opticalConfig.M = psfFile.root.M[0]
opticalConfig.ftl = psfFile.root.ftl[0]
opticalConfig.fobj = opticalConfig.ftl / opticalConfig.M
opticalConfig.mla_type = MLAType.periodic

# This variable controls the min posible defocus in image space,
# as the sampling of the Fourier space depends on the propagation distance
opticalConfig.minDefocus = -5 * opticalConfig.M ** 2

# Define names of variables to learn
vars_to_learn = ["wave_prop.propagation_distance"]
learning_rates = [lr]

# Create a Microscope
WBMicro = Microscope(
    optic_config=opticalConfig, members_to_learn=vars_to_learn, psf_in=psf_in
).to(device)
# Microscope for GT
WBMicroGT = Microscope(
    optic_config=opticalConfig, members_to_learn=[], psf_in=psf_in
).to(device)

# Fetch variables to learn from microscope
trainable_vars = WBMicro.get_trainable_variables()
# Create pairs for optimizer, in case of more than one parameter to optimize
trainable_vars_and_lr = len(vars_to_learn) * [0]
for varId, var in enumerate(trainable_vars):
    trainable_vars_and_lr[varId] = {"params": var, "lr": learning_rates[varId]}

# Define loss functions
crit = nn.MSELoss()
optimizer = optim.Adam(trainable_vars_and_lr, lr=lr)

# Generate GT defocused image
WBMicroGT.wave_prop.propagation_distance = nn.Parameter(
    torch.tensor([GT_defocus * opticalConfig.M ** 2]).to(device)
)
Gt_img = WBMicroGT(obj_image)
Gt_img = Gt_img.detach()
# Reinit the propagation distance randomly, to then optimize it
WBMicro.wave_prop.propagation_distance.data = torch.tensor(
    [-50.0 * opticalConfig.M ** 2]
).to(device)

# Arrays for storing statistics
errors = []
predictions = []

# Optimize
for ep in range(nEpochs):
    plt.clf()
    optimizer.zero_grad()
    # Predict defocused image with current defocus
    currImg = WBMicro(obj_image)
    # Compute error
    diff = crit(Gt_img, currImg)
    curr_error = diff.detach().item()
    # Propagate gradients back
    diff.backward()
    # Update wave_prop
    optimizer.step()

    # Store errors
    errors.append(curr_error)

    # Store current prediction
    curr_defocus = WBMicro.wave_prop.propagation_distance.detach() / (
        opticalConfig.M ** 2
    )
    predictions.append(curr_defocus)

    print(str(ep) + " MSE: " + str(curr_error))

    # Display results
    plt.subplot(2, 2, 2)
    plt.imshow(currImg[0, 0, :, :].detach().cpu().numpy())
    plt.title("Current Guess")
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.subplot(2, 2, 1)
    plt.imshow(Gt_img[0, 0, :, :].detach().cpu().numpy())
    plt.title("GT image")
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    ax = plt.subplot(2, 2, 3)
    ax.plot(errors, alpha=0.9, color="b", label="Image")
    # ax.plot(errorsPM, alpha=0.9, color="r" , label='PM')
    ax.hlines(0, 0, len(errors) + 1, linewidth=1, color="k")
    ax.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.title(
        "L2 loss image: "
        + "{:06.3f}".format(curr_error)
        + "\npropagation error: "
        + "{:06.3f}".format((GT_defocus - curr_defocus).item())
    )

    ax = plt.subplot(2, 3, 6)
    ax.plot(predictions, alpha=0.9, color="b", label="Image")
    plt.plot(
        GT_defocus * torch.ones(len(predictions)).detach().numpy(),
        alpha=0.3,
        color="r",
        label="GT defocus",
    )
    ax.legend(loc="lower right")
    plt.xlabel("epoch")
    plt.ylabel("Defocus")
    plt.title("Defocus of image")

    plt.suptitle("Microscope refocusing with WaveBlocks")
    plt.pause(0.1)
    plt.show()
psfFile.close()
