# Example script of WaveBlocks framework.
# This script uses a Fourier Light Field (FLF) microscope with a Microlens Array (MLA)
# The aim of this experiment is to look at the functionality of the FLF microscope with a MLA
# Therefore we forward project a volume of a fish, generating the GT_LF_image.

# Erik Riedel & Josef Kamysek
# erik.riedel@tum.de & josef@kamysek.com
# 15/10/2020, Munich, Germany

from waveblocks.microscopes.fourier_lightfield_mla_micro import Microscope
import waveblocks as ob
import matplotlib.pyplot as plt
import numpy as np
import tables
import torch
import torch.nn.functional as F
import waveblocks
import waveblocks.reconstruction.deconvolution.richardson_lucy as RL
from waveblocks.utils.misc_tools import volume_2_projections
import pathlib
import logging

waveblocks.set_logging(debug_mla=True, debug_microscope=True, debug_richardson_lucy=False, debug_optimizer=True)
logger = logging.getLogger("Waveblocks")


torch.set_num_threads(8)


# Optical Parameters
depth_range = [-500,500]#[-500, 500]
depth_step = 25
wvlgs = [0.65, 1.4]
M = 6
NA = [0.5, 0.5]#2
# M = 4 

depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
nDepths = len(depths)

nLabels = 5
depths_ticks_labels = [str(d) for d in depths]
depths_ticks_labels = depths_ticks_labels[::int(nDepths/nLabels)]
depths_ticks_positions = np.linspace(0, 500, len(depths_ticks_labels)).astype(int)

# Change size of volume you are looking at
vol_xy_size = int(151)#* M/4)
vol_xy_size = vol_xy_size + (1-vol_xy_size%2)
n_iterations = 500

# Configuration parameters
maxVolume = 1
nEpochs = 1
file_path = pathlib.Path(__file__).parent.absolute()
output_path = file_path.parent.joinpath("runs/LF_PM/")
data_path = file_path.parent.joinpath("data")
prefix = "_basic_example_"
# Fetch Device to use

device = torch.device(
    "cuda:" + str(ob.get_free_gpu()) if torch.cuda.is_available() else "cpu"
)

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
GT_volume /= GT_volume.max()
GT_volume *= maxVolume
GT_volume[GT_volume<0.15] = 0

GT_volume = F.pad(GT_volume,4*(20,))
# Create opticalConfig object with the information from the microscope
opticalConfig = ob.OpticConfig()

opticalConfig.scattering = ob.scattering()

# Update optical config from input PSF
mla_shape = [1023, 1023]
# mla_shape = [2047, 2047]
# Lateral size of PSF in pixels
psf_size = mla_shape[0]
# Microscope numerical aperture
opticalConfig.PSF_config.NA = NA[0]
# Microscope magnification
opticalConfig.PSF_config.M = M
# Microscope tube-lens focal length
opticalConfig.PSF_config.Ftl = 165000
# Objective focal length = Ftl/M
opticalConfig.PSF_config.fobj = (
    opticalConfig.PSF_config.Ftl / opticalConfig.PSF_config.M
)
# Immersion refractive index
opticalConfig.PSF_config.ni = 1
# Sample refractive index
# opticalConfig.PSF_config.ns = 1.5

# Camera
opticalConfig.sensor_pitch = 10

# MLA
opticalConfig.use_mla = True
opticalConfig.mla_type = ob.MLAType.coordinate
# Distance between micro lenses centers
opticalConfig.MLAPitch = 90

# Number of pixels behind a single lens
opticalConfig.Nnum = 2 * [opticalConfig.MLAPitch // opticalConfig.sensor_pitch]
opticalConfig.Nnum = [int(n + (1 if (n % 2 == 0) else 0)) for n in opticalConfig.Nnum]

# Distance between the mla and the sensor
opticalConfig.mla2sensor = 35400/10

# MLA focal length
opticalConfig.fm = opticalConfig.mla2sensor

# Define phase_mask to initialize
opticalConfig.use_relay = True
opticalConfig.relay_focal_length = 125000
opticalConfig.relay_separation = opticalConfig.relay_focal_length * 2
opticalConfig.relay_aperture_width = 38100

# Enable fourier conv
opticalConfig.use_fft_conv = True

sf = 2.2
ofX = -1278/sf + mla_shape[0]//2
ofY = -1067/sf + mla_shape[1]//2
mla_coordinates = [
    (ofX+387/sf, ofY+1052/sf),
    (ofX+492/sf, ofY+1334/sf),
    (ofX+501/sf, ofY+766/sf),
    (ofX+568/sf, ofY+1636/sf),
    (ofX+580/sf, ofY+469/sf),
    (ofX+858/sf, ofY+1209/sf),
    (ofX+862/sf, ofY+910/sf),
    (ofX+867/sf, ofY+1858/sf),
    (ofX+885/sf, ofY+252/sf),
    (ofX+917/sf, ofY+1504/sf),
    (ofX+923/sf, ofY+611/sf),
    (ofX+1271/sf, ofY+1977/sf),
    (ofX+1278/sf, ofY+1067/sf),
    (ofX+1280/sf, ofY+1662/sf),
    (ofX+1281/sf, ofY+1362/sf),
    (ofX+1291/sf, ofY+152/sf),
    (ofX+1292/sf, ofY+465/sf),
    (ofX+1291/sf, ofY+768/sf),
    (ofX+1637/sf, ofY+1519/sf),
    (ofX+1655/sf, ofY+620/sf),
    (ofX+1681/sf, ofY+1868/sf),
    (ofX+1702/sf, ofY+266/sf),
    (ofX+1705/sf, ofY+1219/sf),
    (ofX+1710/sf, ofY+920/sf),
    (ofX+1981/sf, ofY+1655/sf),
    (ofX+1999/sf, ofY+493/sf),
    (ofX+2066/sf, ofY+1351/sf),
    (ofX+2067/sf, ofY+794/sf),
    (ofX+2184/sf, ofY+1078/sf)
]

def interpolate_side(img,size=[500,500]):
    if size is None:
        size = img.shape[-2:]
    return F.interpolate(img.permute(0,2,1,3),size).max(0)[0].max(0)[0]

def interpolate_slide(img,size=[500,500]):
    if size is None:
        size = img.shape[-2:]
    return F.interpolate(img.permute(0,2,1,3),size)

plt.ion()
plt.figure(figsize=(10,10))
plt.show()

for ix,wvl in enumerate(wvlgs):

    # Emission wavelength
    opticalConfig.PSF_config.wvl = wvl
    opticalConfig.PSF_config.NA = NA[ix]

    # Setup scattering 

    opticalConfig.scattering.sigma_x = 0.6*opticalConfig.PSF_config.wvl
    opticalConfig.scattering.seed_density = opticalConfig.PSF_config.wvl/4
    opticalConfig.scattering.ls = 500000

    # Define PSF
    # PSF = ob.PSF(opticalConfig)
    PSF = ob.PSFScatter(opticalConfig, depths, psf_size)
    _, psfIn = PSF.forward(
        opticalConfig.sensor_pitch / opticalConfig.PSF_config.M, psf_size, depths
    )
    absPSF = psfIn.clone().abs()**2

    with torch.no_grad():
        WBMicro = Microscope(
            optic_config=opticalConfig,
            members_to_learn=[],
            psf_in=psfIn,
            mla_coordinates=mla_coordinates,
            mla_shape=mla_shape,
        ).to(device)
        WBMicro.eval()
        # Create observed FLF image
        GT_FLF_img = WBMicro(GT_volume.detach()).detach()
        final_psf = WBMicro.psf.clone().abs()**2

        logger.setLevel(logging.INFO)
        currVol, img_est, losses = RL.RichardsonLucy(
            WBMicro, GT_FLF_img, n_iterations, GT_volume.shape)
        logger.setLevel(logging.DEBUG)

    MTF = torch.log(ob.batch_fftshift3d_real(ob.compute_OTF(absPSF).abs()))
    final_MTF = torch.log(ob.batch_fftshift3d_real(ob.compute_OTF(currVol).abs()))

    final_psf = torch.log(final_psf)
    absPSF = torch.log(absPSF)
    n_plots = 5
    plt.subplot(len(wvlgs),n_plots,ix*n_plots+1)
    plt.imshow(
        interpolate_side(absPSF).detach().cpu().numpy()
    )
    plt.ylabel('z in um')
    plt.xlabel('x')
    plt.yticks(depths_ticks_positions,depths_ticks_labels)
    plt.xticks([],[])
    plt.title('PSF Gi-La, depth: ZX psf. wvl: ' + str(opticalConfig.PSF_config.wvl))

    plt.subplot(len(wvlgs),n_plots,ix*n_plots+2)
    plt.imshow(
        # interpolate_side(MTF).detach().cpu().numpy()
        interpolate_slide(MTF)[0,psf_size//2,:,:].detach().cpu().numpy()
        # absPSF[0,nDepths//2,...].detach().cpu().numpy()
    )
    plt.title('MTF')
    plt.ylabel('fz')
    plt.xlabel('fx')
    plt.yticks([],[])
    plt.xticks([],[])

    plt.subplot(len(wvlgs),n_plots,ix*n_plots+3)
    plt.imshow(
        # interpolate_side(MTF).detach().cpu().numpy()
        # interpolate_side(final_psf).detach().cpu().numpy()
        volume_2_projections(currVol.permute(0,2,3,1).unsqueeze(1)).squeeze().detach().cpu().numpy()
    )
    plt.title('recon')

    plt.subplot(len(wvlgs),n_plots,ix*n_plots+4)
    plt.imshow(
        # interpolate_side(MTF).detach().cpu().numpy()
        interpolate_slide(final_MTF)[0,final_MTF.shape[2]//2,:,:].detach().cpu().numpy()
    )
    plt.title('final_recon_MTF')
    plt.ylabel('fz')
    plt.xlabel('fx')
    plt.yticks([],[])
    plt.xticks([],[])

    plt.subplot(len(wvlgs),n_plots,ix*n_plots+5)
    plt.imshow(
        # interpolate_side(MTF).detach().cpu().numpy()
        GT_FLF_img[0,0,...].detach().cpu().numpy()
    )
    plt.title('FLFMimg')


    plt.pause(0.1)
    plt.draw()

plt.savefig('result_' + str(n_iterations) + 'it__' + str(min(depths)) + '_' + str(max(depths)) + 'd__' + str(opticalConfig.PSF_config.M) + 'X_' + str(opticalConfig.PSF_config.NA) + 'NA__' + str(len(wvlgs)) + 'wvl_'+str(opticalConfig.scattering.ls) + '_scatFD.png')
plt.show(block=True)