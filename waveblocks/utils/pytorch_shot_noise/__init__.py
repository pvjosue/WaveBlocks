import torch
import numpy as np
import torch.distributions as tdist

def apply_noise(x, max_snr, min_snr):
    curr_max = x.float().max()
    # Update new signal power
    signal_power = (min_snr + (max_snr-min_snr) * torch.rand(1)).item()
    x = signal_power/curr_max * x
    # Add noise
    x = add_camera_noise(x).type_as(x)
    x = curr_max/signal_power * x
    return x

# Function to add camera noise
# http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
def add_camera_noise(input_irrad_gray, qe=0.82, sensitivity=5.88,
                     dark_noise=2.29, bitdepth=16, baseline=0, full_well=30000):
    
    # input_irrad_gray = input_irrad_gray.float()
    # Convert gray to photons
    input_irrad_electrons = input_irrad_gray / sensitivity #* full_well / (2**bitdepth-1)
    input_irrad_electrons[input_irrad_electrons<0] = 0.0
    input_irrad_photons = input_irrad_electrons / qe

    # Add shot noise
    distribution_poiss = torch.distributions.poisson.Poisson(input_irrad_photons.float()+1e-8)
    distribution_norm = torch.distributions.normal.Normal(torch.zeros_like(input_irrad_photons), dark_noise)
    photons = distribution_poiss.sample()
    
    # Convert to electrons
    electrons = qe * photons
    
    # Add dark noise
    electrons_out = electrons + distribution_norm.sample()
    
    # Convert to ADU and add baseline
    max_adu     = np.int(2**bitdepth - 1)
    adu         = (electrons_out * sensitivity).int() # Convert to discrete numbers
    adu += baseline
    adu[adu > max_adu] = max_adu # models pixel saturation
    adu[adu < 0] = 0
    return adu


def add_camera_noise_numpy(input_irrad_gray, qe=0.82, sensitivity=5.88,
                     dark_noise=2.29, bitdepth=16, baseline=100, full_well=30000,
                     rs=np.random.RandomState(seed=42)):
    
    # input_irrad_gray = input_irrad_gray.float()
    # Convert gray to photons
    input_irrad_electrons = input_irrad_gray / sensitivity #* full_well / (2**bitdepth-1)
    input_irrad_electrons[input_irrad_electrons<0] = 0.0
    input_irrad_photons = input_irrad_electrons / qe

    # Add shot noise
    photons = rs.poisson(input_irrad_photons.numpy(), size=input_irrad_photons.shape)
    
    # Convert to electrons
    electrons = qe * photons
    
    # Add dark noise
    electrons_out = electrons + rs.normal(scale=dark_noise, size=electrons.shape) + electrons
    
    # Convert to ADU and add baseline
    max_adu     = np.int(2**bitdepth - 1)
    adu         = (electrons_out * sensitivity).astype(np.int) # Convert to discrete numbers
    adu += baseline
    adu[adu > max_adu] = max_adu # models pixel saturation
    
    return torch.from_numpy(adu)