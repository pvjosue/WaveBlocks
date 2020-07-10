import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class Microscope(ob.OpticBlock):
    def __init__(self, psfIn, vars_to_learn, optical_config):
        super(Microscope, self).__init__(optical_config, vars_to_learn)

        # Added normalization of PSF
        self.psf = nn.Parameter(psfIn/psfIn.sum(), requires_grad=True)
        
        self.sampling_rate = optical_config.sensor_pitch
        field_length = psfIn.shape[2]

        self.useMLA = optical_config.useMLA
        self.useRelays = optical_config.useRelays
        
        if self.useRelays:
            # Relays
            self.lens1 = ob.Lens(self.optic_config, [], optical_config.relay_focal_lenght, \
                self.sampling_rate, optical_config.relay_apperture_width, field_length, \
                    optical_config.relay_focal_lenght, optical_config.relay_focal_lenght)
            self.lens2 = ob.Lens(self.optic_config, [], optical_config.relay_focal_lenght, \
                self.sampling_rate, optical_config.relay_apperture_width, field_length, \
                    optical_config.relay_focal_lenght, optical_config.relay_focal_lenght)
            
            # Setup Spatial Light Modulator

            # Compute pixel size at fourier plane of first lens
            Fs = 1.0/self.sampling_rate
            cyclesPerum = Fs / field_length
            # Incoherent resolution limit
            resolutionLimit = optical_config.PSF_config.wvl/(optical_config.PSF_config.NA*optical_config.PSF_config.M)
            nPixInFourier = resolutionLimit / cyclesPerum
            # Diameter of the objective back focal plane, which acts as the entrance pupil for our system
            dObj = optical_config.PSF_config.fobj * optical_config.PSF_config.NA
            # Sampling size on the fourier domain
            self.fourierMetricSampling = dObj / nPixInFourier

            # Phase mask
            pm_sampling_rate = optical_config.pm_sampling
            pm_shape = optical_config.pm_shape
            pm_image = None
            if hasattr(optical_config,'pm_image'):
                pm_image = optical_config.pm_image
            if pm_sampling_rate is None:
                pm_sampling_rate = self.fourierMetricSampling
            else:
                self.fourierMetricSampling = pm_sampling_rate
            
            pm_correction_img = None
            if hasattr(optical_config,'pm_zero'):
                pm_correction_img = optical_config.pm_zero

            self.phaseMask = ob.DiffractiveElement(self.optic_config, [], pm_sampling_rate,\
                pm_shape, pm_image, self.optic_config.max_phase_shift, 45, [0,0], pm_correction_img)

        if self.useMLA:
            # MLA
            self.MLA = ob.MicroLensArray(self.optic_config, [], optical_config.fm, self.sampling_rate, self.psf.shape[2:4], optical_config.Nnum, optical_config.Nnum, None, 0)
            # propagation
            self.mla2sensor = ob.WavePropagation(self.optic_config, [], self.sampling_rate, optical_config.mla2sensor, field_length)
        
        # Camera
        self.camera = ob.Camera(self.optic_config, [], self.sampling_rate)

    def forward(self, realObject):

        # propagate a certain distance
        psf = self.psf

        if self.useRelays:
            # First lens of relay
            psf = self.lens1(psf)
            # interact with phase mask at fourier plane
            psf = self.phaseMask(psf, self.fourierMetricSampling)
            # Second lens of relay
            psf = self.lens2(psf)
        
        if self.useMLA:
            # MLA
            psf5D = self.MLA(psf, self.sampling_rate)
            # propagate from MLA to sensor
            psfAtSensor = self.mla2sensor(psf5D)
            # compute final PSF and convolve with object
            convolvedObj,psf = self.camera(realObject, psfAtSensor, self.MLA)
        else:
            convolvedObj,psf = self.camera(realObject, psf)
        return convolvedObj
