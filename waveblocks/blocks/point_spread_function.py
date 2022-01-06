# Third party libraries imports
import numpy as np
import torch
import scipy
import scipy.interpolate
import scipy.special

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock
import waveblocks.utils.complex_operations as co
import waveblocks.blocks as ob

class PSF(OpticBlock):
    """Class that coputes the PSF of a point source based on the Fast Gibson Lanni computation"""

    # https://github.com/MicroscPSF/MicroscPSF-Py
    def __init__(self, optic_config, members_to_learn):
        super(PSF, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )

        if not hasattr(optic_config, "PSF_config"):
            optic_config.PSF_config = self.get_default_PSF_config()

        # Emission wavelength
        self.wvl = optic_config.PSF_config.wvl
        # Objective magnification
        self.M = optic_config.PSF_config.M
        # Objective Numerical Aperture
        self.NA = optic_config.PSF_config.NA
        # Tube-lens focal length
        self.Ftl = optic_config.PSF_config.Ftl
        # Specimen refractive index (RI)
        self.ns = optic_config.PSF_config.ns
        # Coverlip RI design value
        self.ng0 = optic_config.PSF_config.ng0
        # Coverlip RI experimental value
        self.ng = optic_config.PSF_config.ng
        # Immersion medium RI design value
        self.ni0 = optic_config.PSF_config.ni0
        # Immersion medium RI experimental value
        self.ni = optic_config.PSF_config.ni
        # Working distance (immersion medium thickness) design value
        self.ti0 = optic_config.PSF_config.ti0
        # Coverslip thickness (immersion medium thickness) design value
        self.tg0 = optic_config.PSF_config.tg0
        # Coverslip thickness (immersion medium thickness) experimental value
        self.tg = optic_config.PSF_config.tg
        # Offset of focal plane to coverslip, negative is closer to objective
        self.zv = optic_config.PSF_config.zv
        # Internal constants.
        self.num_basis = (
            5000  # Number of rescaled Bessels that approximate the phase function.
        )
        self.rho_samples = 1000  # Number of pupil sample along the radial direction.

    def forward(self, xy_step, xy_lateral_size, z_particle_range):
        assert (
            xy_lateral_size % 2 != 0
        ), "PSF odd size not optimal! A center pixel is needed for the best resolution"
        # psf = self.gLXYZParticleScan(xy_step, xy_lateral_size, z_particle_range, normalize=False, zv=self.zv)
        psf = self.gLXYZFocalScan(xy_step, xy_lateral_size, z_particle_range).astype(np.complex64)
        # Convert numpy psf to torch PSF
        psfOut = torch.from_numpy(psf)
        #  = torch.cat(
        #     (
        #         torch.from_numpy(psf.real).unsqueeze(3),
        #         torch.from_numpy(psf.imag).unsqueeze(3),
        #     ),
        #     3,
        # )
        return psf, psfOut.unsqueeze(0)

    def calcRv(self, dxy, xy_size, sampling=2):
        """
        Calculate rv vector, this is 2x up-sampled.
        """
        rv_max = np.sqrt(0.5 * xy_size * xy_size) + 1
        return np.arange(0, rv_max * dxy, dxy / sampling)

    def gLZRParticleScan(self, rv, pz, normalize=True, zd=None, zv=0.0):
        """
        Calculate radial G-L at specified radius and z values. This is models the PSF
        you would measure by scanning the particle relative to the microscopes focus.
        mp - The microscope parameters dictionary.
        rv - A numpy array containing the radius values.
        pz - A numpy array containing the particle z position above the coverslip (positive values only)
                in microns.
        normalize - Normalize the PSF to unit height.
        wvl - Light wavelength in microns.
        zd - Actual camera position in microns. If not specified the microscope tube length is used.
        zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
        """
        if zd is None:
            zd = self.Ftl

        pz = np.array([pz])
        zd = np.array([zd])
        zv = np.array([zv])

        return self.gLZRScan(pz, rv, zd, zv, normalize=normalize)

    def configure(self, wvl):
        # Scaling factors for the Fourier-Bessel series expansion
        min_wavelength = 0.436  # microns
        scaling_factor = (
            self.NA * (3 * np.arange(1, self.num_basis + 1) - 2) * min_wavelength / wvl
        )

        # Not sure this is completely correct for the case where the axial
        # location of the flourophore is 0.0.
        #
        max_rho = (
            min([self.NA, self.ng0, self.ng, self.ni0, self.ni, self.ns]) / self.NA
        )

        return [scaling_factor, max_rho]

    def gLZRScan(self, pz, rv, zd, zv, normalize=False):
        """
        Calculate radial G-L at specified radius. This function is primarily designed
        for internal use. Note that only one pz, zd and zv should be a numpy array
        with more than one element. You can simulate scanning the focus, the particle
        or the camera but not 2 or 3 of these values at the same time.
        mp - The microscope parameters dictionary.
        pz - A numpy array containing the particle z position above the coverslip (positive values only).
        rv - A numpy array containing the radius values.
        zd - A numpy array containing the actual camera position in microns.
        zv - A numpy array containing the relative z offset value of the coverslip (negative is closer to the objective).
        normalize - Normalize the PSF to unit height.
        wvl - Light wavelength in microns.
        """
        [scaling_factor, max_rho] = self.configure(self.wvl)
        rho = np.linspace(0.0, max_rho, self.rho_samples)

        a = (
            self.NA * self.Ftl / np.sqrt(self.M * self.M + self.NA * self.NA)
        )  # Aperture radius at the back focal plane.
        k = 2.0 * np.pi / self.wvl

        ti = zv.reshape(-1, 1) + self.ti0
        pz = pz.reshape(-1, 1)
        zd = zd.reshape(-1, 1)

        opdt = self.OPD(rho, ti, pz, zd)

        # Sample the phase
        # phase = numpy.cos(opdt) + 1j * numpy.sin(opdt)
        phase = np.exp(1j * opdt)

        # Define the basis of Bessel functions
        # Shape is (number of basis functions by number of rho samples)
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

        # Compute the approximation to the sampled pupil phase by finding the least squares
        # solution to the complex coefficients of the Fourier-Bessel expansion.
        # Shape of C is (number of basis functions by number of z samples).
        # Note the matrix transposes to get the dimensions correct.
        C, residuals, _, _ = np.linalg.lstsq(J.T, phase.T, rcond=1)

        rv = rv * self.M
        b = k * a * rv.reshape(-1, 1) / zd

        # Convenience functions for J0 and J1 Bessel functions
        J0 = lambda x: scipy.special.jv(0, x)
        J1 = lambda x: scipy.special.jv(1, x)

        # See equation 5 in Li, Xue, and Blu
        denom = scaling_factor * scaling_factor - b * b
        R = (
            scaling_factor * J1(scaling_factor * max_rho) * J0(b * max_rho) * max_rho
            - b * J0(scaling_factor * max_rho) * J1(b * max_rho) * max_rho
        )
        R /= denom

        # The transpose places the axial direction along the first dimension of the array, i.e. rows
        # This is only for convenience.
        # PSF_rz = (numpy.abs(R.dot(C))**2).T
        PSF_rz = R.dot(C).T

        # Normalize to the maximum value
        if normalize:
            PSF_rz /= np.max(PSF_rz)

        return PSF_rz

    def psfRZToPSFXYZ(self, dxy, xy_size, rv, PSF_rz):
        """
        Use interpolation to create a 3D XYZ PSF from a 2D ZR PSF.
        """
        # Create XY grid of radius values.
        c_xy = float(xy_size) * 0.5
        xy = np.mgrid[0:xy_size, 0:xy_size] + 0.5
        r_pixel = dxy * np.sqrt(
            (xy[1] - c_xy) * (xy[1] - c_xy) + (xy[0] - c_xy) * (xy[0] - c_xy)
        )

        # Create XYZ PSF by interpolation.
        PSF_xyz = np.zeros((PSF_rz.shape[0], xy_size, xy_size), dtype=complex)
        for i in range(PSF_rz.shape[0]):
            psf_rz_interp_r = scipy.interpolate.interp1d(rv, PSF_rz[i, :].real)
            psf_rz_interp_r = psf_rz_interp_r(r_pixel.ravel()).reshape(xy_size, xy_size)
            psf_rz_interp_i = scipy.interpolate.interp1d(rv, PSF_rz[i, :].imag)
            psf_rz_interp_i = psf_rz_interp_i(r_pixel.ravel()).reshape(xy_size, xy_size)

            PSF_xyz[i, :, :] = psf_rz_interp_r + 1j * psf_rz_interp_i

        return PSF_xyz

    def gLXYZParticleScan(self, dxy, xy_size, pz, normalize=True, zd=None, zv=0.0):
        """
        Calculate 3D G-L PSF. This is models the PSF you would measure by scanning a particle
        through the microscopes focus.
        This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
        is the zeroth dimension of the PSF.
        mp - The microscope parameters dictionary.
        dxy - Step size in the XY plane.
        xy_size - Number of pixels in X/Y.
        pz - A numpy array containing the particle z position above the coverslip (positive values only)
                in microns.
        normalize - Normalize the PSF to unit height.
        wvl - Light wavelength in microns.
        zd - Actual camera position in microns. If not specified the microscope tube length is used.
        zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
        """
        # Calculate rv vector, this is 2x up-sampled.
        rv = self.calcRv(dxy, xy_size)

        # Calculate radial/Z PSF.
        PSF_rz = self.gLZRParticleScan(rv, pz, normalize=normalize, zd=zd, zv=zv)

        # Create XYZ PSF by interpolation.
        return self.psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz)

    def gLXYZFocalScan(self, dxy, xy_size, zv, normalize=True, pz=0.0, zd=None):
        """
        Calculate 3D G-L PSF. This is models the PSF you would measure by scanning the microscopes
        focus.
        This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
        is the zeroth dimension of the PSF.
        mp - The microscope parameters dictionary.
        dxy - Step size in the XY plane.
        xy_size - Number of pixels in X/Y.
        zv - A numpy array containing the (relative) z offset values of the coverslip (negative is closer to the objective).
        normalize - Normalize the PSF to unit height.
        pz - Particle z position above the coverslip (positive values only).
        wvl - Light wavelength in microns.
        zd - Actual camera position in microns. If not specified the microscope tube length is used.
        """
        # Calculate rv vector, this is 2x up-sampled.
        rv = self.calcRv(dxy, xy_size)

        # Calculate radial/Z PSF.
        PSF_rz = self.gLZRFocalScan(rv, zv, normalize=normalize, pz=pz, zd=zd)

        # Create XYZ PSF by interpolation.
        return self.psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz)

    def gLZRFocalScan(self, rv, zv, normalize=True, pz=0.0, zd=None):
        """
        Calculate radial G-L at specified radius and z values. This is models the PSF
        you would measure by scanning the microscopes focus.
        mp - The microscope parameters dictionary.
        rv - A numpy array containing the radius values.
        zv - A numpy array containing the (relative) z offset values of the coverslip (negative is
                closer to the objective) in microns.
        normalize - Normalize the PSF to unit height.
        pz - Particle z position above the coverslip (positive values only).
        wvl - Light wavelength in microns.
        zd - Actual camera position in microns. If not specified the microscope tube length is used.
        """
        if zd is None:
            zd = self.Ftl

        pz = np.array([pz])
        zd = np.array([zd])

        return self.gLZRScan(pz, rv, zd, zv, normalize=normalize)

    def OPD(self, rho, ti, pz, zd):
        """
        Calculate phase aberration term.
        mp - The microscope parameters dictionary.
        rho - Rho term.
        ti - Coverslip z offset in microns.
        pz - Particle z position above the coverslip in microns.
        wvl - Light wavelength in microns.
        zd - Actual camera position in microns.
        """
        NA = self.NA
        ns = self.ns
        ng0 = self.ng0
        ng = self.ng
        ni0 = self.ni0
        ni = self.ni
        ti0 = self.ti0
        tg = self.tg
        tg0 = self.tg0
        Ftl = self.Ftl

        a = NA * Ftl / self.M  # Aperture radius at the back focal plane.
        k = 2.0 * np.pi / self.wvl  # Wave number of emitted light.

        OPDs = pz * np.sqrt(ns * ns - NA * NA * rho * rho)  # OPD in the sample.
        OPDi = ti * np.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * np.sqrt(
            ni0 * ni0 - NA * NA * rho * rho
        )  # OPD in the immersion medium.
        OPDg = tg * np.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * np.sqrt(
            ng0 * ng0 - NA * NA * rho * rho
        )  # OPD in the coverslip.
        OPDt = (
            a * a * (Ftl - zd) * rho * rho / (2.0 * Ftl * zd)
        )  # OPD in camera position.

        return k * (OPDs + OPDi + OPDg + OPDt)


class PSFScatter(PSF):
    def __init__(self, optic_config, depths, psf_size, members_to_learn=[]):
        super(PSFScatter, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )

        if hasattr(optic_config, 'scattering'):
            self.sigma_x = optic_config.scattering.sigma_x              # this determines g. to obtain the relation between g and sigma_x, use BPM_Estimating_g
            self.seed_density = optic_config.scattering.seed_density    # phase seed density, 
            self.ls = optic_config.scattering.ls                        # scattering mean free path
        else:
            self.sigma_x = 0.6*optic_config.PSF_config.wvl             # this determines g. to obtain the relation between g and sigma_x, use BPM_Estimating_g
            self.seed_density = optic_config.PSF_config.wvl/4           # phase seed density, 
            self.ls = 50000                                               # scattering mean free path
        
        self.psf_size = psf_size
        self.depth_step = abs(depths[0]-depths[1])
        self.waveProp = ob.WavePropagation(
            optic_config=optic_config,
            members_to_learn=[],
            sampling_rate=optic_config.sensor_pitch,
            shortest_propagation_distance=-self.depth_step*optic_config.PSF_config.M**2,
            field_length=psf_size,
            allow_max_sampling=True
        )
    
    def forward(self, xy_step, xy_lateral_size, z_particle_range):
        _,clean_psf = super().forward(self.optic_config.sensor_pitch/self.optic_config.PSF_config.M, self.psf_size, z_particle_range[0])
        wave_prop_psf = clean_psf.clone().repeat(1,len(z_particle_range),1,1)
        for i,d in enumerate(range(1,len(z_particle_range))):
            curr_depth_step = torch.tensor(abs(z_particle_range[0]-z_particle_range[1]))
            max_ph = np.sqrt(curr_depth_step/self.ls/10)*np.pi 
            curr_pm = ob.compute_random_phase_screen(self.sigma_x, self.seed_density, max_ph, self.optic_config.sensor_pitch, self.psf_size, self.optic_config.PSF_config.wvl, self.optic_config.k)

            self.waveProp.propagation_distance.data = -curr_depth_step*self.optic_config.PSF_config.M**2
            curr_psf = wave_prop_psf[:,d-1,...].unsqueeze(1).clone()
            new_psf = self.waveProp(curr_psf * curr_pm)
            wave_prop_psf[:,d,...] =  new_psf
        return 0,wave_prop_psf

def F(x):
    return co.batch_fftshift2d(torch.fft.fft2(co.batch_ifftshift2d(x)))
def Ft(x):
    return co.batch_fftshift2d(torch.fft.ifft2(co.batch_ifftshift2d(x)))

def rF(x):
    return co.batch_fftshift2d(torch.fft.rfft2(co.batch_fftshift2d_real(x)))

def compute_random_phase_screen(sigma_x, seed_density, sigma_p, dx_pixel, N, wvl, k):

    u = np.sort(
        np.concatenate(
            (
                np.arange(start=0, stop=-N/2, step=-1, dtype="float"),
                np.arange(
                    start=1, stop=N/2, step=1, dtype="float"
                ),
            )
        )
        ) * dx_pixel
    xx, yy = np.meshgrid(u, u)
    Nsame = round(seed_density/dx_pixel)

    ph_mask = np.random.normal(0, sigma_p, [len(u), len(u)])
    if Nsame>1:
        for ii in range(1,N-Nsame, Nsame):
            for jj in range(1,N-Nsame, Nsame):
                ph_mask[ii : ii+Nsame, jj: jj+Nsame] = ph_mask(ii,jj)
    
    ph_kernel = np.exp(-(xx**2 + yy**2)/2/sigma_x**2) / (2*np.pi * sigma_x**2)
    ph_kernel /= ph_kernel.sum()

    # Compute cutoff of evanecence wave
    Xsize = N*dx_pixel
    du = 1/Xsize
    umax = 1/(2*dx_pixel)
    u = np.arange(start=-umax, stop=umax-du, step=du) 
    U, V = np.meshgrid(u, u)

    k2 = np.pi * wvl * (U**2 + V**2)
    eva = (k2<np.pi/wvl).astype(float)

    ph_mean1 = ph_mask.mean()
    ph_std1 = ph_mask.std()

    ph_kernel = torch.from_numpy(ph_kernel).unsqueeze(0).unsqueeze(0)
    ph_mask = torch.from_numpy(ph_mask).unsqueeze(0).unsqueeze(0)
    eva = torch.from_numpy(eva).unsqueeze(0).unsqueeze(0)

    # cut off evanecent wave in ph_mask
    ph_mask = torch.real(Ft((F(ph_kernel) * F(ph_mask)) * eva))#[:,:,:,:int(np.ceil(len(u)/2))]))

    ph_mean2 = ph_mask.mean()
    ph_std2 = ph_mask.std()

    ph_mask = (ph_mask-ph_mean2) * ph_std1/ph_std2 + ph_mean1
    
    ph_mask = torch.exp(1j * ph_mask) 
    # co.exp_complex(
    #     torch.cat((
    #         torch.zeros_like(ph_mask).unsqueeze(-1),
    #         ph_mask.unsqueeze(-1)
    #     ), 4)
    # )
    return ph_mask

def compute_OTF(psf, z_pad_amount=0):
    psf = torch.nn.functional.pad(psf,(0,0,0,0,z_pad_amount,z_pad_amount))
    psf_numpy = psf[0,...].cpu().detach().numpy()
    # OTF_torch = torch.zeros_like(psf, dtype=torch.complex64)# torch.fft.rfft2(psf[:,0,...].unsqueeze(1)).repeat(1,psf.shape[1],1,1)
    # for k in range(psf.shape[1]):
    #     curr_MTF = torch.fft.rfft2(psf[:,k,...].unsqueeze(1))
    #     curr_MTF = torch.cat((curr_MTF, torch.flip(curr_MTF[...,:-2],(2,3))), 3)
    #     OTF_torch[:,k,...] =  curr_MTF# np.fft.rfftn(psf_numpy,axes=[1,2,0])
    
    OTF_torch = torch.from_numpy(np.fft.fftn(psf_numpy,axes=[0,1,2])).unsqueeze(0)
    return OTF_torch

