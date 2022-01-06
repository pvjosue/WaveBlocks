# Third party libraries imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock
import waveblocks.utils.complex_operations as co


class WavePropagationRS(OpticBlock):
    """Class that finds the correct diffraction approximation (Rayleight-sommerfield, Fresnel or Fraunhofer) for the given distance and field size(apperture), and populates propagation_function"""

    def __init__(
        self,
        optic_config,
        members_to_learn,
        sampling_rate,
        shortest_propagation_distance,
        field_length,
    ):

        super(WavePropagationRS, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )
        self.sampling_rate = nn.Parameter(
            torch.tensor([sampling_rate], dtype=torch.float32, requires_grad=True)
        )
        self.propagation_distance = nn.Parameter(
            torch.tensor(
                [shortest_propagation_distance], dtype=torch.float32, requires_grad=True
            )
        )
        self.field_length = nn.Parameter(
            torch.tensor([field_length], dtype=torch.float32, requires_grad=True)
        )

        self.impulse_response = nn.Parameter(None, requires_grad=True)
        self.h = nn.Parameter(None, requires_grad=True)

        # Field metric size
        L = torch.mul(self.sampling_rate, self.field_length)

        ideal_rate = torch.abs(
            self.optic_config.PSF_config.wvl * self.propagation_distance / L
        )
        # Sampling defines if aliases apear or not, for shorter propagation distances higher sampling is needed
        # here we limit the ammount of sampling posible to 2500, which allow a propagation_distance as small as 400um
        ideal_samples_no = torch.min(
            torch.tensor([2500.0], dtype=torch.float32),
            torch.ceil(torch.div(L, ideal_rate)),
        )
        self.ideal_samples_no = nn.Parameter(
            torch.add(ideal_samples_no, 1 - torch.remainder(ideal_samples_no, 2)),
            requires_grad=True,
        )
        self.rate = nn.Parameter(
            torch.div(L, self.ideal_samples_no), requires_grad=True
        )

        u = np.sort(
            np.concatenate(
                (
                    np.arange(start=0, stop=-L / 2, step=-self.rate, dtype="float"),
                    np.arange(
                        start=self.rate, stop=L / 2, step=self.rate, dtype="float"
                    ),
                )
            )
        )

        X, Y = np.meshgrid(u, u)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        self.XY = nn.Parameter(torch.mul(X, X) + torch.mul(Y, Y), requires_grad=True)
        # compute Rayleight-Sommerfield impulse response, which works for any propagation distance
        # self.compute_impulse_response()

    def forward(self, fieldIn):
        if self.propagation_distance == 0.0:
            return fieldIn
            # self.propagation_distance.data = torch.tensor(0.001, dtype=torch.float32).to(self.propagation_distance.device)

        # check for space variant PSF
        if fieldIn.ndimension() == 6:
            field = fieldIn.view(
                fieldIn.shape[0], -1, fieldIn.shape[4], fieldIn.shape[5]
            )
        else:
            field = fieldIn
        fieldOut = field  # .clone()
        nDepths = field.shape[1]

        # Pad before fft
        # padSize = 4*[self.ideal_samples_no.item()//2)]

        ## Rayleight-Sommerfield
        Z = self.propagation_distance
        # compute coordinates
        rho = torch.sqrt(self.XY + torch.mul(Z, Z))
        # h = z./(1i*lambda.*rho.^2).* exp(1i*k.*rho);
        h = (Z / 
            (1j * self.optic_config.PSF_config.wvl * rho**2) * 
            torch.exp(1j * self.optic_config.k
                    * torch.sign(self.propagation_distance)
                    * rho)
        )

        # Pad impulse response
        # h = torch.cat((F.pad(h.unsqueeze(0).unsqueeze(0)[:,:,:,:,0], padSize, 'reflect').unsqueeze(4),\
        # F.pad(h.unsqueeze(0).unsqueeze(0)[:,:,:,:,1], padSize, 'reflect').unsqueeze(4)),4)
        impulse_response_all_depths = self.rate * self.rate * torch.fft.fft2(h)

        # # Pad impulse response
        # impulse_response_all_depths = torch.cat((F.pad(impulse_response_all_depths[:,:,:,:,0], padSize, 'reflect').unsqueeze(4),\
        # 	F.pad(impulse_response_all_depths[:,:,:,:,1], padSize, 'reflect').unsqueeze(4)),4)

        # iterate depths
        # outAllDepths = field.clone() # maybe consuming memory
        for k in range(nDepths):
            currSlice = field[:, k, :, :].unsqueeze(1)
            # Resample input idealy
            inUpReal = f.interpolate(
                torch.real(currSlice),
                size=(
                    int(self.ideal_samples_no.item()),
                    int(self.ideal_samples_no.item()),
                ),
                mode="bilinear",
                align_corners=False,
            )
            inUpImag = f.interpolate(
                torch.imag(currSlice),
                size=(
                    int(self.ideal_samples_no.item()),
                    int(self.ideal_samples_no.item()),
                ),
                mode="bilinear",
                align_corners=False,
            )

            # pad input before fft
            # inUpReal = F.pad(inUpReal, padSize, 'reflect')
            # inUpImag = F.pad(inUpImag, padSize, 'reflect')

            inUp = inUpReal + 1j*inUpImag

            fft1 = torch.fft.fft2(inUp)

            fft11 = fft1 * impulse_response_all_depths

            out1 = co.batch_ifftshift2d(torch.fft.ifft2(fft11))

            # crop padded after both ffts
            # out1 = out1[:,:,padSize[3]:-padSize[2],padSize[1]:-padSize[0],:]

            outSize = list(field.shape[-2:])
            out2R = f.interpolate(
                torch.real(out1), outSize, mode="bilinear", align_corners=False
            )
            out2I = f.interpolate(
                torch.imag(out1), outSize, mode="bilinear", align_corners=False
            )

            outRI = out2R + 1j*out2I

            fieldOut[:, k, :, :] = outRI

        if fieldIn.ndimension() == 6:
            fieldOut = fieldOut.view(fieldIn.shape)
        return fieldOut

    def __str__(self):
        return (
            "sampling_rate: "
            + str(self.sampling_rate)
            + " , "
            + "propagation_distance: "
            + str(self.propagation_distance)
            + " , "
            + "field_length: "
            + str(self.field_length)
            + " , "
            + "method: "
            + self.method
            + " , "
            + "propagation_function: "
            + self.propagation_function
        )


class WavePropagation(OpticBlock):
    """Class that propagates a wave-front with the Beam Propagation Method. see: https://github.com/BUNPC/Beam-Propagation-Method"""

    def __init__(
        self,
        optic_config,
        members_to_learn,
        sampling_rate,
        shortest_propagation_distance,
        field_length,
        allow_max_sampling=False
    ):

        super(WavePropagation, self).__init__(optic_config, members_to_learn)
        self.sampling_rate = nn.Parameter(
            torch.tensor([sampling_rate], dtype=torch.float32, requires_grad=True)
        )
        self.propagation_distance = nn.Parameter(
            torch.tensor(
                [shortest_propagation_distance], dtype=torch.float32, requires_grad=True
            )
        )
        self.field_length = nn.Parameter(
            torch.tensor([field_length], dtype=torch.float32, requires_grad=True)
        )

        # # Field metric size
        L = torch.mul(self.sampling_rate, self.field_length)

        u = np.sort(
            np.concatenate(
                (
                    np.arange(start=0, stop=-self.field_length/2, step=-1, dtype="float"),
                    np.arange(
                        start=1, stop=self.field_length/2, step=1, dtype="float"
                    ),
                )
            )
        )
        wvl = self.optic_config.PSF_config.wvl
        LX = self.field_length*self.sampling_rate
        u = wvl * u / LX.item()

        uu, vv = np.meshgrid(u, u)
        uu = torch.from_numpy(uu).float()
        vv = torch.from_numpy(vv).float()
        
        freq_map = (uu**2 + vv**2) #* 10000
        self.mask = (freq_map/(wvl**2)<((1/wvl)**2)).float()#.unsqueeze(-1).repeat(1,1,2)


        k2 = (1-uu**2 - vv**2)#.unsqueeze(-1).repeat(1,1,2)
        positive = k2>=0
        k2_r = torch.zeros_like(k2)#.unsqueeze(-1).repeat(1,1,2)
        k2_i = torch.zeros_like(k2)
        k2_r[positive] = torch.sqrt(k2[positive] *  self.mask[positive])
        k2_i[~positive] = torch.sqrt(-k2[~positive] *  self.mask[~positive])
        self.k2 = k2_r + 1j * k2_i# torch.cat((k2_r.unsqueeze(-1),k2_i.unsqueeze(-1)),2)


    def forward(self, fieldIn):
        def F(x):
            return co.batch_fftshift2d(torch.fft.fft2(co.batch_ifftshift2d(x)))
        def Ft(x):
            return co.batch_fftshift2d(torch.fft.ifft2(co.batch_ifftshift2d(x)))
        # H =exp(1i*k*dz(m-1)*sqrt((1-uu.^2-vv.^2).*eva));  


        # check for space variant PSF
        if fieldIn.ndimension() == 6:
            field = fieldIn.view(
                fieldIn.shape[0], -1, fieldIn.shape[4], fieldIn.shape[5]
            )
        else:
            field = fieldIn
        fieldOut = field.clone()
        nDepths = field.shape[1]
        outSize = list(field.shape[-3:-1])


        H = torch.exp( 1j * self.optic_config.k
                    * self.propagation_distance
                    # * torch.sign(self.propagation_distance)
                    * torch.real(self.k2)
                    .to(field.device)
                )
        #self.debug = torch.real(H)
        for k in range(nDepths):
            currSlice = field[:, k, ...].unsqueeze(1)
            # Resample input idealy
            inUpReal = f.interpolate(
                torch.real(currSlice),
                size=self.k2.shape,
                mode="bicubic", 
                align_corners=False,
            )
            inUpImag = f.interpolate(
                torch.imag(currSlice),
                size=self.k2.shape,
                mode="bicubic", 
                align_corners=False,
            )

            inUp = inUpReal + 1j * inUpImag

            out1 = Ft(F(inUp) * H)

            outSize = list(field.shape[-2:])
            out2R = f.interpolate(
                torch.real(out1), outSize, 
                mode="bicubic", 
                align_corners=False
            )
            out2I = f.interpolate(
                torch.imag(out1), outSize, 
                mode="bicubic", 
                align_corners=False
            )

            

            outRI = out2R + 1j * out2I

            fieldOut[:, k, ...] = outRI
        
        if fieldIn.ndimension() == 6:
            fieldOut = fieldOut.view(fieldIn.shape)
        return fieldOut
            
   
    def __str__(self):
        return (
            "sampling_rate: "
            + str(self.sampling_rate)
            + " , "
            + "propagation_distance: "
            + str(self.propagation_distance)
            + " , "
            + "field_length: "
            + str(self.field_length)
            + " , "
            + "method: "
            + self.method
            + " , "
            + "propagation_function: "
            + self.propagation_function
        )