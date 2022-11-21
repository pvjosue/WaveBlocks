# This file implements the Richardson Lucy deconvolution Algorithm.
# There are 2 seperate implementations, one using standard convolution, the other using FFT convolution
# When convolving an image and kernel of similar size, the FFT convolution scales better with larger kernel/image sizes

# Erik Riedel & Josef Kamysek
# erik.riedel@tum.de & josef@kamysek.com
# 15/10/2020, Munich, Germany


# Third party library imports
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
import math
import logging
from waveblocks.utils.misc_tools import volume_2_projections
from tqdm import tqdm

# Waveblocks imports
logger = logging.getLogger("Waveblocks")


def RichardsonLucy(
    microscope,
    image,
    n_iterations,
    volume_shape,
    NxyExt=0,
    output_path=None,
    ROISize=None,
    eps=1e-8,
):
    with torch.no_grad():
        nDepths = microscope.psf.shape[1]
        losses = []
        
        # Initial volumes
        volume_pred = torch.ones(
            image.shape[0], nDepths, image.shape[2], image.shape[3], device=image.device
        )

        img_pred = 0
        microscope.camera.fft_paddings_ready = False
        for nIt in tqdm(range(n_iterations)):
            # Current image guess
            # img_pred = microscope.forward(volume_pred, compute_psf=False)

            img_pred, _, _ = microscope.camera.forward(
                volume_pred, microscope.psf.float(),
                full_psf_graph=True
            )

            # img_pred[img_pred < 0] = 0

            # Image usually has even number of pixels so we use different crop method
            cropAmountX = (img_pred.shape[2] - image.shape[2]) // 2
            cropAmountY = (img_pred.shape[3] - image.shape[3]) // 2
            img_pred = img_pred[
                :,
                :,
                cropAmountX : cropAmountX + image.shape[2],
                cropAmountY : cropAmountY + image.shape[3],
            ]   

            # Compute Error
            error = torch.div(image, img_pred + eps)
            error = error.repeat(1, nDepths, 1, 1)

            # MSE
            loss = torch.sum(torch.pow(image - img_pred, 2)) / (
                image.shape[2] * image.shape[3]
            )
            losses.append(loss.item())

            # Compute volume update
            _, _, volume_update = microscope.camera.forward(
                error, microscope.psf.float(), is_backwards=True
            )

            # Update current volume
            volume_pred = torch.mul(volume_pred, volume_update)

            # normalize volume
            # currVol = torch.div((currVol - torch.min(currVol)), (torch.max(currVol) - torch.min(currVol)),)

            # logger.info(
            #     "nIt {}/{} - error: {}".format(nIt + 1, n_iterations, loss.item())
            # )

            if output_path is not None and nIt%10==0:
                plt.clf()
                plt.subplot(2,3,1)
                plt.imshow(img_pred[0, 0, :, :].detach().cpu().numpy())
                plt.title('Img pred')
                plt.subplot(2,3,2)
                plt.imshow(volume_2_projections(volume_pred.permute(0,2,3,1).unsqueeze(1)).squeeze().detach().cpu().numpy())
                plt.title('vol_prediction')
                plt.subplot(2,3,3)
                plt.imshow(error[0, 0, :, :].detach().cpu().numpy())
                plt.title('img_error')
                plt.subplot(2,1,2)
                plt.plot(losses)
                plt.yscale("log", base=2)
                plt.title('losses')
                plt.savefig(f'{output_path}/output_{nIt}.png')

        # Crop volume as all calculations are made with the size of the image
        vol_pred_crop_width = volume_shape[2]
        vol_pred_crop_height = volume_shape[3]
        vol_pred_center_x = math.floor(volume_pred.shape[2] / 2)
        vol_pred_center_y = math.floor(volume_pred.shape[3] / 2)
        volume_pred = volume_pred[
            :,
            :,
            vol_pred_center_x
            - math.floor(vol_pred_crop_width / 2) : vol_pred_center_x
            + math.ceil(vol_pred_crop_width / 2),
            vol_pred_center_y
            - math.floor(vol_pred_crop_height / 2) : vol_pred_center_y
            + math.ceil(vol_pred_crop_height / 2),
        ]

        return volume_pred, img_pred, losses


def RichardsonLucy_conv(
    PSF, image, n_iterations, NxyExt=0, writer=None, ROISize=None, eps=1e-8
):

    logger.warning(
        "ATTENTION: USING PYTORCH CONVOLUTION INSTEAD OF FFT!!! THIS WILL BE VERY SLOW!"
    )
    with torch.no_grad():
        nDepths = PSF.shape[1]
        # Crop volume if ROI specified
        """
        if ROISize is not None:
            newImgStart = torch.tensor(image.shape[2:4]) // 2 - ROISize
            newImgEnd = torch.tensor(image.shape[2:4]) // 2 + ROISize
            newImgEnd -= (newImgEnd - newImgStart) % 2
            image = image[
                :,
                :,
                newImgStart[0]: newImgEnd[0],
                newImgSizeStart[1] : newImgSizeEnd[1],
            ]
        """
        # PSF format (1, z, x, y, Complex)
        # image format (1,1,x,y)
        # Compute proper padding TODO: Make resulting images size 2^n if that improves fft speed
        padImage = torch.zeros(4, dtype=int)
        padPSF = torch.zeros(4, dtype=int)

        if PSF.shape[3] > image.shape[3]:
            padImage[0] = math.floor((PSF.shape[3] - image.shape[3]) / 2)
            padImage[1] = math.ceil((PSF.shape[3] - image.shape[3]) / 2)

        elif image.shape[3] > PSF.shape[3]:
            padPSF[0] = math.floor((image.shape[3] - PSF.shape[3]) / 2)
            padPSF[1] = math.ceil((image.shape[3] - PSF.shape[3]) / 2)

        if PSF.shape[2] > image.shape[2]:
            padImage[2] = math.floor((PSF.shape[2] - image.shape[2]) / 2)
            padImage[3] = math.ceil((PSF.shape[2] - image.shape[2]) / 2)

        elif image.shape[2] > PSF.shape[2]:
            padPSF[2] = math.floor((image.shape[2] - PSF.shape[2]) / 2)
            padPSF[3] = math.ceil((image.shape[2] - PSF.shape[2]) / 2)

        # Pad image and PSF
        image = f.pad(image, list(padImage.numpy()))
        PSF = f.pad(PSF, list(padPSF.numpy()))
        PSF_flip = PSF.float().clone().flip(2).flip(3)

        # Initial volumes
        currVol = torch.ones(
            image.shape[0], nDepths, image.shape[2], image.shape[3], device=image.device
        )

        volUpdate = torch.ones(currVol.shape, device=image.device)
        for nIt in range(n_iterations):
            # Current image guess
            imgEst = torch.zeros(image.shape, device=image.device)
            for i in range(currVol.shape[1]):
                imgEst += torch.nn.functional.conv2d(
                    currVol[:, i, :, :].unsqueeze(0),
                    PSF[:, i, :, :].unsqueeze(0),
                    padding=(PSF.shape[2] // 2),
                )[:, :, 1:, 1:]

            # imgEst[imgEst < 0] = 0

            # Compute Error
            error = torch.div(image, imgEst)
            error = error.repeat(1, nDepths, 1, 1)

            # MSE
            loss = torch.sum(torch.pow(image - imgEst, 2)) / (
                image.shape[2] * image.shape[3]
            )

            # Compute volume update
            for i in range(currVol.shape[1]):
                tmp = torch.nn.functional.conv2d(
                    error[:, i, :, :].unsqueeze(0),
                    PSF_flip[:, i, :, :].unsqueeze(0),
                    padding=(PSF.shape[2] // 2),
                )
                volUpdate[0, i, :, :] = tmp[0, 0, 1:, 1:]
            # volUpdate[volUpdate < 0] = 0

            # normalize update
            # volUpdate = torch.div((volUpdate - torch.min(volUpdate)),(torch.max(volUpdate) - torch.min(volUpdate)),)
            # volUpdate = volUpdate[:,:,1:,1:]
            # Update current volume
            currVol = torch.mul(currVol, volUpdate)

            # logger.info(
            #     "nIt {}/{} - error: {}".format(nIt + 1, n_iterations, loss.item())
            # )

        return currVol, imgEst


#     NxyExt=201;
#     Nxy=size(PSF,1)+NxyExt*2;
#     Nz=size(PSF,3);

#     PSF_A=padarray(PSF,[NxyExt NxyExt 0],0,'both');
#     % PSF_B=padarray(PSF2,[NxyExt NxyExt 0],0,'both');
#     PSF_A=gpuArray(PSF_A);
#     % PSF_B=gpuArray(PSF_B);

#     NxyAdd=round((Nxy/RatioAB-Nxy)/2);
#     NxySub=round(Nxy*(1-RatioAB)/2)+NxyAdd;

#     gpuTmp1=gpuArray(complex(single(zeros(Nxy+NxyAdd*2,Nxy+NxyAdd*2))));
#     gpuTmp2=single(gpuTmp1);
#     gpuTmp3=zeros(Nxy,Nxy,'single','gpuArray');
#     gpuObjReconTmp=zeros(Nxy,Nxy,'single','gpuArray');
#     gpuObjRecon=gpuArray(ones(2*ROISize,2*ROISize,Nz,'single'));

#     ImgExp=gpuArray(padarray(single(ImgMultiView),[NxyExt NxyExt],0,'both'));
#     ImgEst=zeros(Nxy,Nxy,'single','gpuArray');
#     Ratio=zeros(Nxy,Nxy,'single','gpuArray');
#     gpuDevice()
#     for ii=1:ItN
#         display(['iteration: ' num2str(ii)]);
#         tic;
#         ImgEst=ImgEst*0;
#         for jj=1:Nz
#             gpuObjReconTmp(Nxy/2-ROISize+1:Nxy/2+ROISize,Nxy/2-ROISize+1:Nxy/2+ROISize)=gpuObjRecon(:,:,jj);
#             %         gpuTmp1(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd)=fftshift(fft2(gpuObjReconTmp));
#             %         gpuTmp2=abs(ifft2(ifftshift(gpuTmp1)));
#             ImgEst=ImgEst+max(real(ifft2(fft2(ifftshift(single(PSF_A(:,:,jj)))).*fft2(gpuObjReconTmp))),0);
#             %             +max(real(ifft2(fft2(ifftshift(single(PSF_B(:,:,jj)))).*fft2(gpuTmp2(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd)))),0);
#             %         ImgEst=ImgEst+max(real(ifft2(fft2(ifftshift(single(PSF_A(:,:,jj)))).*fft2(gpuObjReconTmp))),0)...
#             %             +max(real(ifft2(fft2(ifftshift(single(PSF_B(:,:,jj)))).*fft2(gpuTmp2(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd)))),0);
#         end
#         gpuTmp4=ImgExp(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)./(ImgEst(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)+eps);
#         Ratio=Ratio*0+single(mean(gpuTmp4(:))*(ImgEst>(max(ImgEst(:))/200)));
#         Ratio(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)=ImgExp(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)./(ImgEst(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)+eps);
#         gpuTmp2=gpuTmp2*0;
#         for jj=1:Nz
#             %           gpuTmp1(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd)=fftshift(fft2(Ratio).*conj(fft2(ifftshift(single(PSF_B(:,:,jj))))));
#             %           gpuTmp2(NxySub+1:end-NxySub,NxySub+1:end-NxySub)=abs(ifft2(ifftshift(gpuTmp1(NxySub+1:end-NxySub,NxySub+1:end-NxySub))));
#             gpuObjReconTmp(Nxy/2-ROISize+1:Nxy/2+ROISize,Nxy/2-ROISize+1:Nxy/2+ROISize)=gpuObjRecon(:,:,jj);
#             gpuTmp3=gpuObjReconTmp.*(max(real(ifft2(fft2(Ratio).*conj(fft2(ifftshift(single(PSF_A(:,:,jj))))))),0));%+gpuTmp2(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd))/2;
#             gpuObjRecon(:,:,jj)=gpuTmp3(Nxy/2-ROISize+1:Nxy/2+ROISize,Nxy/2-ROISize+1:Nxy/2+ROISize);
#         end
#         toc
#         % draw max projection views of restored 3d volume
#         figure(1);
#         subplot(1,3,1);
#         imagesc(squeeze(max(gpuObjRecon,[],3)));
#         title(['iteration ' num2str(ii) ' xy max projection']);
#         xlabel('x');
#         ylabel('y');
#         axis equal;

#         subplot(1,3,2);
#         imagesc(squeeze(max(gpuObjRecon,[],2)));
#         title(['iteration ' num2str(ii) ' yz max projection']);
#         xlabel('z');
#         ylabel('y');
#         axis equal;

#         subplot(1,3,3);
#         imagesc(squeeze(max(gpuObjRecon,[],1)));
#         title(['iteration ' num2str(ii) ' xz max projection']);
#         xlabel('z');
#         ylabel('x');
#         axis equal;
#         drawnow
#     end
#     ObjRecon = gather(gpuObjRecon);
#     save([outFolder,'ObjRecon',num2str(nFile),'.mat'],'ObjRecon');
#     MIPs=[max(ObjRecon,[],3) squeeze(max(ObjRecon,[],2));squeeze(max(ObjRecon,[],1))' zeros(size(ObjRecon,3),size(ObjRecon,3))];
#     figure(2);imagesc(MIPs);axis image;
#     nFile = nFile+1;
# end
