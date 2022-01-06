# Third party libraries imports
import matplotlib.pyplot as plt
import torch

def debug_psf(psf, z):
    plt.imshow(torch.real(psf[0, z, :, :]).detach().cpu().numpy())
    plt.show()


def debug_mla(mla):
    plt.gca().set_aspect('auto')
    plt.imshow(torch.real(mla).detach().numpy())
    plt.show()
