# Third party libraries imports
import matplotlib.pyplot as plt
import torch

def debug_psf(psf, z):
    """
    Plots a specified z direction of a psf
    """
    
    try:
        plt.imshow(torch.real(psf[0, z, :, :]).detach().cpu().numpy())
        plt.show()
    except Exception as inst:
        print(inst)



def debug_mla(mla):
    """
    Plots a microlens array
    """
    
    try:
        plt.imshow(torch.real(mla).detach().numpy())
        plt.show()
    except Exception as inst:
        print(inst)
