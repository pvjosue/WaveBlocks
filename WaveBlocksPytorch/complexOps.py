import torch
import numpy as np

# expComplex: computes complex exponential
# taken from: https://www.math.wisc.edu/~angenent/Free-Lecture-Notes/freecomplexnumbers.pdf 
# div and mult from: https://svn.python.org/projects/python/trunk/Demo/classes/Complex.py
def expComplex(x): # xreal and imag stacked in last dim
    # Z=X+i*Y, EXP(Z) = EXP(X)*(COS(Y)+i*SIN(Y)).
    reshaped = x.view(-1,2) # reshape into real and imaginary

    ea = torch.exp(reshaped[:,0])
    
    reshapedR = torch.mul(ea, torch.cos(reshaped[:,1]))
    reshapedI = torch.mul(ea, torch.sin(reshaped[:,1]))
    reshapedRI = torch.cat((reshapedR.unsqueeze(1),reshapedI.unsqueeze(1)), 1)
    return reshapedRI.view(x.shape)
    

def divComplex(x, other):
    otherRI = other.view(-1,2)
    xRI = x.view(-1,2)
    d = torch.add(torch.mul(otherRI[:,0], otherRI[:,0]), torch.mul(otherRI[:,1], otherRI[:,1]))
#     if not d: raise ZeroDivisionError, 'Complex division'

    outR = torch.div(torch.add(torch.mul(xRI[:,0], otherRI[:,0]), torch.mul(xRI[:,1], otherRI[:,1])), d)
    outI = torch.div(torch.sub(torch.mul(xRI[:,1], otherRI[:,0]), torch.mul(xRI[:,0], otherRI[:,1])), d)
    

    out = torch.cat((outR.unsqueeze(1), outI.unsqueeze(1)), 1)
    if x.ndimension() > other.ndimension():
        return out.view(x.shape)
    else:
        return out.view(other.shape)

def mulComplex(x, other):
        otherRI = torch.split(other,1,dim=other.ndimension()-1)
        xRI = torch.split(x,1,dim=x.ndimension()-1)
        
        outR = torch.sub(torch.mul(xRI[0], otherRI[0]), torch.mul(xRI[1], otherRI[1]))
        outI = torch.add(torch.mul(xRI[0], otherRI[1]), torch.mul(xRI[1], otherRI[0]))

        out = torch.cat((outR, outI), outR.ndimension()-1)
        return out
        # def __mul__(self, other):
        # other = ToComplex(other)
        # return Complex(self.re*other.re - self.im*other.im,
        #                self.re*other.im + self.im*other.re)

def absSquareComplex(x):
    # get real and imaginary part
    xRI = x.view(-1,2)
    # abs(z)**2 = x**2+y**2
    # output loses complex last dimension
    return torch.add( torch.mul(xRI[:,0], xRI[:,0]), torch.mul(xRI[:,1], xRI[:,1]) ).view(x.shape[0:-1])

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

# Create circular mask for lenses
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask