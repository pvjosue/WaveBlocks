import torch
import time
import numpy as np


nTries = 10

# Torch CPU 4.331078767776489 seconds
x = torch.rand(4,120,600,600)#.cuda()
p = torch.distributions.poisson.Poisson(x)

times = []
for n in range(nTries):
    torch.cuda.synchronize()
    t0 = time.time()
    ytorch = p.sample()
    torch.cuda.synchronize()
    t1 = time.time()
    times.append(t1-t0)
print(min(times))


# Torch GPU 0.004835605621337891
x = torch.rand(4,120,600,600).cuda()
p = torch.distributions.poisson.Poisson(x)

times = []
for n in range(nTries):
    torch.cuda.synchronize()
    t0 = time.time()
    ytorch = p.sample()
    torch.cuda.synchronize()
    t1 = time.time()
    times.append(t1-t0)
print(min(times))

# Numpy 
x = torch.rand(4,120,600,600)
rs = np.random.RandomState(seed=42)

times = []
for n in range(nTries):
    torch.cuda.synchronize()
    t0 = time.time()
    ytorch = rs.poisson(x.numpy(), size=x.shape)
    torch.cuda.synchronize()
    t1 = time.time()
    times.append(t1-t0)
print(min(times))