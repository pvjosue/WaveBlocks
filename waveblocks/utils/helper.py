# Python imports
import subprocess

# Third party libraries imports
import numpy as np


def get_free_gpu():
    """
    Helper function to select a free gpu
    """
    
    _, output = subprocess.getstatusoutput(
        "nvidia-smi -q -d Memory |grep -A4 GPU|grep Free"
    )
    memory_available = [int(s) for s in output.split() if s.isdigit()]
    return np.argmax(memory_available)
