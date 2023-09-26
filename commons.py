import torch
import numpy as np

# convert torch dtype to numpy dtype
def np_dtype(dtype:torch.dtype):
    if dtype == torch.float64:
        return np.float64
    elif dtype == torch.float32:
        return np.float32
    elif dtype == torch.int64:
        return np.int64
    elif dtype == torch.int32:
        return np.int32
    else:
        raise NotImplementedError