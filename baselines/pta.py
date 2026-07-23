import numpy as np
from utils.normalize import minmax_norm

def pta_saliency(dolp):

    d = dolp.astype(np.float32)

    # 偏振强度增强
    S = d ** 1.5

    return minmax_norm(S)