import numpy as np
from utils.normalize import minmax_norm

def lrasr_saliency(img):

    gray = np.mean(img, axis=2)

    # 简化RPCA思想
    mean = np.mean(gray)
    sparse = np.abs(gray - mean)

    return minmax_norm(sparse)