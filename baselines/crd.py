import numpy as np
from utils.normalize import minmax_norm

def crd_saliency(img):

    mean_color = np.mean(img.reshape(-1,3), axis=0)

    diff = img - mean_color

    S = np.sqrt(np.sum(diff**2, axis=2))

    return minmax_norm(S)