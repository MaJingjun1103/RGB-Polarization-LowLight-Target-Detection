import cv2
import numpy as np
from utils.normalize import minmax_norm

def lsmad_saliency(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (15,15), 0)

    S = np.abs(gray - blur)

    return minmax_norm(S)