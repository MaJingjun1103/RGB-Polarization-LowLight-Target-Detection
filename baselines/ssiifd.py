import numpy as np
from utils.normalize import minmax_norm

def ssiifd_saliency(img):

    gray = np.mean(img, axis=2)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)

    S = np.abs(np.fft.ifft2(np.fft.ifftshift(magnitude)))

    return minmax_norm(S.real)