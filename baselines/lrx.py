import numpy as np
from utils.normalize import minmax_norm

def lrx_saliency(img):

    gray = np.mean(img, axis=2)

    # SVD分解（低秩背景）
    U, S, Vt = np.linalg.svd(gray, full_matrices=False)

    k = 5  # 保留低秩
    low_rank = (U[:, :k] * S[:k]) @ Vt[:k, :]

    sparse = np.abs(gray - low_rank)

    return minmax_norm(sparse)