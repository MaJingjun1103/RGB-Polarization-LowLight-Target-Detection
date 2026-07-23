import numpy as np
import cv2
from utils.color import rgb_to_lab, pol_to_lab
from utils.edge import compute_edge_scharr, postprocess_edge
from fusion import fusion

def compute_saliency(lab):

    lab = lab.astype(np.float32)
    mean_vec = np.mean(lab.reshape(-1,3), axis=0)
    diff = lab - mean_vec

    S = np.sqrt(
        0.5 * diff[:,:,0]**2 +
        2.0 * diff[:,:,1]**2 +
        2.0 * diff[:,:,2]**2
    )

    S = S / (np.max(S)+1e-8)
    S = S ** 0.8
    return S


def detect(rgb, dolp, keep_n=1):

    lab = rgb_to_lab(rgb)
    S_rgb = compute_saliency(lab)

    lab_pol = pol_to_lab(dolp)
    S_pol = compute_saliency(lab_pol)

    edge = compute_edge_scharr(rgb)
    E = postprocess_edge(edge)

    FD = fusion(S_rgb, S_pol, E)

    FD = (FD - FD.min()) / (FD.max() - FD.min() + 1e-8)
    FD = cv2.GaussianBlur(FD, (5,5), 0)

    th = FD.mean() + FD.std()
    mask = (FD > th).astype(np.uint8)

    # 连通域筛选
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    final_mask = np.zeros_like(mask)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = np.argsort(areas)[::-1][:keep_n]

        for i in idx:
            final_mask[labels == (i+1)] = 1

    return FD, final_mask