import os
import cv2
import numpy as np

from detect import detect

# baselines
from baselines.lrx import lrx_saliency
from baselines.crd import crd_saliency
from baselines.lrasr import lrasr_saliency
from baselines.lsmad import lsmad_saliency
from baselines.pta import pta_saliency
from baselines.ssiifd import ssiifd_saliency


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def save_gray(path, m):
    m = normalize(m)
    m = (m * 255).astype("uint8")
    cv2.imwrite(path, m)


def save_color(path, m):
    m = normalize(m)
    m = (m * 255).astype("uint8")
    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    cv2.imwrite(path, m)


def main():

    rgb_path = "data/sample/rgb.png"
    dolp_path = "data/sample/dolp.png"

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # ===== 读取 =====
    rgb = cv2.imread(rgb_path)
    dolp = cv2.imread(dolp_path, 0)

    if rgb is None or dolp is None:
        print("❌ 图像路径错误")
        return

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # =========================
    # ⭐ OUR METHOD
    # =========================
    FD, mask = detect(rgb, dolp)

    FD_clean = FD.copy()
    FD_clean[mask == 0] *= 0.1
    FD_clean[mask == 1] = np.clip(FD_clean[mask == 1] * 1.5, 0, 1)

    save_color(os.path.join(save_dir, "ours.png"), FD_clean)

    # =========================
    # ⭐ BASELINES
    # =========================
    save_gray(os.path.join(save_dir, "LRX.png"), lrx_saliency(rgb))
    save_gray(os.path.join(save_dir, "CRD.png"), crd_saliency(rgb))
    save_gray(os.path.join(save_dir, "LRASR.png"), lrasr_saliency(rgb))
    save_gray(os.path.join(save_dir, "LSMAD.png"), lsmad_saliency(rgb))
    save_gray(os.path.join(save_dir, "PTA.png"), pta_saliency(dolp))
    save_gray(os.path.join(save_dir, "SSIIFD.png"), ssiifd_saliency(rgb))

    print("✅ All results saved in /results/")


if __name__ == "__main__":
    main()