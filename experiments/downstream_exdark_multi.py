"""ExDark Downstream — Multi-Model 비교 (Zero-DCE / SCI / EnlightenGAN / IAT / LUNA).

목적
----
``experiments/downstream_exdark.py`` 가 [원본] vs [LUNA] 2 종 비교라면,
본 스크립트는 동일한 ExDark Test split (또는 전체 7,363 장) 에 대해
SOTA Low-Light Enhancement 모델 5 종으로 향상한 결과를 모두 동일한
YOLOv8n(COCO pre-trained, conf=0.25) 으로 평가하여
``mAP@0.5 / mAP@0.5:0.95 / Precision / Recall`` 을 비교한다.

파이프라인
----------
각 이미지 1 장에 대해:
1. PIL 로 원본 RGB 로드 (해상도 보존)
2. 각 모델 m 에 대해:
   * m 의 전처리로 모델 입력 텐서 생성
   * m.forward → enhanced tensor
   * uint8 RGB (H, W, 3) 로 원본 해상도 복원
3. enhanced RGB 를 YOLOv8n 에 입력 → 검출 boxes/classes/confs
4. ExDark annotation (bbGt v3) 파싱 → GT
5. (model 별) ``DetectionAccumulator`` 에 (gt, pred) 누적
모든 이미지 처리 후 각 모델별 ``compute()`` → 비교 표 / CSV 저장.

사용 예
-------
.. code-block:: bash

    python experiments/downstream_exdark_multi.py \\
        --luna_ckpt        checkpoints/ext_lol_v2_real_stage2_best.pth \\
        --zerodce_ckpt     C:/대학교/Projects/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth \\
        --sci_ckpt         C:/대학교/Projects/SCI/CVPR/weights/medium.pt \\
        --iat_ckpt         C:/대학교/Projects/Illumination-Adaptive-Transformer/IAT_enhance/best_Epoch_lol_v1.pth \\
        --enlightengan_ckpt C:/대학교/Projects/EnlightenGAN/checkpoints/enlightening/200_net_G_A.pth \\
        --conf 0.25

옵션 ``--skip_models LUNA,IAT`` 로 일부 모델만 평가하거나, ``--max_samples N``
으로 디버그 / 빠른 검증이 가능하다. 가중치를 지정하지 않은 모델은 평가
대상에서 자동 제외된다 (원본 baseline 은 항상 포함).

가중치 다운로드 방법은 README 참고. 본 파일 하단의 ``MODEL_NOTES`` 도 참조.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- 프로젝트 루트 (CODES/) 를 sys.path 등록 ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Windows 콘솔(cp949) 한글 안전
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

# 기존 LOL 스크립트의 LUNA helper / Accumulator / GT 파서 재사용
from experiments.downstream_detection import (  # type: ignore
    load_luna_generator,
    pil_to_norm_tensor,
    norm_tensor_to_uint8_rgb,
)
from experiments.downstream_exdark import (  # type: ignore
    EXDARK_CLASSES,
    EXDARK_TO_COCO,
    COCO_TO_EXDARK,
    TARGET_COCO_IDS,
    DetectionAccumulator,
    ExDarkSample,
    collect_exdark_samples,
    parse_bbgt_v3,
    _try_tqdm,
)


HRULE = "=" * 132
SUBRULE = "-" * 132


# ===========================================================================
# Import 격리 — 여러 프로젝트가 동일 이름 'model' 을 갖고 있어 발생하는
# sys.modules 캐시 충돌을 방지한다.
# ===========================================================================
@contextlib.contextmanager
def _isolated_repo_import(repo_path: Path, *conflict_names: str):
    """``repo_path`` 를 ``sys.path[0]`` 으로 일시 등록하고 import 후 원래대로 복원한다.

    Zero-DCE / SCI / IAT 가 모두 top-level ``model`` 모듈/패키지를 가지고 있어
    먼저 import 된 프로젝트의 ``model`` 이 ``sys.modules`` 에 캐시되면 이후 다른
    프로젝트의 ``from model import ...`` 가 캐시된 다른 프로젝트의 모듈을 보거나
    (``ImportError: cannot import name 'X'``), 패키지가 아닌데 패키지로 접근해
    (``No module named 'model.IAT_main'; 'model' is not a package``) 실패한다.

    이 컨텍스트는:
      1. ``conflict_names`` 에 해당하는 sys.modules 항목을 잠시 옆으로 빼두고,
      2. ``repo_path`` 를 sys.path 0번에 끼워 넣은 뒤,
      3. yield 종료 시 컨텍스트 안에서 새로 캐시된 동일 이름들을 제거하고
         이전 값을 복원, ``sys.path`` 도 진입 전 상태로 되돌린다.

    Class 객체는 import 시점에 ``__globals__`` (모듈 dict) 참조를 잡고 있으므로
    sys.modules 에서 제거해도 이후 forward / 가중치 로드는 정상 동작한다.
    """
    repo = str(repo_path.resolve())
    saved_path = list(sys.path)
    saved_mods: Dict[str, Any] = {}

    def _matches(n: str) -> bool:
        return any(n == c or n.startswith(c + ".") for c in conflict_names)

    for n in list(sys.modules):
        if _matches(n):
            saved_mods[n] = sys.modules.pop(n)
    sys.path.insert(0, repo)
    try:
        yield
    finally:
        for n in list(sys.modules):
            if _matches(n):
                sys.modules.pop(n, None)
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path


# ===========================================================================
# 모델별 어댑터 — 통일 인터페이스: enhance(pil) -> uint8 RGB (H, W, 3)
# ===========================================================================
class _BaseAdapter:
    """모든 향상 모델은 이 인터페이스를 따른다.

    * ``name`` — 콘솔/CSV 라벨 (e.g. ``"Zero-DCE"``)
    * ``load(ckpt, device)`` — 가중치 로드, 실패 시 RuntimeError
    * ``enhance(pil) -> np.ndarray`` — uint8 RGB, 원본 해상도
    """
    name: str = "Base"

    def __init__(self, device: str) -> None:
        self.device = device
        self.model: Optional[torch.nn.Module] = None

    def load(self, ckpt_path: Path) -> None:
        raise NotImplementedError

    def enhance(self, pil: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def n_params(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())


# ---------------------------------------------------------------------------
# 1) Zero-DCE
# ---------------------------------------------------------------------------
class ZeroDCEAdapter(_BaseAdapter):
    """Zero-Reference Deep Curve Estimation (CVPR 2020).

    * Forward: ``_, enhanced, _ = DCE_net(x)``; x 는 [0,1] RGB.
    * 출력은 [0,1] (보통은 0~1.05 정도까지도 가는데, clip 0..1).
    """
    name = "Zero-DCE"

    def __init__(self, device: str, zerodce_repo: Path) -> None:
        super().__init__(device)
        # Zero-DCE 코드 path 격리 등록 (model.py 가 들어있는 폴더)
        with _isolated_repo_import(zerodce_repo, "model"):
            from model import enhance_net_nopool  # type: ignore
        self._cls = enhance_net_nopool

    def load(self, ckpt_path: Path) -> None:
        net = self._cls().to(self.device)
        sd = torch.load(ckpt_path, map_location=self.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd, strict=True)
        net.eval()
        self.model = net

    @torch.no_grad()
    def enhance(self, pil: Image.Image) -> np.ndarray:
        arr = np.asarray(pil, dtype=np.float32) / 255.0          # (H, W, 3)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        x = x.to(self.device)
        _, enh, _ = self.model(x)
        enh = enh.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (enh * 255.0).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2) SCI (Self-Calibrated Illumination, CVPR 2022)
# ---------------------------------------------------------------------------
class SCIAdapter(_BaseAdapter):
    """SCI (Self-Calibrated Illumination).

    * Forward: ``i, r = Finetunemodel(input)`` — r = input / i (clamp 0..1).
    * Input 은 [0,1] (3, H, W), output r 도 [0,1].
    """
    name = "SCI"

    def __init__(self, device: str, sci_repo: Path) -> None:
        super().__init__(device)
        # SCI 의 model.py 는 같은 폴더의 loss.py / utils.py 를 import 하므로
        # 같은 이름이 다른 프로젝트(Zero-DCE 등)에서 캐시돼 있지 않도록 함께 격리.
        with _isolated_repo_import(sci_repo, "model", "loss", "utils"):
            from model import Finetunemodel  # type: ignore
        self._cls = Finetunemodel

    def load(self, ckpt_path: Path) -> None:
        # Finetunemodel 은 생성자에서 weights 를 받음 (file path).
        # 단 torch.load 가 GPU 텐서로 저장돼 있을 수 있으니 device 안전 가공.
        net = self._cls(str(ckpt_path)).to(self.device)
        net.eval()
        self.model = net

    @torch.no_grad()
    def enhance(self, pil: Image.Image) -> np.ndarray:
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        _i, r = self.model(x)
        r = r.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (r * 255.0).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3) EnlightenGAN (BMVC 2021)
# ---------------------------------------------------------------------------
class EnlightenGANAdapter(_BaseAdapter):
    """EnlightenGAN — Unet_resize_conv (self-attention U-Net, BMVC 2021).

    체크포인트: ``200_net_G_A.pth`` (공식 release). 입력 [-1,1] + gray attention.
    """
    name = "EnlightenGAN"

    def __init__(self, device: str, eg_repo: Path) -> None:
        super().__init__(device)
        # EnlightenGAN 의 networks.py 는 lib/nn 의 SynchronizedBN 등을 import.
        # CUDA 전용 SynchronizedBN 우회: syn_norm=False 로 두면 일반 BN 사용.
        with _isolated_repo_import(eg_repo, "models", "model"):
            from models.networks import Unet_resize_conv  # type: ignore
        self._cls = Unet_resize_conv

    def _make_opts(self) -> types.SimpleNamespace:
        # 공식 predict.py 가 쓰는 옵션 (sid_unet_resize / skip=1 / self_attn / use_norm=1)
        return types.SimpleNamespace(
            self_attention=True, use_norm=1, syn_norm=False, use_avgpool=0,
            tanh=False, times_residual=True, skip=1.0,
            linear_add=False, linear=False,
            latent_threshold=False, latent_norm=False,
            noise=0, input_linear=False,
        )

    def load(self, ckpt_path: Path) -> None:
        opt = self._make_opts()
        net = self._cls(opt, skip=opt.skip).to(self.device)
        sd = torch.load(ckpt_path, map_location=self.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # 일부 release 의 키에 module. 접두사가 붙어있을 수 있음
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        # strict=False — DropOut 등 학습 전용 키 차이 흡수
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing:
            # 핵심 weight 가 비면 알림 (BN running stats 같은 건 OK)
            critical = [m for m in missing if "running" not in m and "num_batches" not in m]
            if critical:
                print(f"  [warn] EnlightenGAN missing keys (critical): {critical[:5]}{' …' if len(critical)>5 else ''}")
        if unexpected:
            print(f"  [warn] EnlightenGAN unexpected keys: {unexpected[:5]}{' …' if len(unexpected)>5 else ''}")
        net.eval()
        self.model = net

    @torch.no_grad()
    def enhance(self, pil: Image.Image) -> np.ndarray:
        # 원본 (H, W, 3) → 텐서 [-1, 1]
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = t * 2.0 - 1.0                                # [-1, 1]
        # gray = 1 - (0.299*(r+1) + 0.587*(g+1) + 0.114*(b+1)) / 2   ∈ [0, 1]
        r = x[:, 0] + 1.0
        g = x[:, 1] + 1.0
        b = x[:, 2] + 1.0
        gray = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 2.0
        gray = gray.unsqueeze(1)                         # (1, 1, H, W)

        out, _latent = self.model(x, gray)
        out = out.clamp(-1.0, 1.0)
        out = (out + 1.0) * 0.5                          # [0, 1]
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (out * 255.0).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4) IAT (Illumination-Adaptive Transformer, BMVC 2022)
# ---------------------------------------------------------------------------
class IATAdapter(_BaseAdapter):
    """IAT — Illumination Adaptive Transformer.

    Forward: ``_, _, enhanced = IAT()(x)``; x 는 [0,1] (B, 3, H, W).
    출력은 clamp(1e-8, 1.0) → [0,1].
    """
    name = "IAT"

    def __init__(self, device: str, iat_repo: Path) -> None:
        super().__init__(device)
        # IAT 의 model/ 은 패키지 — Zero-DCE/SCI 의 동명 'model' (모듈) 캐시 격리 필수.
        with _isolated_repo_import(iat_repo, "model"):
            from model.IAT_main import IAT  # type: ignore
        self._cls = IAT

    def load(self, ckpt_path: Path) -> None:
        net = self._cls().to(self.device)
        sd = torch.load(ckpt_path, map_location=self.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        net.eval()
        self.model = net

    @torch.no_grad()
    def enhance(self, pil: Image.Image) -> np.ndarray:
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        _, _, enh = self.model(x)
        enh = enh.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (enh * 255.0).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5) LUNA (본 프로젝트, hybrid_v1)
# ---------------------------------------------------------------------------
class LUNAAdapter(_BaseAdapter):
    """LUNA — Light U-Net with Attention (이 repo 의 LightEnhanceGenerator).

    256×256 으로 학습됐기 때문에 입력은 BILINEAR 256×256 으로 맞추고,
    출력을 다시 원본 해상도로 BILINEAR 복원한다 (downstream_exdark.py 와 동일).
    """
    name = "LUNA"

    def __init__(self, device: str, image_size: int = 256) -> None:
        super().__init__(device)
        self.image_size = image_size

    def load(self, ckpt_path: Path) -> None:
        self.model = load_luna_generator(ckpt_path, device=self.device)

    @torch.no_grad()
    def enhance(self, pil: Image.Image) -> np.ndarray:
        W, H = pil.size
        norm = pil_to_norm_tensor(pil, self.image_size).to(self.device)
        out = self.model(norm)
        rgb256 = norm_tensor_to_uint8_rgb(out)             # (256, 256, 3) uint8
        if (H, W) != rgb256.shape[:2]:
            return np.array(Image.fromarray(rgb256).resize((W, H), Image.BILINEAR),
                            dtype=np.uint8)
        return rgb256


# ---------------------------------------------------------------------------
# 6) Original (baseline, 향상 없음)
# ---------------------------------------------------------------------------
class IdentityAdapter(_BaseAdapter):
    """원본 저조도 이미지 그대로 — baseline."""
    name = "Original"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self.model = torch.nn.Identity()  # placeholder

    def load(self, ckpt_path: Optional[Path]) -> None:
        return

    def enhance(self, pil: Image.Image) -> np.ndarray:
        return np.array(pil, dtype=np.uint8)


# ===========================================================================
# YOLO 추론 헬퍼 — RGB → (boxes, classes, confs)
# ===========================================================================
def _yolo_predict(model, rgb: np.ndarray, conf: float, device: str
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ultralytics YOLO 추론 + 클래스 필터링.

    * 입력은 BGR 컨벤션 → RGB[..., ::-1] 로 뒤집어 전달.
    * 검출 중 ExDark 12 클래스에 대응하는 COCO ID 만 유지.
    """
    bgr = rgb[..., ::-1].copy()
    res = model.predict(bgr, conf=conf, verbose=False, device=device)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32))
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
    cnf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    mask = np.array([c in TARGET_COCO_IDS for c in cls], dtype=bool)
    return xyxy[mask], cls[mask], cnf[mask]


# ===========================================================================
# 멀티 모델 비교표 / CSV
# ===========================================================================
def _fmt(x: Optional[float], d: int = 4) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{d}f}"


def print_overall_table(results: Dict[str, Dict[str, Any]],
                         timings: Dict[str, float]) -> None:
    """전체 (overall) mAP / P / R / 평균 시간 비교 표."""
    print(HRULE)
    print(" ExDark Downstream — Multi-Model Comparison (YOLOv8n COCO pre-trained, conf=0.25)")
    print(HRULE)
    print(f"  {'Model':<14} | {'mAP@0.5':>9} {'mAP@0.5:0.95':>13} | "
          f"{'Precision':>10} {'Recall':>9} | "
          f"{'#imgs':>7} {'#GT':>7} {'#det':>7} | {'sec/img':>8}")
    print(SUBRULE)
    # 기준선: Original 의 metric 으로 Δ 표시
    base = results.get("Original")
    for name, res in results.items():
        sec = timings.get(name, 0.0)
        d50 = ""
        if base is not None and name != "Original":
            d50 = f" ({res['map50']-base['map50']:+.3f})"
        print(f"  {name:<14} | {_fmt(res['map50'], 4):>9}{d50:<10} {_fmt(res['map'], 4):>13} | "
              f"{_fmt(res['p50'], 4):>10} {_fmt(res['r50'], 4):>9} | "
              f"{res['n_images']:>7} {res['n_gts']:>7} {res['n_preds']:>7} | "
              f"{sec:>8.3f}")
    print(HRULE)


def print_per_class_ap50(results: Dict[str, Dict[str, Any]]) -> None:
    """클래스별 AP@0.5 만 한눈에 보이는 가로 정렬 표."""
    print(" Per-Class AP@0.5")
    print(SUBRULE)
    header = f"  {'Class':<14} | {'#GT':>5} |"
    for name in results:
        header += f" {name:>13}"
    print(header)
    print(SUBRULE)
    classes = sorted(set().union(*[set(r["per_class"].keys()) for r in results.values()]))
    for cid in classes:
        # any model 의 메타데이터로 클래스 이름 / n_gt
        meta = None
        for r in results.values():
            if cid in r["per_class"]:
                meta = r["per_class"][cid]
                break
        cname = meta["name"] if meta else COCO_TO_EXDARK.get(cid, str(cid))
        n_gt = meta["n_gt"] if meta else 0
        row = f"  {cname:<14} | {n_gt:>5} |"
        for name, r in results.items():
            v = r["per_class"].get(cid, {}).get("ap50")
            row += f" {_fmt(v, 4):>13}"
        print(row)
    print(HRULE)


def save_comparison_csv(results: Dict[str, Dict[str, Any]],
                         timings: Dict[str, float],
                         out_path: Path) -> None:
    """모델 × (overall + per-class) 비교 CSV.

    스키마::

        model, scope, class, n_gt, n_pred, ap50, ap, precision, recall, sec_per_img

    ``scope`` 는 ``"overall"`` 또는 ``"class"``. ``class`` 는 overall 행에서
    빈 문자열, per-class 행에서는 ExDark 클래스명.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scope", "class", "n_gt", "n_pred",
                    "ap50", "ap", "precision", "recall", "sec_per_img"])
        for name, r in results.items():
            sec = timings.get(name, 0.0)
            # overall
            w.writerow([name, "overall", "", r["n_gts"], r["n_preds"],
                        f"{r['map50']:.6f}", f"{r['map']:.6f}",
                        f"{r['p50']:.6f}", f"{r['r50']:.6f}", f"{sec:.6f}"])
            # per-class
            for cid in sorted(r["per_class"].keys()):
                pc = r["per_class"][cid]
                w.writerow([name, "class", pc["name"], pc["n_gt"], pc["n_pred"],
                            f"{pc['ap50']:.6f}", f"{pc['ap']:.6f}",
                            f"{pc['p']:.6f}", f"{pc['r']:.6f}", ""])


# ===========================================================================
# Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ExDark Downstream Multi-Model 검출 비교 (Original / Zero-DCE / SCI / "
                    "EnlightenGAN / IAT / LUNA)",
    )
    p.add_argument("--exdark_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\ExDark")
    p.add_argument("--yolo_weights", type=str, default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (기본: cuda 가용 시 cuda)")

    # 각 모델의 가중치 경로 (지정 안 하면 평가에서 자동 제외)
    p.add_argument("--luna_ckpt",    type=str,
                   default="checkpoints/ext_lol_v2_real_stage2_best.pth")
    p.add_argument("--zerodce_ckpt", type=str, default="",
                   help="e.g. C:/대학교/Projects/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth")
    p.add_argument("--sci_ckpt", type=str, default="",
                   help="e.g. C:/대학교/Projects/SCI/CVPR/weights/medium.pt")
    p.add_argument("--enlightengan_ckpt", type=str, default="",
                   help="e.g. C:/대학교/Projects/EnlightenGAN/checkpoints/enlightening/200_net_G_A.pth")
    p.add_argument("--iat_ckpt", type=str, default="",
                   help="e.g. C:/대학교/Projects/Illumination-Adaptive-Transformer/IAT_enhance/best_Epoch_lol_v1.pth")

    # 각 모델의 코드 경로 (default 는 형제 폴더 가정)
    p.add_argument("--zerodce_repo", type=str,
                   default=r"C:\대학교\Projects\Zero-DCE\Zero-DCE_code")
    p.add_argument("--sci_repo", type=str,
                   default=r"C:\대학교\Projects\SCI\CVPR")
    p.add_argument("--enlightengan_repo", type=str,
                   default=r"C:\대학교\Projects\EnlightenGAN")
    p.add_argument("--iat_repo", type=str,
                   default=r"C:\대학교\Projects\Illumination-Adaptive-Transformer\IAT_enhance")
    p.add_argument("--luna_image_size", type=int, default=256,
                   help="LUNA 입력 해상도 (학습과 동일)")

    p.add_argument("--results_dir", type=str,
                   default="experiments/results/exdark_multi")
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 이면 전체, 양수면 앞에서 N 개만 (디버그)")
    p.add_argument("--all_splits", action="store_true",
                   help="split 필터 off — 7363 장 전체 (Train+Val+Test)")
    p.add_argument("--skip_models", type=str, default="",
                   help="콤마 구분 — 평가에서 제외할 모델 (e.g. 'IAT,LUNA')")

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def build_adapters(args: argparse.Namespace) -> List[_BaseAdapter]:
    """CLI 인자 → 평가할 어댑터 리스트 구성.

    * 가중치가 비어있거나 파일이 없으면 해당 모델은 자동 skip.
    * ``--skip_models`` 에 이름이 들어있으면 강제 제외.
    """
    skip = {s.strip() for s in args.skip_models.split(",") if s.strip()}
    plans: List[Tuple[str, Callable[[], _BaseAdapter], str]] = [
        ("Original",     lambda: IdentityAdapter(args.device), ""),
        ("Zero-DCE",     lambda: ZeroDCEAdapter(args.device, Path(args.zerodce_repo)),
                         args.zerodce_ckpt),
        ("SCI",          lambda: SCIAdapter(args.device, Path(args.sci_repo)),
                         args.sci_ckpt),
        ("EnlightenGAN", lambda: EnlightenGANAdapter(args.device, Path(args.enlightengan_repo)),
                         args.enlightengan_ckpt),
        ("IAT",          lambda: IATAdapter(args.device, Path(args.iat_repo)),
                         args.iat_ckpt),
        ("LUNA",         lambda: LUNAAdapter(args.device, image_size=args.luna_image_size),
                         args.luna_ckpt),
    ]
    adapters: List[_BaseAdapter] = []
    for name, factory, ckpt in plans:
        if name in skip:
            print(f"  [skip]  {name}: --skip_models 로 제외됨")
            continue
        if name != "Original":
            if not ckpt:
                print(f"  [skip]  {name}: 가중치 미지정 (--*_ckpt 비어있음)")
                continue
            if not Path(ckpt).is_file():
                print(f"  [skip]  {name}: 가중치 파일 없음 → {ckpt}")
                continue
        try:
            ad = factory()
            ad.load(Path(ckpt) if ckpt else None)
            n = ad.n_params()
            print(f"  [load]  {name:<13}  params={n:>10,}  ckpt={ckpt or '(none)'}")
            adapters.append(ad)
        except Exception as e:
            print(f"  [fail]  {name}: 로드 실패 — {type(e).__name__}: {e}")
    return adapters


def main() -> int:
    args = parse_args()
    device = args.device

    # ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics 미설치.  pip install ultralytics")
        return 1

    print(HRULE)
    print(" ExDark Multi-Model Detection Benchmark")
    print(HRULE)
    print(f"  device       : {device}")
    print(f"  exdark_root  : {args.exdark_root}")
    print(f"  yolo_weights : {args.yolo_weights}")
    print(f"  conf_thresh  : {args.conf}")
    print(SUBRULE)

    exdark_root = Path(args.exdark_root)
    if not (exdark_root / "images").is_dir() or not (exdark_root / "annotations").is_dir():
        print(f"[error] ExDark 폴더 구조가 맞지 않습니다: {exdark_root}")
        return 1

    # ---- 어댑터 구성 ----
    print(" Loading enhancement models …")
    adapters = build_adapters(args)
    if not adapters:
        print("[error] 평가할 모델이 0 개입니다 — 가중치 경로를 확인하세요.")
        return 1
    print(SUBRULE)

    # ---- YOLOv8 ----
    print(f" Loading YOLO  : {args.yolo_weights} …")
    yolo = YOLO(args.yolo_weights)
    print(f"   COCO classes: {len(yolo.names)}")
    print(SUBRULE)

    # ---- ExDark 샘플 수집 ----
    splits = None if args.all_splits else (3,)
    samples = collect_exdark_samples(exdark_root, splits=splits)
    if not samples:
        print("[error] 샘플이 0 개입니다 — imageclasslist.txt / annotation 확인 필요")
        return 1
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    split_desc = "Test only (split=3)" if not args.all_splits else "ALL splits (1+2+3)"
    print(f" samples       : {len(samples)}  [{split_desc}]")
    print(SUBRULE)

    # ---- 모델별 Accumulator + 시간 측정 ----
    accs: Dict[str, DetectionAccumulator] = {ad.name: DetectionAccumulator() for ad in adapters}
    timings_sum: Dict[str, float] = {ad.name: 0.0 for ad in adapters}
    timings_n: Dict[str, int] = {ad.name: 0 for ad in adapters}

    tqdm = _try_tqdm()
    iterator = tqdm(samples, desc="ExDark-multi", unit="img", ncols=110) if tqdm else samples

    # ---- 메인 루프 ----
    with torch.no_grad():
        for img_id, sm in enumerate(iterator):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] 이미지 로드 실패 {sm.image_path.name}: {e}")
                continue

            # GT (한 번만 파싱 — 모든 모델이 공유)
            gt_records = parse_bbgt_v3(sm.ann_path)
            if gt_records:
                gt_boxes = np.array(
                    [[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in gt_records],
                    dtype=np.float32,
                )
                gt_classes = np.array([c for (c, *_r) in gt_records], dtype=np.int64)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_classes = np.zeros((0,), dtype=np.int64)

            # 모델별 향상 + YOLO + 누적
            for ad in adapters:
                t0 = time.perf_counter()
                try:
                    enh_rgb = ad.enhance(pil)
                except Exception as e:
                    print(f"  [warn] {ad.name} enhance 실패 {sm.image_path.name}: {e}")
                    continue
                pb, pc, pf = _yolo_predict(yolo, enh_rgb, args.conf, device)
                accs[ad.name].add(img_id, gt_boxes, gt_classes, pb, pc, pf)
                timings_sum[ad.name] += (time.perf_counter() - t0)
                timings_n[ad.name] += 1

    # ---- 평가 ----
    print()
    print(" Computing mAP / Precision / Recall …")
    results: Dict[str, Dict[str, Any]] = {}
    for ad in adapters:
        results[ad.name] = accs[ad.name].compute()
    timings: Dict[str, float] = {
        name: (timings_sum[name] / timings_n[name]) if timings_n[name] else 0.0
        for name in results
    }

    # ---- 출력 ----
    print()
    print_overall_table(results, timings)
    print()
    print_per_class_ap50(results)

    # ---- CSV 저장 ----
    results_dir = Path(args.results_dir).resolve()
    csv_path = results_dir / "multi_model_comparison.csv"
    save_comparison_csv(results, timings, csv_path)
    print(f"  Saved CSV → {csv_path}")
    print(HRULE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
