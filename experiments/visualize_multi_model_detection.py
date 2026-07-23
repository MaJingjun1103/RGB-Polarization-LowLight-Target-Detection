"""ExDark Detection Qualitative Comparison — 6-Panel (Original / IAT / SCI / Zero-DCE / LUNA / GT).

목적
----
``downstream_exdark_multi.py`` 가 5 모델의 ``mAP`` 를 수치 비교한다면,
본 스크립트는 **정성 (qualitative)** 비교용 PNG 한 장을 생성한다::

    행 = 자동 선별된 ExDark 이미지 5 장
    열 = Original | IAT | SCI | Zero-DCE | LUNA | GT (총 6 패널)

각 모델 패널에는 해당 모델로 향상된 이미지 위에 YOLOv8n(COCO pre-trained)
검출 결과 bbox 가 그려진다. 모델별로 다른 색을 써서 어느 모델이 어떤
박스를 잡았는지 한눈에 구분할 수 있다.

선별 기준
---------
ExDark Test split (또는 ``--all_splits`` 로 전체)에서 GT annotation 만
빠르게 읽어 다음 조건을 모두 만족하는 이미지만 후보로 둔다.

* ``#bboxes  >= --min_boxes``    (기본 10)
* ``#classes >= --min_classes``  (기본 3)

그 다음 점수::

    score = (#unique_classes) * 100
          + (#interesting_classes ∈ {People, Car, Bus, Bicycle, Motorbike,
                                     Cat, Dog}) * 20
          + (#bboxes)

내림차순 정렬한 뒤, 이미지의 **메인 폴더(class_dir) 다양성**을 위해 동일
폴더가 이미 선택돼 있으면 greedy 로 한 번 건너뛰며 상위 N 장 선택.
``--samples 2015_02406.jpg,...`` 로 수동 지정도 가능.

사용 예
-------
.. code-block:: bash

    python experiments/visualize_multi_model_detection.py ^
        --luna_ckpt    checkpoints/loli_street_30000_stage2_best.pth ^
        --iat_ckpt     C:/대학교/Projects/Illumination-Adaptive-Transformer/IAT_enhance/best_Epoch_lol_v1.pth ^
        --sci_ckpt     C:/대학교/Projects/SCI/CVPR/weights/medium.pt ^
        --zerodce_ckpt C:/대학교/Projects/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth ^
        --conf 0.25
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --- 프로젝트 루트 (CODES/) 를 sys.path 에 등록 ---
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
from PIL import Image, ImageDraw, ImageFont

from experiments.downstream_exdark import (  # type: ignore
    COCO_TO_EXDARK,
    EXDARK_TO_COCO,
    TARGET_COCO_IDS,
    collect_exdark_samples,
    parse_bbgt_v3,
)
from experiments.downstream_exdark_multi import (  # type: ignore
    IATAdapter,
    IdentityAdapter,
    LUNAAdapter,
    SCIAdapter,
    ZeroDCEAdapter,
    _BaseAdapter,
)

HRULE = "=" * 100
SUBRULE = "-" * 100

# 패널 컬럼 순서 (좌 → 우)
COLUMN_ORDER: Tuple[str, ...] = ("Original", "IAT", "SCI", "Zero-DCE", "LUNA", "GT")

# 모델별 색상 (RGB) — bbox + 타이틀 배경에 동일하게 사용
PANEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Original": (220,  50,  50),   # 빨강
    "IAT":      (255, 140,  30),   # 주황
    "SCI":      (240, 210,  40),   # 노랑
    "Zero-DCE": ( 50, 110, 230),   # 파랑
    "LUNA":     ( 40, 200,  70),   # 초록
    "GT":       (100, 200, 240),   # 하늘색
}

# 선별 점수에서 가중치를 더 받는 "관심 객체" — 검출 비교 데모에서 유의미한 클래스
INTEREST_CLASSES: Set[str] = {
    "People", "Car", "Bus", "Bicycle", "Motorbike", "Cat", "Dog",
}


# ===========================================================================
# 선별 — annotation 만 빠르게 스캔 (모델 추론 없음)
# ===========================================================================
class _Score:
    __slots__ = ("n_box", "n_cls", "n_int", "total", "names")

    def __init__(self, n_box: int, n_cls: int, n_int: int,
                 total: int, names: Set[str]) -> None:
        self.n_box, self.n_cls, self.n_int = n_box, n_cls, n_int
        self.total, self.names = total, names


def score_sample(ann_path: Path) -> _Score:
    """(#box, #class, #interest, score) — score = #class*100 + #interest*20 + #box."""
    boxes = parse_bbgt_v3(ann_path)
    if not boxes:
        return _Score(0, 0, 0, 0, set())
    coco_ids = {c for (c, *_r) in boxes}
    names = {COCO_TO_EXDARK[c] for c in coco_ids if c in COCO_TO_EXDARK}
    n_int = len(names & INTEREST_CLASSES)
    total = len(coco_ids) * 100 + n_int * 20 + len(boxes)
    return _Score(len(boxes), len(coco_ids), n_int, total, names)


def select_samples(
    samples,
    n: int,
    min_boxes: int,
    min_classes: int,
    avoid_dup_class_dir: bool = True,
    forced_names: Optional[List[str]] = None,
):
    """하드 필터(#box, #class) + 점수 정렬 + 폴더 다양성 greedy 선택.

    Parameters
    ----------
    forced_names : list[str] | None
        파일명을 명시하면 그 순서로 우선 선택 (필터/스코어 무시).
    """
    picked: List = []
    if forced_names:
        idx_by_name = {s.image_path.name: s for s in samples}
        for nm in forced_names:
            sm = idx_by_name.get(nm.strip())
            if sm is not None and sm not in picked:
                picked.append(sm)
        if len(picked) >= n:
            return picked[:n], {sm.image_path.name: score_sample(sm.ann_path) for sm in picked[:n]}

    # 1) annotation 스캔 + 하드 필터
    cached: Dict[str, _Score] = {sm.image_path.name: score_sample(sm.ann_path) for sm in samples}
    pool = [
        sm for sm in samples
        if (sc := cached[sm.image_path.name]).n_box >= min_boxes
           and sc.n_cls >= min_classes
    ]
    # 2) 점수 정렬
    pool.sort(key=lambda sm: (
        -cached[sm.image_path.name].total,
        -cached[sm.image_path.name].n_cls,
        -cached[sm.image_path.name].n_box,
        sm.image_path.name,
    ))

    # 3) 폴더 다양성 greedy
    seen_dirs = {sm.class_dir for sm in picked}
    leftovers: List = []
    for sm in pool:
        if sm in picked:
            continue
        if avoid_dup_class_dir and sm.class_dir in seen_dirs:
            leftovers.append(sm)
            continue
        picked.append(sm)
        seen_dirs.add(sm.class_dir)
        if len(picked) >= n:
            return picked, cached

    # 4) 부족하면 중복 허용으로 채움
    for sm in leftovers:
        if len(picked) >= n:
            break
        picked.append(sm)
    return picked[:n], cached


# ===========================================================================
# YOLO 추론 — ExDark 12 클래스(=COCO 부분집합)만 유지
# ===========================================================================
def yolo_predict_filtered(model, rgb: np.ndarray, conf: float, device: str
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# 패널 렌더링
# ===========================================================================
def _load_font(size: int) -> ImageFont.ImageFont:
    """굵은 한글 안전 폰트 우선, 실패 시 기본."""
    for path in (
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        return 8 * len(text), 12


def _contrast_text(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """배경 밝기에 따라 라벨 텍스트를 흰/검정 자동 선택 (YIQ luminance)."""
    r, g, b = bg
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (10, 10, 10) if lum > 160 else (255, 255, 255)


def render_panel(
    rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: List[str],
    color: Tuple[int, int, int],
    title: str,
    panel_w: int,
    panel_h: int,
    title_bar_h: int = 30,
) -> np.ndarray:
    """단일 패널: letterbox 향상 이미지 + bbox + 상단 타이틀 띠 (해당 모델 색).

    원본 좌표는 letterbox 변환 그대로 스케일링된다.
    """
    H_in, W_in = rgb.shape[:2]
    scale = min(panel_w / W_in, panel_h / H_in)
    new_w = max(1, int(round(W_in * scale)))
    new_h = max(1, int(round(H_in * scale)))

    pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (panel_w, panel_h + title_bar_h), (32, 32, 32))
    # 타이틀 띠 — 모델 색
    bar = Image.new("RGB", (panel_w, title_bar_h), color)
    canvas.paste(bar, (0, 0))

    off_x = (panel_w - new_w) // 2
    off_y = title_bar_h + (panel_h - new_h) // 2
    canvas.paste(pil_img, (off_x, off_y))

    draw = ImageDraw.Draw(canvas)
    font_box = _load_font(max(11, panel_h // 28))
    font_title = _load_font(max(14, title_bar_h - 12))
    text_color = _contrast_text(color)

    # bbox
    line_w = max(2, panel_h // 180)
    for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
        bx1 = off_x + x1 * scale
        by1 = off_y + y1 * scale
        bx2 = off_x + x2 * scale
        by2 = off_y + y2 * scale
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=line_w)
        tw, th = _text_size(draw, lab, font_box)
        ty = max(int(by1) - th - 4, title_bar_h)
        draw.rectangle([bx1, ty, bx1 + tw + 6, ty + th + 3], fill=color)
        draw.text((bx1 + 3, ty + 1), lab, fill=text_color, font=font_box)

    # 타이틀
    tw, th = _text_size(draw, title, font_title)
    draw.text(((panel_w - tw) // 2, max(0, (title_bar_h - th) // 2 - 1)),
              title, fill=text_color, font=font_title)

    return np.array(canvas, dtype=np.uint8)


# ===========================================================================
# 그리드 조립
# ===========================================================================
def build_row(panels: List[np.ndarray], gap: int = 6) -> np.ndarray:
    h = panels[0].shape[0]
    sep = np.full((h, gap, 3), 255, dtype=np.uint8)
    out: List[np.ndarray] = []
    for i, p in enumerate(panels):
        if i > 0:
            out.append(sep)
        out.append(p)
    return np.concatenate(out, axis=1)


def build_grid(rows: List[np.ndarray], gap: int = 8) -> np.ndarray:
    w = rows[0].shape[1]
    sep = np.full((gap, w, 3), 255, dtype=np.uint8)
    out: List[np.ndarray] = []
    for i, r in enumerate(rows):
        if i > 0:
            out.append(sep)
        out.append(r)
    return np.concatenate(out, axis=0)


# ===========================================================================
# 어댑터 로드 (Original, IAT, SCI, Zero-DCE, LUNA — 사용자 지정 순서)
# ===========================================================================
def build_adapters(args: argparse.Namespace) -> Dict[str, _BaseAdapter]:
    plans: List[Tuple[str, str]] = [
        ("Original",  ""),
        ("IAT",       args.iat_ckpt),
        ("SCI",       args.sci_ckpt),
        ("Zero-DCE",  args.zerodce_ckpt),
        ("LUNA",      args.luna_ckpt),
    ]
    factories = {
        "Original":  lambda: IdentityAdapter(args.device),
        "IAT":       lambda: IATAdapter(args.device, Path(args.iat_repo)),
        "SCI":       lambda: SCIAdapter(args.device, Path(args.sci_repo)),
        "Zero-DCE":  lambda: ZeroDCEAdapter(args.device, Path(args.zerodce_repo)),
        "LUNA":      lambda: LUNAAdapter(args.device, image_size=args.luna_image_size),
    }
    adapters: Dict[str, _BaseAdapter] = {}
    for name, ckpt in plans:
        if name != "Original":
            if not ckpt:
                print(f"  [skip]  {name}: 가중치 미지정 — 빈 패널로 표시됨")
                continue
            if not Path(ckpt).is_file():
                print(f"  [skip]  {name}: 가중치 파일 없음 → {ckpt}")
                continue
        try:
            ad = factories[name]()
            ad.load(Path(ckpt) if ckpt else None)
            print(f"  [load]  {name:<10}  params={ad.n_params():>10,}  ckpt={ckpt or '(none)'}")
            adapters[name] = ad
        except Exception as e:
            print(f"  [fail]  {name}: 로드 실패 — {type(e).__name__}: {e}")
    return adapters


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ExDark 6-패널 정성 비교 (Original / IAT / SCI / Zero-DCE / LUNA / GT)",
    )
    p.add_argument("--exdark_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\ExDark")
    p.add_argument("--yolo_weights", type=str, default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default=None)

    # 가중치 (요청 사양 기본값)
    p.add_argument("--luna_ckpt",    type=str,
                   default="checkpoints/loli_street_30000_stage2_best.pth")
    p.add_argument("--iat_ckpt",     type=str,
                   default=r"C:/대학교/Projects/Illumination-Adaptive-Transformer/IAT_enhance/best_Epoch_lol_v1.pth")
    p.add_argument("--sci_ckpt",     type=str,
                   default=r"C:/대학교/Projects/SCI/CVPR/weights/medium.pt")
    p.add_argument("--zerodce_ckpt", type=str,
                   default=r"C:/대학교/Projects/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth")

    # 외부 repo 경로
    p.add_argument("--zerodce_repo", type=str,
                   default=r"C:\대학교\Projects\Zero-DCE\Zero-DCE_code")
    p.add_argument("--sci_repo", type=str,
                   default=r"C:\대학교\Projects\SCI\CVPR")
    p.add_argument("--iat_repo", type=str,
                   default=r"C:\대학교\Projects\Illumination-Adaptive-Transformer\IAT_enhance")
    p.add_argument("--luna_image_size", type=int, default=256)

    # 선별 / 시각화 옵션
    p.add_argument("--num_images", type=int, default=5,
                   help="비교 행 수 (선별할 이미지 개수)")
    p.add_argument("--min_boxes", type=int, default=10,
                   help="GT bbox 최소 개수 (하드 필터)")
    p.add_argument("--min_classes", type=int, default=3,
                   help="GT 내 distinct class 최소 수 (하드 필터)")
    p.add_argument("--samples", type=str, default="",
                   help="콤마 구분 파일명 — 자동 선별 대신 강제 지정")
    p.add_argument("--allow_dup_class_dir", action="store_true",
                   help="이미지 폴더(class_dir) 중복 허용 (다양성 끄기)")
    p.add_argument("--all_splits", action="store_true",
                   help="후보 풀을 Train+Val+Test 로 확장 (기본은 Test only)")
    p.add_argument("--panel_w", type=int, default=480)
    p.add_argument("--panel_h", type=int, default=320)
    p.add_argument("--out_dir", type=str,
                   default="experiments/results/exdark_multi/visual",
                   help="PNG 저장 폴더")
    p.add_argument("--out_name", type=str, default="",
                   help="PNG 파일명 (생략 시 타임스탬프 자동)")
    p.add_argument("--per_image", action="store_true",
                   help="각 행(이미지 1장)도 개별 PNG 로 추가 저장")

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    args = parse_args()
    device = args.device

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics 미설치.  pip install ultralytics")
        return 1

    print(HRULE)
    print(" ExDark Multi-Model Qualitative (6-panel)")
    print(HRULE)
    print(f"  device       : {device}")
    print(f"  exdark_root  : {args.exdark_root}")
    print(f"  yolo_weights : {args.yolo_weights}  (conf={args.conf})")
    print(f"  filters      : #box>={args.min_boxes}, #class>={args.min_classes}")
    print(f"  out_dir      : {args.out_dir}")
    print(SUBRULE)

    exdark_root = Path(args.exdark_root)
    if not (exdark_root / "images").is_dir() or not (exdark_root / "annotations").is_dir():
        print(f"[error] ExDark 폴더 구조 오류: {exdark_root}")
        return 1

    # ---- 어댑터 로드 ----
    print(" Loading enhancement models …")
    adapters = build_adapters(args)
    if "Original" not in adapters:
        adapters["Original"] = IdentityAdapter(device)
        adapters["Original"].load(None)
    print(SUBRULE)

    # ---- YOLO ----
    print(f" Loading YOLO  : {args.yolo_weights} …")
    yolo = YOLO(args.yolo_weights)
    print(SUBRULE)

    # ---- 후보 + 선별 ----
    splits = None if args.all_splits else (3,)
    samples = collect_exdark_samples(exdark_root, splits=splits)
    if not samples:
        print("[error] 후보 샘플 0 개")
        return 1
    forced = [s for s in args.samples.split(",") if s.strip()] if args.samples else None
    picked, cached = select_samples(
        samples, args.num_images,
        min_boxes=args.min_boxes,
        min_classes=args.min_classes,
        avoid_dup_class_dir=not args.allow_dup_class_dir,
        forced_names=forced,
    )
    if not picked:
        print(f"[error] 선별된 이미지 0 장 — 필터(#box>={args.min_boxes}, "
              f"#class>={args.min_classes}) 를 완화해보세요")
        return 1
    print(" Selected images:")
    for i, sm in enumerate(picked):
        sc = cached.get(sm.image_path.name) or score_sample(sm.ann_path)
        cls_str = ",".join(sorted(sc.names))
        print(f"   [{i+1}] {sm.image_path.name:<22} dir={sm.class_dir:<9} "
              f"#cls={sc.n_cls} #box={sc.n_box} #int={sc.n_int} "
              f"score={sc.total:>4}  classes=[{cls_str}]")
    print(SUBRULE)

    # ---- 6 패널 행 생성 ----
    rows: List[np.ndarray] = []
    per_image_paths: List[Tuple[str, np.ndarray]] = []
    panel_w, panel_h = args.panel_w, args.panel_h

    for sm in picked:
        try:
            pil = Image.open(sm.image_path).convert("RGB")
        except Exception as e:
            print(f"  [warn] 이미지 로드 실패 {sm.image_path.name}: {e}")
            continue

        # GT
        gt_records = parse_bbgt_v3(sm.ann_path)
        gt_boxes = np.array(
            [[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in gt_records],
            dtype=np.float32,
        ) if gt_records else np.zeros((0, 4), dtype=np.float32)
        gt_classes = np.array(
            [c for (c, *_r) in gt_records], dtype=np.int64,
        ) if gt_records else np.zeros((0,), dtype=np.int64)
        gt_labels = [COCO_TO_EXDARK.get(int(c), str(int(c))) for c in gt_classes]

        panels: List[np.ndarray] = []
        for col in COLUMN_ORDER:
            color = PANEL_COLORS[col]
            if col == "GT":
                panel = render_panel(
                    np.array(pil, dtype=np.uint8),
                    gt_boxes, gt_labels, color,
                    title=f"GT ({len(gt_boxes)} obj)",
                    panel_w=panel_w, panel_h=panel_h,
                )
                panels.append(panel)
                continue

            ad = adapters.get(col)
            if ad is None:
                blank = np.full((panel_h, panel_w, 3), 60, dtype=np.uint8)
                panels.append(render_panel(
                    blank, np.zeros((0, 4), np.float32), [], color,
                    title=f"{col}  (skipped)",
                    panel_w=panel_w, panel_h=panel_h,
                ))
                continue

            try:
                enh_rgb = ad.enhance(pil)
            except Exception as e:
                print(f"  [warn] {col} enhance 실패 {sm.image_path.name}: {e}")
                enh_rgb = np.array(pil, dtype=np.uint8)

            pb, pc, pf = yolo_predict_filtered(yolo, enh_rgb, args.conf, device)
            pred_labels = [
                f"{COCO_TO_EXDARK.get(int(c), str(int(c)))} {pf[i]:.2f}"
                for i, c in enumerate(pc)
            ]
            panels.append(render_panel(
                enh_rgb, pb, pred_labels, color,
                title=f"{col} ({len(pb)} det)",
                panel_w=panel_w, panel_h=panel_h,
            ))

        row = build_row(panels, gap=6)
        rows.append(row)
        per_image_paths.append((sm.image_path.stem, row))

    if not rows:
        print("[error] 만들어진 행 0 개")
        return 1

    # ---- 좌측 라벨 + 그리드 ----
    grid = build_grid(rows, gap=8)
    label_w = 170
    title_bar_h = 30
    row_step = (panel_h + title_bar_h) + 8

    annotated = Image.new("RGB", (grid.shape[1] + label_w, grid.shape[0]), (255, 255, 255))
    annotated.paste(Image.fromarray(grid), (label_w, 0))
    draw = ImageDraw.Draw(annotated)
    font_left = _load_font(15)
    for i, sm in enumerate(picked):
        y_center = i * row_step + (panel_h + title_bar_h) // 2
        text = f"{sm.image_path.stem}\n[{sm.class_dir}]"
        draw.multiline_text((10, y_center - 18), text,
                            fill=(20, 20, 20), font=font_left, spacing=2)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"comparison_6panel_{time.strftime('%Y%m%d_%H%M%S')}.png"
    out_path = out_dir / out_name
    annotated.save(out_path)
    print(f"  Saved grid → {out_path}  ({annotated.size[0]}×{annotated.size[1]})")

    # ---- per-image 옵션 ----
    if args.per_image:
        for stem, row in per_image_paths:
            p = out_dir / f"{stem}_6panel.png"
            Image.fromarray(row).save(p)
            print(f"  Saved row  → {p}")

    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
