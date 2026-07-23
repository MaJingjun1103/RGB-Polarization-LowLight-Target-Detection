"""단일 이미지 저조도 향상 (LUNA / hybrid_v1).

학습된 Generator 로 한 장의 이미지를 밝게(저조도→정상조도) 복원한다.
평가 스크립트와 달리 정답(high) 페어가 필요 없고, 원본 해상도를 보존한다.

* 입력 이미지를 RGB 로 로드 → [-1, 1] 정규화 (학습과 동일: to_tensor*2-1)
* U-Net 의 stride-2 다운샘플 3회를 위해 가로/세로를 8 의 배수로 reflect-pad
* G(x).clamp(-1,1) → (x+1)*127.5 역정규화 → pad 영역 crop → 저장

사용 예::

    python enhance_image.py --input ../../1.webp --output ../../1_enhanced.png
    python enhance_image.py --input ../../1.webp            # 기본 출력: <stem>_enhanced.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 콘솔 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from models import LightEnhanceGenerator


def load_generator(checkpoint_path: Path, device: str) -> LightEnhanceGenerator:
    """체크포인트에서 Generator 만 로드. 저장된 conv_config/base_filters 로 구조 복원."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    base_filters = 32
    conv_config = None
    use_attention = True
    if isinstance(state, dict):
        base_filters = int(state.get("base_filters", base_filters))
        conv_config = state.get("conv_config", None)
        use_attention = bool(state.get("use_attention", use_attention))

    G = LightEnhanceGenerator(
        base_filters=base_filters,
        use_attention=use_attention,
        conv_config=conv_config,
    ).to(device)

    g_state = state["generator"] if (isinstance(state, dict) and "generator" in state) else state
    G.load_state_dict(g_state)
    G.eval()
    return G


@torch.no_grad()
def enhance(G: LightEnhanceGenerator, img: Image.Image, device: str) -> Image.Image:
    """단일 PIL 이미지를 향상하여 PIL 이미지로 반환 (원본 해상도 유지)."""
    img = img.convert("RGB")
    W, H = img.size

    # [-1, 1] 정규화 (학습 transform 과 동일: to_tensor*2-1)
    t = (TF.to_tensor(img) * 2.0 - 1.0).unsqueeze(0).to(device)  # (1,3,H,W)

    # 8 의 배수로 reflect-pad (stride-2 ×3 = 8)
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")

    fake = G(t).clamp(-1.0, 1.0)
    fake = fake[:, :, :H, :W]  # pad 영역 제거

    arr = ((fake.squeeze(0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LUNA 단일 이미지 저조도 향상")
    parser.add_argument("--input", required=True, help="입력 이미지 경로 (저조도)")
    parser.add_argument("--output", default=None, help="출력 경로 (기본: <stem>_enhanced.png)")
    parser.add_argument(
        "--checkpoint",
        default=str(here / "checkpoints" / "hybrid_v1_stage2_best.pth"),
        help="Generator 체크포인트 경로",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda 또는 cpu",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"[error] input not found: {in_path}")
        return 1
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_enhanced.png")

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"[error] checkpoint not found: {ckpt}")
        return 1

    print(f"  checkpoint : {ckpt}")
    print(f"  device     : {args.device}")
    print(f"  input      : {in_path.resolve()}")

    G = load_generator(ckpt, args.device)
    img = Image.open(in_path)
    print(f"  size       : {img.size[0]}x{img.size[1]}")

    out = enhance(G, img, args.device)
    out.save(out_path)
    print(f"  saved      : {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
