"""Phase 2: text -> 3D pipeline with CLI args.

Chains HunyuanDiT (text-to-image, Tencent native) into Hunyuan3D-2.1's
image-to-3D + texture pipeline. Each run lands in its own directory under
outputs/ so multiple prompts don't overwrite each other.

Examples:
    # Minimal — auto-generates outputs/<timestamp>_<slug>/
    python text_to_3d.py --prompt "a chibi-style fox figurine"

    # Named output directory for repeated iteration on one subject
    python text_to_3d.py --prompt "..." --output-dir outputs/chess

    # Shape only (skip the slow paint stage, ~30s vs ~10min)
    python text_to_3d.py --prompt "..." --skip-paint

    # More views + higher texture resolution
    python text_to_3d.py --prompt "..." --views 9 --resolution 768

Outputs per run:
    input.png        T2I output (with background)
    input_rmbg.png   after background removal
    shape.glb        untextured shape mesh
    textured.glb     final textured mesh   (skipped if --skip-paint)
    metadata.json    prompt, seed, settings, timings
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

import torch
from diffusers import HunyuanDiTPipeline

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception:
    pass


def slugify(text, max_len=40):
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:max_len] or "run"


def parse_args():
    p = argparse.ArgumentParser(
        description="text -> 3D via HunyuanDiT + Hunyuan3D-2.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt", required=True, help="text prompt for HunyuanDiT")
    p.add_argument("--output-dir", default=None,
                   help="output directory (default: outputs/<timestamp>_<slug>/)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-inference-steps", type=int, default=25,
                   help="HunyuanDiT denoising steps")
    p.add_argument("--guidance-scale", type=float, default=6.0,
                   help="HunyuanDiT classifier-free guidance")
    p.add_argument("--views", type=int, default=6,
                   help="paint pipeline multiview count (6-9)")
    p.add_argument("--resolution", type=int, default=512, choices=[512, 768],
                   help="paint texture resolution")
    p.add_argument("--skip-rembg", action="store_true",
                   help="skip background removal (use if input already has alpha)")
    p.add_argument("--skip-paint", action="store_true",
                   help="skip texture stage; only generate the untextured shape")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"outputs/{ts}_{slugify(args.prompt)}"
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "input": outdir / "input.png",
        "input_rmbg": outdir / "input_rmbg.png",
        "shape": outdir / "shape.glb",
        "textured": outdir / "textured.glb",
        "metadata": outdir / "metadata.json",
    }

    t_start = time.time()
    print(f"[run] output dir : {outdir}")
    print(f"[run] prompt     : {args.prompt}")
    print(f"[run] seed       : {args.seed}")

    # ---------------------------------------------------------------------
    # Stage 0: text -> image
    # ---------------------------------------------------------------------
    t = time.time()
    print("[t2i] loading HunyuanDiT...")
    t2i = HunyuanDiTPipeline.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        torch_dtype=torch.float16,
    ).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = t2i(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    image.save(paths["input"])
    print(f"[t2i] saved -> {paths['input']}  ({time.time()-t:.1f}s)")

    del t2i, generator
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # Stage 0.5: background removal
    # ---------------------------------------------------------------------
    if args.skip_rembg:
        image = image.convert("RGBA")
        image.save(paths["input_rmbg"])
        print(f"[rembg] skipped (saved RGBA as-is -> {paths['input_rmbg']})")
    else:
        t = time.time()
        rembg = BackgroundRemover()
        image = rembg(image.convert("RGB")).convert("RGBA")
        image.save(paths["input_rmbg"])
        print(f"[rembg] saved -> {paths['input_rmbg']}  ({time.time()-t:.1f}s)")

    # ---------------------------------------------------------------------
    # Stage 1: shape generation
    # ---------------------------------------------------------------------
    t = time.time()
    print("[shape] loading pipeline...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    mesh = shape_pipeline(image=image)[0]
    mesh.export(str(paths["shape"]))
    print(f"[shape] saved -> {paths['shape']}  ({time.time()-t:.1f}s)")

    # ---------------------------------------------------------------------
    # Stage 2: texture generation
    # ---------------------------------------------------------------------
    if args.skip_paint:
        print("[paint] skipped")
    else:
        t = time.time()
        print("[paint] loading pipeline...")
        conf = Hunyuan3DPaintConfig(max_num_view=args.views, resolution=args.resolution)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        paint_pipeline(
            mesh_path=str(paths["shape"]),
            image_path=str(paths["input_rmbg"]),
            output_mesh_path=str(paths["textured"]),
        )
        print(f"[paint] saved -> {paths['textured']}  ({time.time()-t:.1f}s)")

    metadata = {
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "views": args.views,
        "resolution": args.resolution,
        "skip_rembg": args.skip_rembg,
        "skip_paint": args.skip_paint,
        "outputs": {k: str(v) for k, v in paths.items() if k != "metadata"},
        "total_seconds": round(time.time() - t_start, 1),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    final_glb = paths["shape"] if args.skip_paint else paths["textured"]
    viewer_path = "/" + str(final_glb).replace(os.sep, "/")
    print()
    print("=" * 60)
    print(f"Done in {time.time()-t_start:.1f}s")
    print(f"Output dir : {outdir}")
    print(f"View at    : http://localhost:7861/viewer/")
    print(f"  → Path   : {viewer_path}")
    print("=" * 60)

    # bpy cleanup segfault avoidance
    os._exit(0)


if __name__ == "__main__":
    main()
