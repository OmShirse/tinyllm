#!/usr/bin/env python3
"""
export_weights.py — Export an existing TinyLLM checkpoint to weights.h

Use this when you already have a trained tinyllm.pt and just want to regenerate
the C header (e.g. after changing quantisation strategy).

Usage:
  python3 export_weights.py
  python3 export_weights.py --checkpoint my_run.pt
  python3 export_weights.py --checkpoint my_run.pt --out /path/to/weights.h
"""

import argparse
from pathlib import Path

import torch

# Import everything from the training script
from train_tinyllm import TinyLLM, export_weights, DEVICE, CHECKPOINT, OUTPUT_H


def main():
    parser = argparse.ArgumentParser(description="Export TinyLLM checkpoint → weights.h")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT),
                        help=f"Path to .pt checkpoint (default: {CHECKPOINT})")
    parser.add_argument("--out",        type=str, default=str(OUTPUT_H),
                        help=f"Output path for weights.h (default: {OUTPUT_H})")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        print("Run train_tinyllm.py first to generate a checkpoint.")
        raise SystemExit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    model = TinyLLM().to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    model.eval()

    export_weights(model, Path(args.out))

    print("\nRebuild the ESP-IDF project:")
    print("  cd llm && idf.py build && idf.py flash monitor")


if __name__ == "__main__":
    main()

