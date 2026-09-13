#!/usr/bin/env python3
"""
train_tinyllm.py — Train TinyLLM and export INT8-quantised weights as weights.h

Architecture (must match components/tinyllm/include/tinyllm.h):
  vocab_size  = 95    (printable ASCII 0x20–0x7E)
  n_embd      = 64
  n_head      = 4
  n_layer     = 2
  block_size  = 32
  ff_dim      = 256

Usage:
  # Download Shakespeare corpus and train from scratch:
  python3 train_tinyllm.py

  # Train on a custom text file:
  python3 train_tinyllm.py --corpus my_corpus.txt

  # Load an existing checkpoint and just export:
  python3 train_tinyllm.py --checkpoint tinyllm.pt --export-only

Requirements:
  pip install torch numpy

The generated weights.h is placed in:
  ../components/tinyllm/include/weights.h
"""

import argparse
import math
import os
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters (MUST match tinyllm.h) ────────────────────────────────────
VOCAB_SIZE  = 95
N_EMBD      = 64
N_HEAD      = 4
N_LAYER     = 2
VOCAB_SIZE    = 96                # printable ASCII 0x20-0x7E (95) + newline (1)
N_EMBD        = 64
N_HEAD        = 4
N_LAYER       = 2
BLOCK_SIZE    = 32
FF_DIM        = N_EMBD * 4          # 256
HEAD_SIZE     = N_EMBD // N_HEAD    # 16
CHAR_OFFSET   = 0x20                # First printable ASCII
TOKEN_NEWLINE = 95                # '\n' token ID

OUTPUT_H       = Path(__file__).parent.parent / "components" / "tinyllm" / "include" / "weights.h"
CHECKPOINT     = Path("tinyllm.pt")
DEFAULT_CORPUS = Path(__file__).parent / "corpus" / "iot_corpus.txt"

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE    = 32       # smaller for tiny corpus
MAX_ITERS     = 50000    # overfit the small corpus
EVAL_ITERS    = 100
EVAL_INTERVAL = 5000
LR            = 1e-3     # higher LR for fast convergence on small data
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(path: str | None) -> str:
    """Load text corpus. Defaults to the IoT corpus."""
    if path:
        print(f"Loading corpus from {path}")
        return Path(path).read_text(encoding="utf-8", errors="replace")
    if DEFAULT_CORPUS.exists():
        print(f"Using IoT corpus: {DEFAULT_CORPUS}")
        return DEFAULT_CORPUS.read_text(encoding="utf-8")
    # Fallback: download Shakespeare
    cache = Path("shakespeare.txt")
    if not cache.exists():
        print(f"Downloading Shakespeare corpus...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, cache)
    return cache.read_text(encoding="utf-8")


def encode(text: str) -> list[int]:
    """Encode text to token IDs. Handles printable ASCII + newline."""
    tokens = []
    for c in text:
        code = ord(c)
        if 0x20 <= code <= 0x7E:
            tokens.append(code - CHAR_OFFSET)
        elif c == '\n':
            tokens.append(TOKEN_NEWLINE)
    return tokens


def decode(tokens: list[int]) -> str:
    chars = []
    for t in tokens:
        if 0 <= t < 95:
            chars.append(chr(t + CHAR_OFFSET))
        elif t == TOKEN_NEWLINE:
            chars.append('\n')
    return "".join(chars)


def get_batch(data: torch.Tensor, batch_size: int):
    ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x  = torch.stack([data[i:i + BLOCK_SIZE]     for i in ix])
    y  = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv  = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD,     bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        q = q.view(B, T, N_HEAD, HEAD_SIZE).transpose(1, 2)
        k = k.view(B, T, N_HEAD, HEAD_SIZE).transpose(1, 2)
        v = v.view(B, T, N_HEAD, HEAD_SIZE).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_SIZE))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y   = att @ v
        y   = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(N_EMBD, FF_DIM, bias=False)
        self.proj = nn.Linear(FF_DIM, N_EMBD, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.RMSNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.RMSNorm(N_EMBD)
        self.ffn  = FFN()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.blocks    = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f      = nn.RMSNorm(N_EMBD)
        # Weight tying: lm_head shares token_emb weights
        self.lm_head   = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits   = self(idx_cond)
            logits   = logits[:, -1, :] / temperature
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model: TinyLLM, data_splits: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    losses = {}
    for split, data in data_splits.items():
        total = 0.0
        for _ in range(EVAL_ITERS):
            x, y = get_batch(data, BATCH_SIZE)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / EVAL_ITERS
    model.train()
    return losses


def train(corpus: str, model: TinyLLM | None = None) -> TinyLLM:
    tokens  = encode(corpus)
    data    = torch.tensor(tokens, dtype=torch.long)
    n_val   = max(1, int(0.1 * len(data)))
    train_d = data[:-n_val]
    val_d   = data[-n_val:]
    splits  = {"train": train_d, "val": val_d}

    if model is None:
        model = TinyLLM().to(DEVICE)
    print(f"Model parameters: {model.param_count():,}")
    print(f"Corpus tokens: {len(tokens):,}  |  Training on: {DEVICE}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=MAX_ITERS)

    t0 = time.time()
    for step in range(1, MAX_ITERS + 1):
        x, y = get_batch(train_d, BATCH_SIZE)
        _, loss = model(x, y)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            losses = estimate_loss(model, splits)
            elapsed = time.time() - t0
            print(f"step {step:5d}/{MAX_ITERS} | "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f} | "
                  f"{elapsed:.1f}s elapsed")
            # Sample a few characters
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            out = model.generate(ctx, 80, temperature=0.8)
            sample_text = decode(out[0].tolist()[1:])
            print(f"  Sample: {sample_text!r}")

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\nCheckpoint saved to {CHECKPOINT}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# INT8 quantisation and export
# ══════════════════════════════════════════════════════════════════════════════

def quantize_to_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantisation. Returns (q_int8, scale_float32)."""
    abs_max = np.abs(w).max()
    if abs_max == 0:
        return np.zeros_like(w, dtype=np.int8), 1.0
    scale  = abs_max / 127.0
    q      = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, float(scale)


def array_to_c_int8(name: str, arr: np.ndarray, comment: str = "") -> str:
    """Render a flat int8 numpy array as a C const int8_t array literal."""
    flat   = arr.flatten()
    n      = len(flat)
    values = ", ".join(str(int(v)) for v in flat)
    cmt    = f"  /* {comment} */" if comment else ""
    return f"const int8_t {name}[{n}] = {{{values}}};{cmt}\n"


def array_to_c_float(name: str, arr: np.ndarray, comment: str = "") -> str:
    """Render a flat float32 numpy array as a C const float array literal."""
    flat   = arr.flatten()
    n      = len(flat)
    values = ", ".join(f"{float(v):.8f}f" for v in flat)
    cmt    = f"  /* {comment} */" if comment else ""
    return f"const float {name}[{n}] = {{{values}}};{cmt}\n"


def export_weights(model: TinyLLM, out_path: Path) -> None:
    """Export all model weights as an INT8-quantised C header file."""
    sd = {k: v.cpu().float().numpy() for k, v in model.state_dict().items()}

    lines: list[str] = []
    lines.append("/* AUTO-GENERATED by tools/train_tinyllm.py — do NOT edit by hand */\n")
    lines.append("#pragma once\n")
    lines.append('#include <stdint.h>\n')
    lines.append('#include "tinyllm.h"\n\n')
    lines.append("/* Dimension shorthands */\n")
    lines.append("#define _V   TINYLLM_VOCAB_SIZE\n")
    lines.append("#define _E   TINYLLM_N_EMBD\n")
    lines.append("#define _L   TINYLLM_N_LAYER\n")
    lines.append("#define _F   TINYLLM_FF_DIM\n\n")

    # Token embedding (also used as lm_head, weight-tied)
    emb_w            = sd["token_emb.weight"]             # [V, E]
    emb_q, emb_scale = quantize_to_int8(emb_w)
    lines.append("/* ── Token embedding / LM head (weight-tied) ────────────────────────── */\n")
    lines.append(array_to_c_int8("g_token_emb", emb_q, f"{VOCAB_SIZE}×{N_EMBD}"))
    lines.append(f"const float g_token_emb_scale = {emb_scale:.8f}f;\n\n")

    # LM head scale (same weights, independent scale for the lm_head direction)
    lm_q, lm_scale = quantize_to_int8(emb_w)
    lines.append(f"const float g_lm_head_scale = {lm_scale:.8f}f;\n\n")

    # Per-layer weights
    lines.append("/* ── Per-layer RMSNorm (fp32, kept exact) ────────────────────────────── */\n")
    for l in range(N_LAYER):
        ln1 = sd[f"blocks.{l}.ln1.weight"]
        ln2 = sd[f"blocks.{l}.ln2.weight"]
        lines.append(f"/* layer {l} */\n")
        lines.append(array_to_c_float(f"_g_ln1_l{l}", ln1))
        lines.append(array_to_c_float(f"_g_ln2_l{l}", ln2))

    # Aggregate into arrays
    lines.append(f"\nconst float g_ln1_weight[_L][_E] = {{\n")
    for l in range(N_LAYER):
        ln1 = sd[f"blocks.{l}.ln1.weight"]
        vals = ", ".join(f"{v:.8f}f" for v in ln1.flatten())
        lines.append(f"    /* layer {l} */ {{{vals}}},\n")
    lines.append("};\n\n")

    lines.append(f"const float g_ln2_weight[_L][_E] = {{\n")
    for l in range(N_LAYER):
        ln2 = sd[f"blocks.{l}.ln2.weight"]
        vals = ", ".join(f"{v:.8f}f" for v in ln2.flatten())
        lines.append(f"    /* layer {l} */ {{{vals}}},\n")
    lines.append("};\n\n")

    # Attention QKV [L][3*E*E]
    lines.append("/* ── Attention QKV projection ────────────────────────────────────────── */\n")
    lines.append(f"const int8_t g_attn_qkv_weight[_L][3 * _E * _E] = {{\n")
    attn_qkv_scales = []
    for l in range(N_LAYER):
        qkv_w = sd[f"blocks.{l}.attn.qkv.weight"]   # [3E, E]
        q, sc = quantize_to_int8(qkv_w)
        attn_qkv_scales.append(sc)
        vals  = ", ".join(str(int(v)) for v in q.flatten())
        lines.append(f"    /* layer {l} */ {{{vals}}},\n")
    lines.append("};\n")
    sc_vals = ", ".join(f"{s:.8f}f" for s in attn_qkv_scales)
    lines.append(f"const float g_attn_qkv_scale[_L] = {{{sc_vals}}};\n\n")

    # Attention proj [L][E*E]
    lines.append("/* ── Attention output projection ─────────────────────────────────────── */\n")
    lines.append(f"const int8_t g_attn_proj_weight[_L][_E * _E] = {{\n")
    attn_proj_scales = []
    for l in range(N_LAYER):
        p_w = sd[f"blocks.{l}.attn.proj.weight"]   # [E, E]
        q, sc = quantize_to_int8(p_w)
        attn_proj_scales.append(sc)
        vals  = ", ".join(str(int(v)) for v in q.flatten())
        lines.append(f"    /* layer {l} */ {{{vals}}},\n")
    lines.append("};\n")
    sc_vals = ", ".join(f"{s:.8f}f" for s in attn_proj_scales)
    lines.append(f"const float g_attn_proj_scale[_L] = {{{sc_vals}}};\n\n")

    # FFN FC [L][F*E]
    lines.append("/* ── FFN FC layer ────────────────────────────────────────────────────── */\n")
    lines.append(f"const int8_t g_ffn_fc_weight[_L][_F * _E] = {{\n")
    ffn_fc_scales = []
    for l in range(N_LAYER):
        fc_w = sd[f"blocks.{l}.ffn.fc.weight"]   # [F, E]
        q, sc = quantize_to_int8(fc_w)
        ffn_fc_scales.append(sc)
        vals  = ", ".join(str(int(v)) for v in q.flatten())
        lines.append(f"    /* layer {l} */ {{{vals}}},\n")
    lines.append("};\n")
    sc_vals = ", ".join(f"{s:.8f}f" for s in ffn_fc_scales)
    lines.append(f"const float g_ffn_fc_scale[_L] = {{{sc_vals}}};\n\n")

    # FFN proj [L][E*F]
    lines.append("/* ── FFN projection layer ───────────────────────────────────────────── */\n")
    lines.append(f"const int8_t g_ffn_proj_weight[_L][_E * _F] = {{\n")
    ffn_proj_scales = []
    for l in range(N_LAYER):
        pr_w = sd[f"blocks.{l}.ffn.proj.weight"]   # [E, F]
        q, sc = quantize_to_int8(pr_w)
        ffn_proj_scales.append(sc)
        vals  = ", ".join(str(int(v)) for v in q.flatten())
        lines.append(f"    /* layer {l} */ {{{vals}}},\n")
    lines.append("};\n")
    sc_vals = ", ".join(f"{s:.8f}f" for s in ffn_proj_scales)
    lines.append(f"const float g_ffn_proj_scale[_L] = {{{sc_vals}}};\n\n")

    # Final LayerNorm
    lines.append("/* ── Final RMSNorm ───────────────────────────────────────────────────── */\n")
    ln_f = sd["ln_f.weight"]
    vals = ", ".join(f"{v:.8f}f" for v in ln_f.flatten())
    lines.append(f"const float g_ln_f_weight[_E] = {{{vals}}};\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")

    total_int8 = (
        emb_q.size +
        N_LAYER * (3 * N_EMBD * N_EMBD + N_EMBD * N_EMBD +
                   FF_DIM * N_EMBD       + N_EMBD * FF_DIM)
    )
    total_fp32 = N_LAYER * 2 * N_EMBD + N_EMBD
    flash_kb   = (total_int8 + total_fp32 * 4) / 1024
    print(f"\nWeights exported to: {out_path}")
    print(f"  INT8 bytes : {total_int8:,}  ({total_int8 // 1024} KB)")
    print(f"  fp32 bytes : {total_fp32 * 4:,}  ({total_fp32 * 4 // 1024} KB)")
    print(f"  Total flash: ~{flash_kb:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global MAX_ITERS
    parser = argparse.ArgumentParser(description="Train TinyLLM and export weights.h")
    parser.add_argument("--corpus",      type=str,  default=None,
                        help="Path to a plain-text training corpus. "
                             "Defaults to downloading Shakespeare.")
    parser.add_argument("--checkpoint",  type=str,  default=None,
                        help="Path to a .pt checkpoint to resume from.")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training; only export --checkpoint to weights.h")
    parser.add_argument("--iters",       type=int,  default=MAX_ITERS,
                        help=f"Training iterations (default: {MAX_ITERS})")
    args = parser.parse_args()

    MAX_ITERS = args.iters

    if args.export_only:
        if not args.checkpoint:
            parser.error("--export-only requires --checkpoint")
        model = TinyLLM().to(DEVICE)
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        model.eval()
    else:
        # Load checkpoint if provided, then always train
        model = TinyLLM().to(DEVICE)
        if args.checkpoint and Path(args.checkpoint).exists():
            print(f"Resuming from checkpoint: {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        corpus = load_corpus(args.corpus)
        print(f"Corpus length: {len(corpus):,} characters")
        model = train(corpus, model)

    export_weights(model, OUTPUT_H)
    print("\nDone! Rebuild the ESP-IDF project to flash the new weights.")
    print("  cd llm && idf.py build && idf.py flash monitor")


if __name__ == "__main__":
    main()

