#pragma once

/**
 * @file weights.h
 * @brief TinyLLM model weights stored in Flash (RODATA).
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  AUTO-GENERATED FILE — do NOT edit by hand.                            │
 * │  Run:  python3 tools/train_tinyllm.py   (trains on corpus)             │
 * │   OR:  python3 tools/export_weights.py  (exports existing checkpoint)  │
 * │  Then copy the generated weights.h here and rebuild.                   │
 * │                                                                         │
 * │  This file contains PLACEHOLDER zeros so the project compiles          │
 * │  before training. Replace with trained weights for useful generation.   │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Layout:
 *   g_token_emb      [VOCAB*EMBD]          int8   (also used as lm_head, weight tying)
 *   g_token_emb_scale                       float
 *   g_ln1_weight     [N_LAYER][EMBD]        float  (RMSNorm, kept as fp32 — tiny)
 *   g_ln2_weight     [N_LAYER][EMBD]        float
 *   g_attn_qkv_weight[N_LAYER][3*EMBD*EMBD] int8
 *   g_attn_qkv_scale [N_LAYER]              float
 *   g_attn_proj_weight[N_LAYER][EMBD*EMBD]  int8
 *   g_attn_proj_scale[N_LAYER]              float
 *   g_ffn_fc_weight  [N_LAYER][FF*EMBD]     int8
 *   g_ffn_fc_scale   [N_LAYER]              float
 *   g_ffn_proj_weight[N_LAYER][EMBD*FF]     int8
 *   g_ffn_proj_scale [N_LAYER]              float
 *   g_ln_f_weight    [EMBD]                 float
 *   g_lm_head_scale                         float
 */

#include <stdint.h>
#include "tinyllm.h"

/* ── Convenience dimension shorthands ──────────────────────────────────────── */
#define _V   TINYLLM_VOCAB_SIZE
#define _E   TINYLLM_N_EMBD
#define _L   TINYLLM_N_LAYER
#define _F   TINYLLM_FF_DIM

/* ── Token embedding / LM head (weight tying) ─────────────────────────────── */
const int8_t g_token_emb[_V * _E] = {0};
const float  g_token_emb_scale     = 0.01f;

/* ── Per-layer RMSNorm weights (fp32, ~512 B total) ───────────────────────── */
const float g_ln1_weight[_L][_E] = {{[0 ... _E-1] = 1.0f}, {[0 ... _E-1] = 1.0f}};
const float g_ln2_weight[_L][_E] = {{[0 ... _E-1] = 1.0f}, {[0 ... _E-1] = 1.0f}};

/* ── Attention: QKV projection  [L][3·E × E] ─────────────────────────────── */
const int8_t g_attn_qkv_weight[_L][3 * _E * _E] = {{0}, {0}};
const float  g_attn_qkv_scale[_L]                = {0.01f, 0.01f};

/* ── Attention: output projection [L][E × E] ─────────────────────────────── */
const int8_t g_attn_proj_weight[_L][_E * _E] = {{0}, {0}};
const float  g_attn_proj_scale[_L]            = {0.01f, 0.01f};

/* ── FFN: FC layer [L][F × E] ────────────────────────────────────────────── */
const int8_t g_ffn_fc_weight[_L][_F * _E] = {{0}, {0}};
const float  g_ffn_fc_scale[_L]            = {0.01f, 0.01f};

/* ── FFN: projection [L][E × F] ──────────────────────────────────────────── */
const int8_t g_ffn_proj_weight[_L][_E * _F] = {{0}, {0}};
const float  g_ffn_proj_scale[_L]            = {0.01f, 0.01f};

/* ── Final RMSNorm ────────────────────────────────────────────────────────── */
const float g_ln_f_weight[_E] = {[0 ... _E-1] = 1.0f};

/* ── LM head scale (token_emb is reused as weight matrix) ─────────────────── */
const float g_lm_head_scale = 0.01f;
