#pragma once

/**
 * @file tinyllm.h
 * @brief TinyLLM — Tiny Decoder-Only Transformer for ESP32 DevKit v1
 *
 * Architecture (character-level, ASCII 0x20-0x7E):
 *   vocab_size  = 95
 *   n_embd      = 64
 *   n_head      = 4   (head_size = 16)
 *   n_layer     = 2
 *   block_size  = 32  (context window)
 *   ff_dim      = 256 (4 × n_embd)
 *
 * Memory budget:
 *   Flash (weights, INT8 + float scales): ~103 KB
 *   SRAM  (activations, KV-cache):        ~66  KB
 */

#include <stdint.h>

/* ── Hyperparameters ──────────────────────────────────────────────────────── */
#define TINYLLM_VOCAB_SIZE   95
#define TINYLLM_N_EMBD       64
#define TINYLLM_N_HEAD       4
#define TINYLLM_HEAD_SIZE    (TINYLLM_N_EMBD / TINYLLM_N_HEAD)   /* 16 */
#define TINYLLM_N_LAYER      2
#define TINYLLM_BLOCK_SIZE   32
#define TINYLLM_FF_DIM       (TINYLLM_N_EMBD * 4)               /* 256 */

/* ── Default generation parameters ───────────────────────────────────────── */
#define TINYLLM_DEFAULT_MAX_TOKENS  200
#define TINYLLM_DEFAULT_TEMPERATURE 0.8f

/* ── Public API ───────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the TinyLLM engine (seeds PRNG, logs model info).
 *        Call once from app_main().
 */
void tinyllm_init(void);

/**
 * @brief Generate text from a prompt and stream characters to stdout (UART0).
 *
 * @param prompt         NUL-terminated prompt string (printable ASCII).
 * @param max_new_tokens Maximum number of tokens to generate.
 * @param temperature    Sampling temperature. 0.0 → greedy; 0.8 → creative.
 */
void tinyllm_generate(const char *prompt, int max_new_tokens, float temperature);

