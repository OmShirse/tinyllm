/**
 * @file tinyllm.c
 * @brief TinyLLM inference engine — decoder-only character-level transformer.
 *
 * Architecture: 2-layer, 64-dim, 4-head, 32-context, 95-vocab (ASCII).
 * Weights live in Flash (RODATA, INT8 quantized).
 * Activations and KV-cache live in SRAM (static allocation, ~66 KB total).
 *
 * See weights.h for the model weight layout.
 * See tokenizer.h for the character encoding.
 */

#include "tinyllm.h"
#include "weights.h"
#include "tokenizer.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "tinyllm";

/* ════════════════════════════════════════════════════════════════════════════
 * Static activation buffers — all in SRAM, never moved.
 * Total: ~66 KB
 * ════════════════════════════════════════════════════════════════════════════ */

/** Residual stream for each position in the current sequence. */
static float s_x[TINYLLM_BLOCK_SIZE][TINYLLM_N_EMBD];

/** General-purpose scratch buffers (one token row each). */
static float s_xb[TINYLLM_N_EMBD];
static float s_xb2[TINYLLM_N_EMBD];

/** Per-head attention score scratch (s_att[h][t]). */
static float s_att[TINYLLM_N_HEAD][TINYLLM_BLOCK_SIZE];

/**
 * KV-cache: stores key/value vectors for every layer, position, head, and
 * head dimension.  16 KB each → 32 KB total for 2 layers.
 */
static float s_k_cache[TINYLLM_N_LAYER][TINYLLM_BLOCK_SIZE]
                       [TINYLLM_N_HEAD][TINYLLM_HEAD_SIZE];
static float s_v_cache[TINYLLM_N_LAYER][TINYLLM_BLOCK_SIZE]
                       [TINYLLM_N_HEAD][TINYLLM_HEAD_SIZE];

/** FFN hidden-layer buffer (256 floats = 1 KB). */
static float s_ffn[TINYLLM_FF_DIM];

/** Output logits (95 floats ≈ 380 B). */
static float s_logits[TINYLLM_VOCAB_SIZE];

/* ════════════════════════════════════════════════════════════════════════════
 * Math primitives
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief RMS Layer Normalization.
 * out[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps)
 */
static void rmsnorm(float *__restrict__ out,
                    const float *__restrict__ x,
                    const float *__restrict__ weight,
                    int size)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)size + 1e-5f);
    for (int i = 0; i < size; i++) out[i] = weight[i] * (ss * x[i]);
}

/**
 * @brief INT8 matrix-vector multiply: out[rows] = W[rows][cols] · x[cols] * scale
 *
 * W is stored as int8 in Flash (RODATA). Accumulation in float32.
 * Scale is a per-tensor float32 dequantisation factor.
 */
static void matmul_q8(float *__restrict__ out,
                      const float *__restrict__ x,
                      const int8_t *__restrict__ W,
                      float scale,
                      int rows,
                      int cols)
{
    for (int i = 0; i < rows; i++) {
        float acc = 0.0f;
        const int8_t *row = W + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            acc += (float)row[j] * x[j];
        }
        out[i] = acc * scale;
    }
}

/**
 * @brief In-place softmax over the first `size` elements of x[].
 */
static void softmax(float *x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum  += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

/**
 * @brief GELU activation (tanh approximation).
 */
static inline float gelu(float v)
{
    const float c = 0.7978845608028654f;  /* sqrt(2/pi) */
    return 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
}

/**
 * @brief Sample a token index from a logit array with temperature scaling.
 *
 * @param logits   Raw logit vector (modified in place).
 * @param size     Vocabulary size.
 * @param temperature  0.0 → greedy argmax; > 0 → temperature sampling.
 * @return Sampled token index.
 */
static int sample_token(float *logits, int size, float temperature)
{
    if (temperature <= 0.0f) {
        /* Greedy: return argmax */
        int best = 0;
        for (int i = 1; i < size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* Temperature scaling then softmax */
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < size; i++) logits[i] *= inv_temp;
    softmax(logits, size);

    /* Multinomial sample */
    float r   = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cdf = 0.0f;
    for (int i = 0; i < size; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return size - 1;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Transformer forward pass
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Run the transformer for a single token at sequence position `pos`.
 *
 * After this call, s_logits[] holds the un-normalised log-probabilities for
 * the *next* token.  The KV-cache is updated in place.
 *
 * @param token  Token ID [0, VOCAB_SIZE).
 * @param pos    Current position in the sequence [0, BLOCK_SIZE).
 */
static void transformer_forward(int token, int pos)
{
    /* ── Token embedding lookup ─────────────────────────────────────────── */
    {
        const int8_t *emb = g_token_emb + (size_t)token * TINYLLM_N_EMBD;
        for (int i = 0; i < TINYLLM_N_EMBD; i++)
            s_x[pos][i] = (float)emb[i] * g_token_emb_scale;
    }

    /* ── Transformer layers ─────────────────────────────────────────────── */
    for (int l = 0; l < TINYLLM_N_LAYER; l++) {

        /* ── Attention sub-layer ────────────────────────────────────────── */

        /* Pre-norm */
        rmsnorm(s_xb, s_x[pos], g_ln1_weight[l], TINYLLM_N_EMBD);

        /* QKV projection: s_xb → [q | k | v], each of size N_EMBD */
        float qkv[3 * TINYLLM_N_EMBD];
        matmul_q8(qkv, s_xb,
                  g_attn_qkv_weight[l], g_attn_qkv_scale[l],
                  3 * TINYLLM_N_EMBD, TINYLLM_N_EMBD);

        const float *q_vec = qkv;
        const float *k_vec = qkv + TINYLLM_N_EMBD;
        const float *v_vec = qkv + 2 * TINYLLM_N_EMBD;

        /* Store k, v into cache */
        for (int h = 0; h < TINYLLM_N_HEAD; h++) {
            const float *kh = k_vec + h * TINYLLM_HEAD_SIZE;
            const float *vh = v_vec + h * TINYLLM_HEAD_SIZE;
            for (int d = 0; d < TINYLLM_HEAD_SIZE; d++) {
                s_k_cache[l][pos][h][d] = kh[d];
                s_v_cache[l][pos][h][d] = vh[d];
            }
        }

        /* Multi-head attention → s_xb */
        memset(s_xb, 0, sizeof(float) * TINYLLM_N_EMBD);
        const float attn_scale = 1.0f / sqrtf((float)TINYLLM_HEAD_SIZE);

        for (int h = 0; h < TINYLLM_N_HEAD; h++) {
            const float *qh = q_vec + h * TINYLLM_HEAD_SIZE;

            /* Dot products with all cached keys (causal: 0..pos) */
            for (int t = 0; t <= pos; t++) {
                float dot = 0.0f;
                for (int d = 0; d < TINYLLM_HEAD_SIZE; d++)
                    dot += qh[d] * s_k_cache[l][t][h][d];
                s_att[h][t] = dot * attn_scale;
            }

            /* Softmax over positions 0..pos */
            float max_a = s_att[h][0];
            for (int t = 1; t <= pos; t++)
                if (s_att[h][t] > max_a) max_a = s_att[h][t];
            float sum_a = 0.0f;
            for (int t = 0; t <= pos; t++) {
                s_att[h][t] = expf(s_att[h][t] - max_a);
                sum_a += s_att[h][t];
            }
            float inv_sum = 1.0f / sum_a;
            for (int t = 0; t <= pos; t++) s_att[h][t] *= inv_sum;

            /* Weighted sum of values → output for this head */
            float *out_h = s_xb + h * TINYLLM_HEAD_SIZE;
            for (int t = 0; t <= pos; t++) {
                float a = s_att[h][t];
                for (int d = 0; d < TINYLLM_HEAD_SIZE; d++)
                    out_h[d] += a * s_v_cache[l][t][h][d];
            }
        }

        /* Output projection */
        matmul_q8(s_xb2, s_xb,
                  g_attn_proj_weight[l], g_attn_proj_scale[l],
                  TINYLLM_N_EMBD, TINYLLM_N_EMBD);

        /* Residual */
        for (int i = 0; i < TINYLLM_N_EMBD; i++) s_x[pos][i] += s_xb2[i];

        /* ── FFN sub-layer ──────────────────────────────────────────────── */

        /* Pre-norm */
        rmsnorm(s_xb, s_x[pos], g_ln2_weight[l], TINYLLM_N_EMBD);

        /* FC: N_EMBD → FF_DIM */
        matmul_q8(s_ffn, s_xb,
                  g_ffn_fc_weight[l], g_ffn_fc_scale[l],
                  TINYLLM_FF_DIM, TINYLLM_N_EMBD);

        /* GELU activation */
        for (int i = 0; i < TINYLLM_FF_DIM; i++) s_ffn[i] = gelu(s_ffn[i]);

        /* Projection: FF_DIM → N_EMBD */
        matmul_q8(s_xb2, s_ffn,
                  g_ffn_proj_weight[l], g_ffn_proj_scale[l],
                  TINYLLM_N_EMBD, TINYLLM_FF_DIM);

        /* Residual */
        for (int i = 0; i < TINYLLM_N_EMBD; i++) s_x[pos][i] += s_xb2[i];
    }

    /* ── Final RMSNorm ──────────────────────────────────────────────────── */
    rmsnorm(s_xb, s_x[pos], g_ln_f_weight, TINYLLM_N_EMBD);

    /* ── LM head (weight-tied with token embedding) ─────────────────────── */
    matmul_q8(s_logits, s_xb,
              g_token_emb, g_lm_head_scale,
              TINYLLM_VOCAB_SIZE, TINYLLM_N_EMBD);
}

/* ════════════════════════════════════════════════════════════════════════════
 * Public API
 * ════════════════════════════════════════════════════════════════════════════ */

void tinyllm_init(void)
{
    /* Seed RNG from hardware microsecond timer for variety between resets */
    srand((unsigned int)(esp_timer_get_time() & 0xFFFFFFFFUL));

    ESP_LOGI(TAG, "┌─────────────────────────────────────────┐");
    ESP_LOGI(TAG, "│           TinyLLM on ESP32              │");
    ESP_LOGI(TAG, "├─────────────────────────────────────────┤");
    ESP_LOGI(TAG, "│ vocab=%-4d  embd=%-3d  heads=%-2d  layers=%-2d│",
             TINYLLM_VOCAB_SIZE, TINYLLM_N_EMBD,
             TINYLLM_N_HEAD, TINYLLM_N_LAYER);
    ESP_LOGI(TAG, "│ context=%-3d  ff_dim=%-3d                 │",
             TINYLLM_BLOCK_SIZE, TINYLLM_FF_DIM);
    ESP_LOGI(TAG, "└─────────────────────────────────────────┘");

    /* Flash weight size (informational) */
    size_t flash_bytes =
        sizeof(g_token_emb) +
        sizeof(g_ln1_weight) + sizeof(g_ln2_weight) +
        sizeof(g_attn_qkv_weight) + sizeof(g_attn_proj_weight) +
        sizeof(g_ffn_fc_weight)   + sizeof(g_ffn_proj_weight) +
        sizeof(g_ln_f_weight);
    ESP_LOGI(TAG, "Weight flash usage: %u KB", (unsigned)(flash_bytes / 1024));
    (void)flash_bytes;  /* silence host-gcc when ESP_LOGI is a no-op */

    /* SRAM usage (static buffers) */
    size_t sram_bytes =
        sizeof(s_x) + sizeof(s_xb) + sizeof(s_xb2) +
        sizeof(s_att) +
        sizeof(s_k_cache) + sizeof(s_v_cache) +
        sizeof(s_ffn) + sizeof(s_logits);
    ESP_LOGI(TAG, "Activation SRAM usage: %u KB", (unsigned)(sram_bytes / 1024));
    (void)sram_bytes;
}

void tinyllm_generate(const char *prompt, int max_new_tokens, float temperature)
{
    /* ── Tokenise prompt ───────────────────────────────────────────────── */
    int tokens[TINYLLM_BLOCK_SIZE];
    int n_prompt = 0;

    for (int i = 0; prompt[i] != '\0' && n_prompt < TINYLLM_BLOCK_SIZE - 1; i++) {
        int t = tinyllm_encode(prompt[i]);
        if (t >= 0) tokens[n_prompt++] = t;
    }
    if (n_prompt == 0) {
        tokens[n_prompt++] = 0;   /* fallback: space token */
    }

    /* ── Reset KV-cache ────────────────────────────────────────────────── */
    memset(s_k_cache, 0, sizeof(s_k_cache));
    memset(s_v_cache, 0, sizeof(s_v_cache));

    int64_t t_start = esp_timer_get_time();

    /* ── Pre-fill: process prompt tokens ──────────────────────────────── */
    int pos = 0;
    for (; pos < n_prompt; pos++) {
        transformer_forward(tokens[pos], pos);
    }

    /* ── Generation loop ──────────────────────────────────────────────── */
    int n_generated = 0;
    while (n_generated < max_new_tokens && pos < TINYLLM_BLOCK_SIZE) {
        /* Copy logits (sample_token modifies them in place) */
        float probs[TINYLLM_VOCAB_SIZE];
        memcpy(probs, s_logits, sizeof(s_logits));

        int next_token = sample_token(probs, TINYLLM_VOCAB_SIZE, temperature);
        char c = tinyllm_decode(next_token);

        putchar(c);
        fflush(stdout);

        n_generated++;

        /* Forward for next step */
        transformer_forward(next_token, pos);
        pos++;
    }

    /* ── Stats ─────────────────────────────────────────────────────────── */
    float elapsed_s  = (float)(esp_timer_get_time() - t_start) / 1e6f;
    float tok_per_s  = (elapsed_s > 0.0f) ? (float)n_generated / elapsed_s : 0.0f;
    printf("\n[%d tokens | %.2f tok/s]\n", n_generated, tok_per_s);
}
