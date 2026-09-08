#pragma once

/**
 * @file tokenizer.h
 * @brief Character-level tokenizer for TinyLLM.
 *
 * Vocabulary: printable ASCII 0x20 (space) → 0x7E (~)  ← 95 tokens
 * Token ID  = char_code - 0x20
 *
 * Special tokens (not stored in flash vocab, handled in code):
 *   TOKEN_BOS = 95   (beginning-of-sequence)
 *   TOKEN_EOS = 96   (end-of-sequence)
 */

#include <stdint.h>

#define TINYLLM_CHAR_OFFSET  0x20   /* First printable ASCII */
#define TOKEN_BOS            95
#define TOKEN_EOS            96

/**
 * @brief Encode a single character to a token ID.
 * @return Token ID [0, 94] or -1 if the character is not in vocabulary.
 */
static inline int tinyllm_encode(char c)
{
    int code = (unsigned char)c;
    if (code >= 0x20 && code <= 0x7E)
        return code - TINYLLM_CHAR_OFFSET;
    return -1;
}

/**
 * @brief Decode a token ID back to its character.
 * @return Printable character, or '?' for out-of-range tokens.
 */
static inline char tinyllm_decode(int token)
{
    if (token >= 0 && token < 95)
        return (char)(token + TINYLLM_CHAR_OFFSET);
    return '?';
}

