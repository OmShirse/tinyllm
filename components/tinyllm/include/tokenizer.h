#pragma once

/**
 * @file tokenizer.h
 * @brief Character-level tokenizer for TinyLLM.
 *
 * Vocabulary (96 tokens):
 *   Token  0 – 94 : printable ASCII 0x20 (space) → 0x7E (~)
 *   Token 95       : newline '\n'  ← used as prompt/response separator
 *
 * Usage in firmware:
 *   - Append '\n' to every user prompt before calling tinyllm_generate()
 *   - tinyllm_generate() stops automatically when it produces token 95
 *   - This gives one clean response per prompt with no trailing newline echo
 */

#include <stdint.h>

#define TINYLLM_CHAR_OFFSET  0x20   /* First printable ASCII */
#define TOKEN_NEWLINE        95     /* '\n' — response separator / stop token */
#define TOKEN_BOS            96
#define TOKEN_EOS            97

/**
 * @brief Encode a single character to a token ID.
 * @return Token ID in [0, 95], or -1 if the character is not in vocabulary.
 */
static inline int tinyllm_encode(char c)
{
    int code = (unsigned char)c;
    if (code >= 0x20 && code <= 0x7E)
        return code - TINYLLM_CHAR_OFFSET;
    if (c == '\n')
        return TOKEN_NEWLINE;
    return -1;
}

/**
 * @brief Decode a token ID back to its character.
 * @return The character, '\n' for token 95, or '?' for out-of-range tokens.
 */
static inline char tinyllm_decode(int token)
{
    if (token >= 0 && token < 95)
        return (char)(token + TINYLLM_CHAR_OFFSET);
    if (token == TOKEN_NEWLINE)
        return '\n';
    return '?';
}
