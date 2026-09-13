/**
 * @file llm.c
 * @brief TinyLLM — Main application entry point.
 *
 * Initialises UART console, prints a boot banner, then enters an interactive
 * prompt loop:
 *   1. Prints "> " prompt to the serial monitor (115200 baud, UART0).
 *   2. Reads a line of user input (up to TINYLLM_BLOCK_SIZE-1 chars).
 *   3. Passes the input to tinyllm_generate() which streams tokens back.
 *   4. Repeats.
 *
 * Flash: USB-serial adapter → /dev/ttyUSB0 (Linux) or COM port (Windows)
 * Monitor: idf.py flash monitor  OR  screen /dev/ttyUSB0 115200
 */

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"

#include "tinyllm.h"

static const char *TAG = "main";

/* Maximum prompt length (leave room for NUL terminator) */
#define MAX_PROMPT_LEN  (TINYLLM_BLOCK_SIZE - 1)

/* ────────────────────────────────────────────────────────────────────────────
 * Input reader state — tracks ANSI escape sequence parsing.
 * ──────────────────────────────────────────────────────────────────────────── */
typedef enum {
    INPUT_NORMAL,       /* regular character */
    INPUT_ESC,          /* received ESC (0x1B), waiting for '[' or 'O' */
    INPUT_ESC_BRACKET,  /* received ESC '[', waiting for final byte */
} input_state_t;

/* ────────────────────────────────────────────────────────────────────────────
 * Read a newline-terminated line from stdin with full editing support:
 *   Backspace / DEL  — erase previous character
 *   Ctrl+U  (0x15)  — erase entire line
 *   Ctrl+C  (0x03)  — cancel current input, return empty string
 *   Arrow keys / F-keys (ANSI ESC sequences) — silently discarded
 *
 * Returns number of characters in buf (excluding NUL).
 * ──────────────────────────────────────────────────────────────────────────── */
static int read_line(char *buf, int max_len)
{
    int idx = 0;
    int c;
    input_state_t state = INPUT_NORMAL;

    while (1) {
        c = getchar();

        if (c == EOF || c < 0) {
            /* No data yet — yield to FreeRTOS scheduler */
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        /* ── ANSI escape sequence state machine ────────────────────────── */
        if (state == INPUT_ESC_BRACKET) {
            /* Final byte of a CSI sequence (e.g. 'A'=Up, 'B'=Down, 'C'=Right,
             * 'D'=Left). Silently consume and return to normal. */
            state = INPUT_NORMAL;
            continue;
        }

        if (state == INPUT_ESC) {
            if (c == '[' || c == 'O') {
                /* CSI or SS3 — one more byte to consume */
                state = INPUT_ESC_BRACKET;
            } else {
                /* Lone ESC or other sequence — ignore */
                state = INPUT_NORMAL;
            }
            continue;
        }

        /* ── Normal character processing ───────────────────────────────── */
        switch (c) {

        case '\x1b':  /* ESC — start of ANSI sequence */
            state = INPUT_ESC;
            break;

        case '\b':    /* Backspace (Ctrl+H) */
        case 0x7F:    /* DEL — sent by most terminals as Backspace */
            if (idx > 0) {
                idx--;
                /* Erase the character on the terminal: back, space, back */
                printf("\b \b");
                fflush(stdout);
            }
            break;

        case 0x15:    /* Ctrl+U — kill entire line */
            while (idx > 0) {
                printf("\b \b");
                idx--;
            }
            fflush(stdout);
            break;

        case 0x03:    /* Ctrl+C — cancel input */
            printf("^C\n");
            fflush(stdout);
            buf[0] = '\0';
            return 0;

        case '\n':
        case '\r':    /* Enter — end of line */
            putchar('\n');
            fflush(stdout);
            buf[idx] = '\0';
            return idx;

        default:
            /* Accept only printable ASCII (matches tokenizer vocab 0x20-0x7E) */
            if (c >= 0x20 && c <= 0x7E && idx < max_len) {
                buf[idx++] = (char)c;
                putchar(c);
                fflush(stdout);
            }
            break;
        }
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * app_main — FreeRTOS entry point
 * ──────────────────────────────────────────────────────────────────────────── */
void app_main(void)
{
    ESP_LOGI(TAG, "═══════════════════════════════════════════");
    ESP_LOGI(TAG, "   TinyLLM · ESP32 DevKit v1 · ESP-IDF    ");
    ESP_LOGI(TAG, "═══════════════════════════════════════════");

    /* Initialise the inference engine */
    tinyllm_init();

    /* Print usage instructions */
    printf("\n");
    printf("TinyLLM ready. Type a prompt and press Enter.\n");
    printf("Commands: :quit  :reset  :temp <0.0-2.0>  :len <1-32>\n");
    printf("\n");

    char prompt[MAX_PROMPT_LEN + 1];
    float temperature = TINYLLM_DEFAULT_TEMPERATURE;
    int   max_tokens  = 100;   /* enough for longest IoT response */

    while (1) {
        /* Print prompt */
        printf("> ");
        fflush(stdout);

        int len = read_line(prompt, MAX_PROMPT_LEN);

        if (len == 0) {
            /* Empty line — just re-prompt */
            continue;
        }

        /* ── Built-in commands ─────────────────────────────────────────── */
        if (strcmp(prompt, ":quit") == 0) {
            printf("Goodbye.\n");
            esp_restart();
        }

        if (strcmp(prompt, ":reset") == 0) {
            printf("Restarting...\n");
            esp_restart();
        }

        float new_temp;
        if (sscanf(prompt, ":temp %f", &new_temp) == 1) {
            if (new_temp >= 0.0f && new_temp <= 2.0f) {
                temperature = new_temp;
                printf("Temperature set to %.2f\n", temperature);
            } else {
                printf("Temperature must be 0.0–2.0\n");
            }
            continue;
        }

        int new_len;
        if (sscanf(prompt, ":len %d", &new_len) == 1) {
            if (new_len >= 1 && new_len <= TINYLLM_BLOCK_SIZE) {
                max_tokens = new_len;
                printf("Max tokens set to %d\n", max_tokens);
            } else {
                printf("Length must be 1–%d\n", TINYLLM_BLOCK_SIZE);
            }
            continue;
        }

        /* ── Generate ──────────────────────────────────────────────────── */
        ESP_LOGI(TAG, "Prompt: \"%s\" | temp=%.2f | max_tokens=%d",
                 prompt, temperature, max_tokens);

        /* Append '\n' so the model sees the same prompt/response separator
         * it was trained on. Generation stops at the next '\n' (token 95). */
        char prompt_nl[MAX_PROMPT_LEN + 2];
        snprintf(prompt_nl, sizeof(prompt_nl), "%s\n", prompt);
        tinyllm_generate(prompt_nl, max_tokens, temperature);
    }
    }
