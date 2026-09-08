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
 * Read a newline-terminated line from stdin (blocking).
 * Returns number of characters read (excluding '\n' and '\0').
 * ──────────────────────────────────────────────────────────────────────────── */
static int read_line(char *buf, int max_len)
{
    int idx = 0;
    int c;

    while (idx < max_len) {
        c = getchar();

        if (c == EOF || c < 0) {
            /* No data yet — yield to FreeRTOS scheduler */
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (c == '\n' || c == '\r') {
            /* Echo newline so the terminal looks right */
            putchar('\n');
            fflush(stdout);
            break;
        }

        /* Echo the character back */
        putchar(c);
        fflush(stdout);

        buf[idx++] = (char)c;
    }

    buf[idx] = '\0';
    return idx;
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
    int   max_tokens  = TINYLLM_DEFAULT_MAX_TOKENS;

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

        tinyllm_generate(prompt, max_tokens, temperature);
    }
}
