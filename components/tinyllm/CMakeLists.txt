idf_component_register(
    SRCS "tinyllm.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_timer
    PRIV_REQUIRES newlib
)

# Keep weights in Flash (rodata), never copy to SRAM
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffast-math)

