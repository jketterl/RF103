#include <stdint.h>
#include <stdio.h>
#include "fir.h"

typedef struct {
    short* raw;
    float* input;
    float* output;
    uint32_t buffersize;
    float* taps;
    uint16_t taps_length;
} fir_decimate_t;

void convert_ui16_c(short* input, float* output, fir_decimate_t* filter, uint32_t length);

void fir_decimate_c_init(fir_decimate_t* filter, uint32_t buffersize);
void fir_decimate_c_close(fir_decimate_t* filter);
void fir_decimate_c(fir_decimate_t* filter, float* input, float* output);
float* get_fir_decimate_input(fir_decimate_t* filter);