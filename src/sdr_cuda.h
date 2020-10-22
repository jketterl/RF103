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
} ddc_t;

void ddc(short* input, float* output, ddc_t* filter, uint32_t length);

void ddc_init(ddc_t* filter, uint32_t buffersize, uint16_t decimation);
void ddc_close(ddc_t* filter);