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
    uint16_t decimation;
    uint16_t* decimation_device;
    double phase_offset;
    double* phase_offset_device;
    double angle_per_sample;
    double* angle_per_sample_device;
} ddc_t;

uint32_t ddc(short* input, float* output, ddc_t* filter, uint32_t length);

void ddc_init(ddc_t* filter, uint32_t buffersize, float freq_offset, uint16_t decimation);
void ddc_close(ddc_t* filter);