extern "C" {
#include "sdr_cuda.h"
}
#include <cuda_fp16.h>

__device__ float i_coeffs[] = { 0, 1, 0, -1};
__device__ float q_coeffs[] = { 1, 0, -1, 0};

__global__ void convert_ui16_c_kernel(int16_t* input, float* output, double* phase_offset, double* angle_per_sample) {
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    float converted = (float)input[i] / SHRT_MAX;
    double angle = *phase_offset + *angle_per_sample * i;
    output[i * 2] = sinpi(angle) * converted;
    output[i * 2 + 1] = cospi(angle) * converted;
}

__global__ void fir_decimate_c_kernel(float* input, float* output, uint16_t* decimation, uint16_t* decimation_offset, float* taps, uint16_t taps_length) {
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t dec_i = *decimation_offset + i * *decimation;
    float acci = 0;
    float accq = 0;
    for (uint16_t k = 0; k < taps_length; k++) {
        int32_t index = dec_i - (taps_length - k);
        acci += input[index * 2] * taps[k];
        accq += input[index * 2 + 1] * taps[k];
    }
    output[i * 2] = acci;
    output[i * 2 + 1] = accq;
}

void ddc_init(ddc_t* filter, uint32_t buffersize, float freq_offset, uint16_t decimation) {
    filter->taps_length = 4 * decimation;
    if (filter->taps_length %2 == 0) filter->taps_length++;
    filter->taps_length = max(filter->taps_length, 121);
    fprintf(stderr, "taps length: %i\n", filter->taps_length);

    filter->buffersize = buffersize;
    float* taps = (float*) malloc(sizeof(float) * filter->taps_length);
    firdes_lowpass_f(taps, filter->taps_length, 0.485/decimation, WINDOW_HAMMING);

    cudaMalloc((void**)&filter->taps, sizeof(float) * filter->taps_length);
    cudaMemcpy(filter->taps, taps, sizeof(float) * filter->taps_length, cudaMemcpyHostToDevice);

    free(taps);

    filter->decimation = decimation;
    cudaMalloc((void**)&filter->decimation_device, sizeof(uint16_t));
    cudaMemcpy(filter->decimation_device, &filter->decimation, sizeof(uint16_t), cudaMemcpyHostToDevice);

    filter->decimation_offset = 0;
    cudaMalloc((void**)&filter->decimation_offset_device, sizeof(uint16_t));
    cudaMemcpy(filter->decimation_offset_device, &filter->decimation, sizeof(uint16_t), cudaMemcpyHostToDevice);

    filter->phase_offset = 0;
    cudaMalloc((void**)&filter->phase_offset_device, sizeof(double));
    cudaMemcpy(filter->phase_offset_device, &filter->phase_offset, sizeof(double), cudaMemcpyHostToDevice);

    filter->angle_per_sample = 2 * freq_offset;
    cudaMalloc((void**)&filter->angle_per_sample_device, sizeof(double));
    cudaMemcpy(filter->angle_per_sample_device, &filter->angle_per_sample, sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&filter->raw, sizeof(int16_t) * filter->buffersize);
    cudaMalloc((void**)&filter->input, sizeof(float) * 2 * (filter->buffersize + filter->taps_length));
    cudaMalloc((void**)&filter->output, sizeof(float) * filter->buffersize);
}

void ddc_close(ddc_t* filter) {
    cudaFree(filter->decimation_device);
    cudaFree(filter->decimation_offset_device);
    cudaFree(filter->phase_offset_device);
    cudaFree(filter->angle_per_sample_device);
    cudaFree(filter->raw);
    cudaFree(filter->input);
    cudaFree(filter->output);
    cudaFree(filter->taps);
}

float* get_fir_decimate_input(ddc_t* filter) {
    return filter->input + (filter->taps_length * 2);
}

extern "C"
uint32_t ddc(int16_t* input, float* output, ddc_t* filter, uint32_t length) {
    cudaMemcpy(filter->raw, input, sizeof(int16_t) * length, cudaMemcpyHostToDevice);

    int blocks = length / 1024;
    // run an extra block if memory does not line up ideally
    if (blocks % 1024 > 0) blocks += 1;
    convert_ui16_c_kernel<<<blocks, 1024>>>(filter->raw, get_fir_decimate_input(filter), filter->phase_offset_device, filter->angle_per_sample_device);

    // move the phase forward
    filter->phase_offset += filter->angle_per_sample * length;
    while (filter->phase_offset > 2) filter->phase_offset -= 2;
    cudaMemcpy(filter->phase_offset_device, &filter->phase_offset, sizeof(double), cudaMemcpyHostToDevice);

    uint32_t out_samples = length / filter->decimation;
    blocks = out_samples / 512;
    // run an extra block if memory does not line up ideally
    if (out_samples % 512) blocks += 1;
    fir_decimate_c_kernel<<<blocks, 512>>>(get_fir_decimate_input(filter), filter->output, filter->decimation_device, filter->decimation_offset_device, filter->taps, filter->taps_length);

    // update decimation offset
    filter->decimation_offset = (filter->decimation_offset + length) % filter->decimation;
    cudaMemcpy(filter->decimation_offset_device, &filter->decimation_offset, sizeof(uint16_t), cudaMemcpyHostToDevice);

    // copy unprocessed samples from the end to the beginning of the input buffer
    cudaMemcpy(filter->input, filter->input + (length * 2), sizeof(float) * filter->taps_length * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(output, filter->output, sizeof(float) * (out_samples * 2), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

    return out_samples;
}
