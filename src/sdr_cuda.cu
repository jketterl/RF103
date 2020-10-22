extern "C" {
#include "sdr_cuda.h"
}
#include <cuda_fp16.h>

__device__ float i_coeffs[] = { 0, 1, 0, -1};
__device__ float q_coeffs[] = { 1, 0, -1, 0};

__global__ void convert_ui16_c_kernel(short* input, float* output) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float converted = (float)input[i] / SHRT_MAX;
    output[i * 2] = converted * i_coeffs[i % 4];
    output[i * 2 + 1] = converted * q_coeffs[i % 4];
}

__global__ void fir_decimate_c_kernel(float* input, float* output, float* taps, uint16_t taps_length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int dec_i = 2 * i;
    float acci = 0;
    float accq = 0;
    for (int k = 0; k < taps_length; k++) {
        int index = dec_i - (taps_length - k);
        acci += input[index * 2] * taps[k];
        accq += input[index * 2 + 1] * taps[k];
    }
    output[i * 2] = acci;
    output[i * 2 + 1] = accq;
}

void ddc_init(ddc_t* filter, uint32_t buffersize, uint16_t decimation) {
    filter->buffersize = buffersize;
    filter->taps_length = 121;
    float* taps = (float*) malloc(sizeof(float) * filter->taps_length);
    firdes_lowpass_f(taps, filter->taps_length, 0.485/decimation, WINDOW_HAMMING);

    cudaMalloc((void**)&filter->taps, sizeof(float) * filter->taps_length);
    cudaMemcpy(filter->taps, taps, sizeof(float) * filter->taps_length, cudaMemcpyHostToDevice);

    free(taps);

    cudaMalloc((void**)&filter->raw, sizeof(short) * filter->buffersize);
    cudaMalloc((void**)&filter->input, sizeof(float) * (filter->buffersize + filter->taps_length * 2));
    cudaMalloc((void**)&filter->output, sizeof(float) * filter->buffersize);
}

void ddc_close(ddc_t* filter) {
    cudaFree(filter->raw);
    cudaFree(filter->input);
    cudaFree(filter->output);
    cudaFree(filter->taps);
}

float* get_fir_decimate_input(ddc_t* filter) {
    return filter->input + (filter->taps_length * 2);
}

extern "C"
void ddc(short* input, float* output, ddc_t* filter, uint32_t length) {
    cudaMemcpy(filter->raw, input, sizeof(short) * length, cudaMemcpyHostToDevice);

    // TODO compensate for incompatible alignment
    int blocks = length / 512;
    convert_ui16_c_kernel<<<blocks, 512>>>(filter->raw, get_fir_decimate_input(filter));
    blocks = length / 20;
    fir_decimate_c_kernel<<<blocks, 10>>>(filter->input, filter->output, filter->taps, filter->taps_length);
    // copy unprocessed samples from the end to the beginning of the input buffer
    cudaMemcpy(filter->input + (length * 2), filter->input, sizeof(float) * filter->taps_length * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(output, filter->output, sizeof(float) * length, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
}
