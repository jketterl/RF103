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

__global__ void fir_filer(float* input, float* output, float* taps, uint32_t taps_length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
}

__global__ void fir_decimate_c_kernel(float* input, float* output, float* taps, uint16_t taps_length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //output[i * 2] = input[i * 2];
    //output[i * 2 + 1] = input[i * 2 + 1];
    float acci = 0;
    float accq = 0;
    for (int k = 0; k < taps_length; k++) {
        int index = i - (taps_length - k);
        acci += input[index * 2] * taps[k];
        accq += input[index * 2 + 1] * taps[k];
    }
    output[i * 2] = acci;
    output[i * 2 + 1] = accq;
}

void fir_decimate_c_init(fir_decimate_t* filter, uint32_t buffersize) {
    filter->buffersize = buffersize;
    filter->taps_length = 121;
    float* taps = (float*) malloc(sizeof(float) * filter->taps_length);
    firdes_lowpass_f(taps, filter->taps_length, .25, WINDOW_HAMMING);

    cudaMalloc((void**)&filter->taps, sizeof(float) * filter->taps_length);
    cudaMemcpy(filter->taps, taps, sizeof(float) * filter->taps_length, cudaMemcpyHostToDevice);

    free(taps);

    cudaMalloc((void**)&filter->input, sizeof(float) * (filter->buffersize + filter->taps_length * 2));
    cudaMalloc((void**)&filter->output, sizeof(float) * filter->buffersize);
}

void fir_decimate_c_close(fir_decimate_t* filter) {
    cudaFree(filter->input);
    cudaFree(filter->output);
}

float* get_fir_decimate_input(fir_decimate_t* filter) {
    return filter->input + (filter->taps_length * 2);
}

extern "C"
void convert_ui16_c(short* input, float* output, fir_decimate_t* filter, uint32_t length) {
    short* device_input;
    cudaMalloc((void**)&device_input, sizeof(short) * length);
    cudaMemcpy(device_input, input, sizeof(short) * length, cudaMemcpyHostToDevice);

    // TODO compensate for incompatible alignment
    int blocks = length / 512;
    convert_ui16_c_kernel<<<blocks, 512>>>(device_input, get_fir_decimate_input(filter));
    blocks = length / 10;
    fir_decimate_c_kernel<<<blocks, 10>>>(filter->input, filter->output, filter->taps, filter->taps_length);
    cudaMemcpy(filter->input + (length * 2), filter->input, sizeof(float) * filter->taps_length * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(output, filter->output, sizeof(float) * length * 2, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
    cudaFree(device_input);
}
