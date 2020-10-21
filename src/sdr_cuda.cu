#include <stdint.h>
#include <cuda_fp16.h>
#include <stdio.h>

__device__ float i_coeffs[] = { 0, 1, 0, -1};
__device__ float q_coeffs[] = { 1, 0, -1, 0};

__global__ void convert_ui16_c_kernel(short* input, float* output) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float converted = (float)input[i] / SHRT_MAX;
    output[i * 2] = converted * i_coeffs[i % 4];
    output[i * 2 + 1] = converted * q_coeffs[i % 4];
}

extern "C"
void convert_ui16_c(short* input, float* output, uint32_t length) {
    short* device_input;
    float* device_output;
    cudaMalloc((void**)&device_input, sizeof(short) * length);
    cudaMalloc((void**)&device_output, sizeof(float) * length * 2);
    cudaMemcpy(device_input, input, sizeof(short) * length, cudaMemcpyHostToDevice);

    // TODO compensate for incompatible alignment
    int blocks = length / 512;
    convert_ui16_c_kernel<<<blocks, 512>>>(device_input, device_output);
    cudaMemcpy(output, device_output, sizeof(float) * length * 2, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
    cudaFree(device_input);
    cudaFree(device_output);
}