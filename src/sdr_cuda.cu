extern "C" {
#include "sdr_cuda.h"
}
#include <cuda_fp16.h>

__device__ float i_coeffs[] = { 0, 1, 0, -1};
__device__ float q_coeffs[] = { 1, 0, -1, 0};

__global__ void convert_ui16_c_kernel(short* input, float* output, float* phase_offset, float* angle_per_sample) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float converted = (float)input[i] / SHRT_MAX;
    float angle = *phase_offset + *angle_per_sample * i;
    output[i * 2] = sinpif(angle) * converted;
    output[i * 2 + 1] = cospif(angle) * converted;
}

__global__ void fir_decimate_c_kernel(float* input, float* output, uint16_t* decimation, float* taps, uint16_t taps_length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int dec_i = i * *decimation;
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

void ddc_init(ddc_t* filter, uint32_t buffersize, float freq_offset, uint16_t decimation) {
    filter->buffersize = buffersize;
    filter->taps_length = 501;
    float* taps = (float*) malloc(sizeof(float) * filter->taps_length);
    firdes_lowpass_f(taps, filter->taps_length, 0.5/decimation, WINDOW_HAMMING);

    cudaMalloc((void**)&filter->taps, sizeof(float) * filter->taps_length);
    cudaMemcpy(filter->taps, taps, sizeof(float) * filter->taps_length, cudaMemcpyHostToDevice);

    free(taps);

    filter->decimation = decimation;
    cudaMalloc((void**)&filter->decimation_device, sizeof(uint16_t));
    cudaMemcpy(filter->decimation_device, &filter->decimation, sizeof(uint16_t), cudaMemcpyHostToDevice);

    filter->phase_offset = 0;
    cudaMalloc((void**)&filter->phase_offset_device, sizeof(float));
    cudaMemcpy(filter->phase_offset_device, &filter->phase_offset, sizeof(float), cudaMemcpyHostToDevice);

    filter->angle_per_sample = 2 * freq_offset;
    cudaMalloc((void**)&filter->angle_per_sample_device, sizeof(float));
    cudaMemcpy(filter->angle_per_sample_device, &filter->angle_per_sample, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&filter->raw, sizeof(short) * filter->buffersize);
    cudaMalloc((void**)&filter->input, sizeof(float) * (filter->buffersize + filter->taps_length * 2));
    cudaMalloc((void**)&filter->output, sizeof(float) * filter->buffersize);
}

void ddc_close(ddc_t* filter) {
    cudaFree(filter->decimation_device);
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
int ddc(short* input, float* output, ddc_t* filter, uint32_t length) {
    cudaMemcpy(filter->raw, input, sizeof(short) * length, cudaMemcpyHostToDevice);

    // TODO compensate for incompatible alignment
    int blocks = length / 1024;
    convert_ui16_c_kernel<<<blocks, 1024>>>(filter->raw, get_fir_decimate_input(filter), filter->phase_offset_device, filter->angle_per_sample_device);
    filter->phase_offset += filter->angle_per_sample * length;
    while (filter->phase_offset > 2) filter->phase_offset -= 2;
    cudaMemcpy(filter->phase_offset_device, &filter->phase_offset, sizeof(float), cudaMemcpyHostToDevice);
    int threads = 1024 / filter->decimation;
    fir_decimate_c_kernel<<<blocks, threads>>>(filter->input, filter->output, filter->decimation_device, filter->taps, filter->taps_length);
    // copy unprocessed samples from the end to the beginning of the input buffer
    cudaMemcpy(filter->input + (length * 2), filter->input, sizeof(float) * filter->taps_length * 2, cudaMemcpyDeviceToDevice);
    uint32_t out_samples = length / filter->decimation;
    cudaMemcpy(output, filter->output, sizeof(float) * (out_samples * 2), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

    return out_samples;
}
