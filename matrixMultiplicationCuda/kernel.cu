
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void matrixMultiplicationKernel(const double* a, const double* b, double* c, const size_t size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    if (col < size && row < size)
    {
        for (int i = 0; i < size; i++)
        {
            sum += a[row * size + i] * b[i * size + col];
        }
        c[row * size + col] = sum;
    }
}

int main()
{
    const size_t size = 1000;


    double* a = new double[size * size];
    double* b = new double[size * size];
    double* c = new double[size * size];

    for (int i = 0; i < size * size; ++i)
    {
        a[i] = 5.0;
        b[i] = 5.0;
        c[i] = 0.0;
    }

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    const size_t sizeBytes = size * size * sizeof(double);
    double* aDevice;
    double* bDevice;
    double* cDevice;

    cudaStatus = cudaMalloc((void**)&aDevice, sizeBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

   

    cudaStatus = cudaMalloc((void**)&bDevice, sizeBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&cDevice, sizeBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMemcpy(aDevice, a, sizeBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 1;
    }

    cudaMemcpy(bDevice, b, sizeBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 1;
    }


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int block_size = 32;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(size / block_size + 1, size / block_size + 1);

    cudaEventRecord(start);
    matrixMultiplicationKernel<<<dimGrid, dimBlock>>>(aDevice, bDevice, cDevice, size);
    cudaEventRecord(stop);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return 1;
    }

    cudaStatus = cudaMemcpy(c, cDevice, sizeBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 1;
    }

    cudaFree(aDevice);
    cudaFree(bDevice);
    cudaFree(cDevice);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA time simple (ms): %f\n", milliseconds);

    int justForWaiting;
    scanf("%d", &justForWaiting);

    return 0;
}