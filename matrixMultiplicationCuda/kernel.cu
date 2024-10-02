
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
    const size_t size = 10;


    double* a = new double[size * size];
    double* b = new double[size * size];
    double* c = new double[size * size];

    for (int i = 0; i < size * size; ++i)
    {
        a[i] = 5.0;
        b[i] = 5.0;
        c[i] = 0.0;
    }


    const size_t sizeBytes = size * size * sizeof(double);
    double* aDevice;
    double* bDevice;
    double* cDevice;

    cudaMalloc((void**)&aDevice, sizeBytes);
    cudaMemcpy(aDevice, a, sizeBytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bDevice, sizeBytes);
    cudaMemcpy(bDevice, b, sizeBytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cDevice, sizeBytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int block_size = 512;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(size / block_size + 1, size / block_size + 1);

    cudaEventRecord(start);
    matrixMultiplicationKernel << <dimGrid, dimBlock >> > (aDevice, bDevice, cDevice, size);

    cudaMemcpy(c, cDevice, sizeBytes, cudaMemcpyDeviceToHost);

    cudaFree(aDevice);
    cudaFree(bDevice);
    cudaFree(cDevice);


    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            printf("%.2lf\t", c[i * size + j]);
        }
        printf("\n");
    }

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA time simple (ms): %f\n", milliseconds);

    int justForWaiting;
    scanf("%d", &justForWaiting);

    return 0;
}