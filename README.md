# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>AIM:</h3>
<h3>ENTER YOUR NAME</h3>
<h3>ENTER YOUR REGISTER NO</h3>
<h3>EX. NO</h3>
<h3>DATE</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```
%%writefile reduction.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// CUDA error checking
#define CHECK(call)                                      \
{                                                        \
    cudaError_t error = call;                            \
    if (error != cudaSuccess)                            \
    {                                                    \
        fprintf(stderr, "Error: %s\n",                   \
                cudaGetErrorString(error));              \
        exit(EXIT_FAILURE);                              \
    }                                                    \
}

// Calculate execution time
double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

// Reduction using loop unrolling by 8
__global__ void reduceUnrolling8(
    unsigned int *g_idata,
    unsigned int *g_odata,
    unsigned int n)
{
    unsigned int tid = threadIdx.x;

    unsigned int idx =
        blockIdx.x * blockDim.x * 8 + threadIdx.x;

    unsigned int *idata =
        g_idata + blockIdx.x * blockDim.x * 8;

    unsigned int sum = 0;

    // Unrolling by 8
    for (int k = 0; k < 8; k++)
    {
        unsigned int index = idx + k * blockDim.x;

        if (index < n)
        {
            sum += g_idata[index];
        }
    }

    idata[tid] = sum;

    __syncthreads();

    // Reduction within the block
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // Store block result
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}


// Reduction using loop unrolling by 16
__global__ void reduceUnrolling16(
    unsigned int *g_idata,
    unsigned int *g_odata,
    unsigned int n)
{
    unsigned int tid = threadIdx.x;

    unsigned int idx =
        blockIdx.x * blockDim.x * 16 + threadIdx.x;

    unsigned int *idata =
        g_idata + blockIdx.x * blockDim.x * 16;

    unsigned int sum = 0;

    // Unrolling by 16
    for (int k = 0; k < 16; k++)
    {
        unsigned int index = idx + k * blockDim.x;

        if (index < n)
        {
            sum += g_idata[index];
        }
    }

    idata[tid] = sum;

    __syncthreads();

    // Reduction within the block
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // Store block result
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}


// CPU recursive reduction
unsigned int recursiveReduce(
    unsigned int *data,
    unsigned int size)
{
    if (size == 1)
        return data[0];

    unsigned int stride = (size + 1) / 2;

    for (unsigned int i = 0; i < size - stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}


int main(int argc, char **argv)
{
    // Select GPU device
    int dev = 0;

    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    printf("Starting reduction on device %d: %s\n",
           dev, deviceProp.name);

    // Array initialization
    unsigned int size = 1 << 24;

    printf("\nArray size: %u\n", size);

    // Block size
    unsigned int blocksize = 512;

    if (argc > 1)
    {
        blocksize = atoi(argv[1]);
    }

    // Validate block size
    if (blocksize == 0 ||
        blocksize > 1024 ||
        (blocksize & (blocksize - 1)) != 0)
    {
        printf("Block size must be a power of 2 between 1 and 1024.\n");
        return EXIT_FAILURE;
    }

    // Calculate grid sizes
    unsigned int grid8 =
        (size + blocksize * 8 - 1) / (blocksize * 8);

    unsigned int grid16 =
        (size + blocksize * 16 - 1) / (blocksize * 16);

    printf("Block size: %u\n", blocksize);
    printf("Grid for Unrolling8: %u\n", grid8);
    printf("Grid for Unrolling16: %u\n", grid16);

    // Allocate host memory
    size_t bytes = size * sizeof(unsigned int);

    unsigned int *h_idata =
        (unsigned int *)malloc(bytes);

    unsigned int *h_odata =
        (unsigned int *)malloc(grid8 * sizeof(unsigned int));

    unsigned int *tmp =
        (unsigned int *)malloc(bytes);

    if (!h_idata || !h_odata || !tmp)
    {
        printf("Host memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (unsigned int i = 0; i < size; i++)
    {
        h_idata[i] = rand() & 0xFF;
    }

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;

    unsigned int cpu_sum = 0;
    unsigned int gpu_sum = 0;

    // Allocate device memory
    unsigned int *d_idata = NULL;
    unsigned int *d_odata = NULL;

    CHECK(cudaMalloc((void **)&d_idata, bytes));

    CHECK(cudaMalloc(
        (void **)&d_odata,
        grid8 * sizeof(unsigned int)));

    // CPU reduction
    iStart = seconds();

    cpu_sum = recursiveReduce(tmp, size);

    iElaps = seconds() - iStart;

    printf("\nCPU Reduction\n");
    printf("Elapsed time: %f sec\n", iElaps);
    printf("CPU Sum: %u\n", cpu_sum);


    // --------------------------------
    // GPU Kernel 1: Unrolling by 8
    // --------------------------------

    CHECK(cudaMemcpy(
        d_idata, h_idata, bytes,
        cudaMemcpyHostToDevice));

    CHECK(cudaDeviceSynchronize());

    iStart = seconds();

    reduceUnrolling8<<<grid8, blocksize>>>(
        d_idata, d_odata, size);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(
        h_odata, d_odata,
        grid8 * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (unsigned int i = 0; i < grid8; i++)
    {
        gpu_sum += h_odata[i];
    }

    printf("\nGPU Unrolling8\n");
    printf("Elapsed time: %f sec\n", iElaps);
    printf("GPU Sum: %u\n", gpu_sum);


    // --------------------------------
    // GPU Kernel 2: Unrolling by 16
    // --------------------------------

    CHECK(cudaMemcpy(
        d_idata, h_idata, bytes,
        cudaMemcpyHostToDevice));

    CHECK(cudaDeviceSynchronize());

    iStart = seconds();

    reduceUnrolling16<<<grid16, blocksize>>>(
        d_idata, d_odata, size);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(
        h_odata, d_odata,
        grid16 * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (unsigned int i = 0; i < grid16; i++)
    {
        gpu_sum += h_odata[i];
    }

    printf("\nGPU Unrolling16\n");
    printf("Elapsed time: %f sec\n", iElaps);
    printf("GPU Sum: %u\n", gpu_sum);


    // --------------------------------
    // Compare results
    // --------------------------------

    printf("\nResult Verification\n");

    if (gpu_sum == cpu_sum)
    {
        printf("Test PASSED!\n");
    }
    else
    {
        printf("Test FAILED!\n");
    }

    // Free host memory
    free(h_idata);
    free(h_odata);
    free(tmp);

    // Free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // Reset GPU
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```
## OUTPUT:
SHOW YOUR OUTPUT HERE
<img width="890" height="462" alt="Screenshot 2026-11-05 091505" src="https://github.com/user-attachments/assets/8db6e64e-d098-4862-93f3-abcfb44c1d88" />

## RESULT:
The host took _________ seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in ________ seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
