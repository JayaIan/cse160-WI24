#include <gputk.h>
const int BLOCK_WIDTH = 16;

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this lab
  __shared__ float subTileM[BLOCK_WIDTH][BLOCK_WIDTH];
  __shared__ float subTileN[BLOCK_WIDTH][BLOCK_WIDTH];
  // Calculate the row index of the Matrix A
  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  // Calculate the column index of Matrix B
  int Col = blockIdx.x*blockDim.x+threadIdx.x;

  // Only operate if Row and Col are within the dimensions of Matrix C
  // if ((Row < numCRows) && (Col < numCColumns)) {
    float Cvalue = 0;
    for (int m = 0; m < ceil((1.0*numAColumns)/(1.0*BLOCK_WIDTH)); ++m) {
      // Collaborative loading of M and N tiles into shared memory
      if (m*BLOCK_WIDTH+threadIdx.x > (numAColumns - 1) || Row >= (numARows)) {
        subTileM[threadIdx.y][threadIdx.x] = 0;
      } else {
        subTileM[threadIdx.y][threadIdx.x] = A[Row*numAColumns+m*BLOCK_WIDTH+threadIdx.x];
      }

      if (m*BLOCK_WIDTH+threadIdx.y > (numBRows - 1) || Col >= (numBColumns)) {
        subTileN[threadIdx.y][threadIdx.x] = 0;
      } else {
        subTileN[threadIdx.y][threadIdx.x] = B[(m*BLOCK_WIDTH+threadIdx.y)*numBColumns+Col];
      }
      __syncthreads();
      for (int k = 0; k < BLOCK_WIDTH; ++k) {
  		  Cvalue += subTileM[threadIdx.y][k] * subTileN[k][threadIdx.x];
      }
   	  __syncthreads();
    }	
    C[Row*numCColumns+Col] = Cvalue;
  // }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numARows * numBColumns * sizeof(float));

  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  gpuTKLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 grid_size(ceil((1.0*numCColumns)/(1.0*BLOCK_WIDTH)), ceil((1.0*numCRows)/(1.0*BLOCK_WIDTH)), 1);
  dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<grid_size, block_size>>> (deviceA, deviceB, deviceC, numARows, numAColumns, 
                                            numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

//   // Print the 2D array A
//   for (int i = 0; i < numARows; ++i) {
//       for (int j = 0; j < numAColumns; ++j) {
//           printf("%f \t", *(hostA + i*numAColumns + j));
//       }
//       printf("\n");
//   }
// printf("\n");
//   // Print the 2D array B
//   for (int i = 0; i < numBRows; ++i) {
//       for (int j = 0; j < numBColumns; ++j) {
//           printf("%f \t", *(hostB + i*numBColumns + j));
//       }
//       printf("\n");
//   }
// printf("\n");
//   // Print the 2D array C
//   for (int i = 0; i < numCRows; ++i) {
//       for (int j = 0; j < numCColumns; ++j) {
//           printf("%f \t", *(hostC + i*numCColumns + j));
//       }
//       printf("\n");
//   }

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
