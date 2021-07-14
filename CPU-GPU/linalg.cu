#include "cuda_utils.h"
#include "linalg.h"
#include <stdio.h>

#ifdef CUDA
extern cublasHandle_t cublas_handle;
#endif

/*
 *  Naive matrix multiply kernels.
 */
__global__ void dgemm_naive(const double *A, const double *B,
                            double *C,
                            const int M, const int N, const int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N)
  {
    double sum = 0.;
    for (int k = 0; k < K; k++)
      sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

__global__ void dgemm_ta_naive(const double *A, const double *B,
                               double *C,
                               const int M, const int N, const int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N)
  {
    double sum = 0;
    for (int k = 0; k < K; k++)
      sum += A[k * M + row] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

__global__ void dgemm_tb_naive(const double *A, const double *B, const double *C,
                               double *D,
                               const int M, const int N, const int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N)
  {
    double sum = 0;
    for (int k = 0; k < K; k++)
      sum += A[row * K + k] * B[col * K + k];
    D[row * N + col] = sum + C[row * N + col];
  }
}

/*
 *  Optimized matrix multiply kernels using shared memory.
 */
// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
__global__ void dgemm_optimized(const double *A, const double *B,
                                double *C,
                                const int M, const int N, const int K,
                                const int SHARED_MEM_DIM)
{
  /*
   * FILLME: fill the code.
   */

  extern __shared__ double partials[];
  // Shared storage for a single tile of A
  double *partials_A = partials;
  // Shared storage for a single tile of B
  double *partials_B = partials_A + SHARED_MEM_DIM * SHARED_MEM_DIM;

  // Get row and column of **output C**
  // (We utilize one thread per output value of C)
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int local_tidx = threadIdx.x;
  int local_tidy = threadIdx.y;

  if (row < M && col < N)
  {
    double temp = 0;
    int i, j;

    // For all "imaginary" blocks (tiles) that are required
    // to traverse dimension K (dimension of dot product)
    // ( & unroll loop for performance)
#pragma unroll
    for (int current_block = 0; current_block < (K - 1) / SHARED_MEM_DIM + 1; current_block++)
    {

      // Column j of matrix A (sliding offset per tile)
      j = current_block * blockDim.x + threadIdx.x;
      // Row i of matrix B (sliding offset per tile)
      i = current_block * blockDim.y + threadIdx.y;

      // Load A[row][j] to shared mem from global mem
      partials_A[local_tidy * SHARED_MEM_DIM + local_tidx] = A[row * K + j];

      // Load B[i][col] to shared mem from global mem
      partials_B[local_tidy * SHARED_MEM_DIM + local_tidx] = B[i * N + col];

      // Synchronize before computation
      // so that shared memory is full !
      __syncthreads();

      // Calculate **partial** dot products ONLY from shared memory !
      // In a single tile, SHARED_MEM_DIM * SHARED_MEM_DIM partial
      // results will be computed.
      // ( & unroll loop for performance)
#pragma unroll
      for (int k = 0; k < SHARED_MEM_DIM; k++)
      {

        // No shared memory bank conflict!
        temp += partials_A[local_tidy * SHARED_MEM_DIM + k] * partials_B[k * SHARED_MEM_DIM + local_tidx];
      }

      // Synchronize before next iteration so that previous
      // values of shared memory are not overwritten BEFORE
      // another thread has not used them yet!
      // (for whatever reason)
      __syncthreads();
    }

    // Now that all tiles in the dimension of dot product have been
    // **accumulated** on 'temp', we can store one result per thread to
    // outpur array.

    // Store result in C
    C[row * N + col] = temp;
  }
}

// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
__global__ void dgemm_ta_optimized(const double *A, const double *B,
                                   double *C,
                                   const int M, const int N, const int K,
                                   const int SHARED_MEM_DIM)
{
  /*
   * FILLME: fill the code.
   */

    extern __shared__ double partials[];
  // Shared storage for a single tile of A
  double *partials_A = partials;
  // Shared storage for a single tile of B
  double *partials_B = partials_A + SHARED_MEM_DIM * SHARED_MEM_DIM;

  // Get row and column of **output C**
  // (We utilize one thread per output value of C)
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int local_tidx = threadIdx.x;
  int local_tidy = threadIdx.y;

  if (row < M && col < N)
  {
    double temp = 0;
    int i, j;

    // For all "imaginary" blocks (tiles) that are required
    // to traverse dimension K (dimension of dot product)
    // ( & unroll loop for performance)
#pragma unroll
    for (int current_block = 0; current_block < (K - 1) / SHARED_MEM_DIM + 1; current_block++)
    {

      // Row j of matrix A (sliding offset per tile)
      j = current_block * blockDim.x + threadIdx.x;
      // Row i of matrix B (sliding offset per tile)
      i = current_block * blockDim.y + threadIdx.y;

      // Load A[j][row] to shared mem from global mem
      // row === column of transposed matrix A
      partials_A[local_tidy * SHARED_MEM_DIM + local_tidx] = A[j * M + row];

      // Load B[i][col] to shared mem from global mem
      partials_B[local_tidy * SHARED_MEM_DIM + local_tidx] = B[i * N + col];

      // Synchronize before computation
      // so that shared memory is full !
      __syncthreads();

      // Calculate **partial** dot products ONLY from shared memory !
      // In a single tile, SHARED_MEM_DIM * SHARED_MEM_DIM partial
      // results will be computed.
      // ( & unroll loop for performance)
#pragma unroll
      for (int k = 0; k < SHARED_MEM_DIM; k++)
      {

        // No shared memory bank conflict!
        temp += partials_A[local_tidy * SHARED_MEM_DIM + k] * partials_B[k * SHARED_MEM_DIM + local_tidx];
      }

      // Synchronize before next iteration so that previous
      // values of shared memory are not overwritten BEFORE
      // another thread has not used them yet!
      // (for whatever reason)
      __syncthreads();
    }

    // Now that all tiles in the dimension of dot product have been
    // **accumulated** on 'temp', we can store one result per thread to
    // outpur array.

    // Store result in C
    C[row * N + col] = temp;
  }
}

// Computes D = A*B'+C, where A is a M by K matrix, B is a N by K matrix, C and D are M by N matrices.
// Matrices are stored in row-major order.
__global__ void dgemm_tb_optimized(const double *A, const double *B, const double *C,
                                   double *D,
                                   const int M, const int N, const int K,
                                   const int SHARED_MEM_DIM)
{
  /*
   * FILLME: fill the code.
   */

  extern __shared__ double partials[];
  // Shared storage for a single tile of A
  double *partials_A = partials;
  // Shared storage for a single tile of B
  double *partials_B = partials_A + SHARED_MEM_DIM * SHARED_MEM_DIM;

  // Get row and column of **output C**
  // (We utilize one thread per output value of C)
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int local_tidx = threadIdx.x;
  int local_tidy = threadIdx.y;

  if (row < M && col < N)
  {
    double temp = 0;
    int i, j;

    // For all "imaginary" blocks (tiles) that are required
    // to traverse dimension K (dimension of dot product)
    // ( & unroll loop for performance)
#pragma unroll
    for (int current_block = 0; current_block < (K - 1) / SHARED_MEM_DIM + 1; current_block++)
    {

      // Column j of matrix A (sliding offset per tile)
      j = current_block * blockDim.x + threadIdx.x;
      // Row i of matrix B (sliding offset per tile)
      i = current_block * blockDim.y + threadIdx.y;

      // Load A[row][j] to shared mem from global mem
      partials_A[local_tidy * SHARED_MEM_DIM + local_tidx] = A[row * K + j];

      // Load B[i][col] to shared mem from global mem
      partials_B[local_tidy * SHARED_MEM_DIM + local_tidx] = B[col * K + i];

      // Synchronize before computation
      // so that shared memory is full !
      __syncthreads();

      // Calculate **partial** dot products ONLY from shared memory !
      // In a single tile, SHARED_MEM_DIM * SHARED_MEM_DIM partial
      // results will be computed.
      // ( & unroll loop for performance)
#pragma unroll
      for (int k = 0; k < SHARED_MEM_DIM; k++)
      {

        // No shared memory bank conflict!
        temp += partials_A[local_tidy * SHARED_MEM_DIM + k] * partials_B[k * SHARED_MEM_DIM + local_tidx];
      }

      // Synchronize before next iteration so that previous
      // values of shared memory are not overwritten BEFORE
      // another thread has not used them yet!
      // (for whatever reason)
      __syncthreads();
    }

    // Now that all tiles in the dimension of dot product have been
    // **accumulated** on 'temp', we can store one result per thread to
    // outpur array.

    // Store result in D
    D[row * N + col] = C[row * N + col] + temp;
  }
}

// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_gpu(const double *A, const double *B, double *C, const int M, const int N, const int K)
{
#ifndef CUBLAS
  const int BLOCK_SIZE = 16;
#if defined(_GPU_GEMM_NAIVE)
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dgemm_naive<<<grid, block>>>(A, B, C, M, N, K);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
#elif defined(_GPU_GEMM_OPT)
  /*
   *  FILLME: Set up the blocks, grid and the shared memory size.
   */
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  size_t shmem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
  dgemm_optimized<<<grid, block, shmem_size>>>(A, B, C, M, N, K, BLOCK_SIZE);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
#endif
#else
  // Matrices are stored in row-major order, but cuBLAS assumes column-major
  // order. We want to compute:
  //         A * B = (A^T)^T * (B^T)^T = A'^T * B'^T = (B' * A')^T
  /*
 *  FILLME: Use cublasDgemm()
 */
  cublasHandle_t handle;
  cublasCreate(&handle);
  const double a1 = 1;
  const double b1 = 0;
  cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K, //m, n, k (because B*A)
              &a1,     //*alpha
              B, N,    //lda, rows of B^T if no trans, cols of B^T otherwise
              A, K,    //ldb
              &b1,     //*beta
              C, N);   //ldc, no transform on C^T
  cublasDestroy(handle);
#endif
}

// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_ta_gpu(const double *A, const double *B, double *C, const int M, const int N, const int K)
{
#ifndef CUBLAS
  const int BLOCK_SIZE = 16;
#if defined(_GPU_GEMM_NAIVE)
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dgemm_ta_naive<<<grid, block>>>(A, B, C, M, N, K);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
#elif defined(_GPU_GEMM_OPT)
  /*
   *  FILLME: Set up the blocks, grid and the shared memory size.
   */
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  size_t shmem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
  dgemm_ta_optimized<<<grid, block, shmem_size>>>(A, B, C, M, N, K, BLOCK_SIZE);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
#endif
#else
  // Matrices are stored in row-major order, but cuBLAS assumes column-major
  // order. We want to compute:
  //         A^T * B = A^T * (B^T)^T = A' * B'^T = (B'*A'^T)^T
  /*
 *  FILLME: Use cublasDgemm()
 */
  cublasHandle_t handle;
  cublasCreate(&handle);
  const double a2 = 1;
  const double b2 = 0;
  cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
              N, M, K, //m, n, k
              &a2,     //*alpha
              B, N,    //lda
              A, M,    //ldb
              &b2,     //*beta
              C, N);   //ldc
  cublasDestroy(handle);

#endif
}

// Computes D = A*B'+C, where A is a M by K matrix, B is a N by K matrix, C and D are M by N matrices.
// Matrices are stored in row-major order.
void dgemm_tb_gpu(const double *A, const double *B, const double *C, double *D, const int M, const int N, const int K)
{
#ifndef CUBLAS
  const int BLOCK_SIZE = 16;
#if defined(_GPU_GEMM_NAIVE)
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dgemm_tb_naive<<<grid, block>>>(A, B, C, D, M, N, K);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
#elif defined(_GPU_GEMM_OPT)
  /*
   *  FILLME: Set up the blocks, grid and the shared memory size.
   */
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  size_t shmem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
  dgemm_tb_optimized<<<grid, block, shmem_size>>>(A, B, C, D, M, N, K, BLOCK_SIZE);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
#endif
#else
  // D = A * B^T
  // Matrices are stored in row-major order, but cuBLAS assumes column-major
  // order. We want to compute:
  //         C = A * B^T = (A^T)^T * B^T  = A'^T * B' = (B'^T * A')^T
  /*
 *  FILLME: Use cublasDgemm()
 */
  cublasHandle_t handle;
  cublasCreate(&handle);
  const double a3 = 1;
  const double b3 = 0;
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
              N, M, K, //m, n, k
              &a3,     //*alpha
              B, K,    //lda
              A, K,    //ldb
              &b3,     //*beta
              D, N);   //ldc
  cublasDestroy(handle);

  // D = C + D
  /*
 *  FILLME: Use cublasDgeam()
 */
  cublasHandle_t handle1;
  cublasCreate(&handle1);
  const double a4 = 1;
  const double b4 = 1;
  cublasDgeam(handle1, CUBLAS_OP_N, CUBLAS_OP_N,
              M, N,  //m, n
              &a4,   //*alpha
              C, M,  //lda
              &b4,   //*beta
              D, M,  //ldb
              D, M); //ldc
  cublasDestroy(handle1);

#endif
}
