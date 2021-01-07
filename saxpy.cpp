#include "hip/hip_runtime.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>


__global__
void saxpy( int n, float a, float* x, float* y )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}


int main( int argc, char** argv )
{
  if( argc < 2 )
  {
    std::cout << "Usage: " << argv[0] << " <buffer size in MB>" << std::endl;
    exit( 1 );
  }
  unsigned int N = (1 << 20) * atoi( argv[1] );  // Buffer size

  std::cout << "Buffer size: " << N << std::endl;

  float* x = new float[N];
  float* y = new float[N];
  float* d_x = NULL;
  float* d_y = NULL;

  hipMalloc( &d_x, N*sizeof(float) );
  hipMalloc( &d_y, N*sizeof(float) );

  std::fill( x, x + N, 1.0 );
  std::fill( y, y + N, 2.0 );

  hipMemcpy( d_x, x, N*sizeof(float), hipMemcpyHostToDevice );
  hipMemcpy( d_y, y, N*sizeof(float), hipMemcpyHostToDevice );

  // Run kernel
  int num_blocks = (N + 255) / 256;
  int block_size = 256;
  hipLaunchKernelGGL(saxpy, dim3(num_blocks), dim3(block_size), 0, 0,  N, 2.0f, d_x, d_y );

  hipMemcpy( y, d_y, N*sizeof(float), hipMemcpyDeviceToHost );

  float maxErr = 0.0f;
  for( unsigned int i = 0; i < N; ++i )
    maxErr = std::max( maxErr, std::abs(y[i] - 4.0f) );
  std::cout << "Max error: " << maxErr << std::endl;

  hipFree( d_x );
  hipFree( d_y );

  delete[] x;
  delete[] y;

  return 0;
}
