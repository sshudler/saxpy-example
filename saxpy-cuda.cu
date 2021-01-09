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

  cudaMalloc( &d_x, N*sizeof(float) );
  cudaMalloc( &d_y, N*sizeof(float) );

  std::fill( x, x + N, 1.0 );
  std::fill( y, y + N, 2.0 );

  cudaMemcpy( d_x, x, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_y, y, N*sizeof(float), cudaMemcpyHostToDevice );

  // Run kernel
  int num_blocks = (N + 255) / 256;
  int block_size = 256;
  saxpy<<<num_blocks, block_size>>>( N, 2.0f, d_x, d_y );

  cudaMemcpy( y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost );

  float maxErr = 0.0f;
  for( unsigned int i = 0; i < N; ++i )
    maxErr = std::max( maxErr, std::abs(y[i] - 4.0f) );
  std::cout << "Max error: " << maxErr << std::endl;

  cudaFree( d_x );
  cudaFree( d_y );

  delete[] x;
  delete[] y;

  return 0;
}
