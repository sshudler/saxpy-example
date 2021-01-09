#include <CL/sycl.hpp>
#include <array>
#include <iostream>

#include "dpc_common.hpp"


int main( int argc, char** argv )
{
  if( argc < 2 )
  {
    std::cout << "Usage: " << argv[0] << " <buffer size in MB>" << std::endl;
    exit( 1 );
  }
  unsigned int N = (1 << 20) * atoi( argv[1] );  // Buffer size

  std::cout << "Buffer size: " << N << std::endl;

  std::vector<float> x_v( N );
  std::vector<float> y_v( N );

  std::fill( x_v.begin(), x_v.end(), 1.0 );
  std::fill( y_v.begin(), y_v.end(), 2.0 );

  float value = 2.0f;

  sycl::gpu_selector d_selector;

  try
  {
    sycl::queue q( d_selector, dpc_common::exception_handler );

    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    sycl::buffer x_buff( x_v );
    sycl::buffer y_buff( y_v );

    sycl::range n_items{ N };

    q.submit([&](auto &h) {
      sycl::accessor x_acc( x_buff, h, sycl::read_only, sycl::noinit );
      sycl::accessor y_acc( y_buff, h, sycl::read_write, sycl::noinit );

      h.parallel_for( n_items, [=](auto i) { y_acc[i] = value * x_acc[i] + y_acc[i];  });
    });
  }
  catch( std::exception const &e  )
  {
    std::cout << "Caught an exception while computing on device -- " << e.what() << std::endl;
    std::terminate();
  }

  float maxErr = 0.0f;
  for( unsigned int i = 0; i < N; ++i )
    maxErr = std::max( maxErr, std::abs(y_v[i] - 4.0f) );
  std::cout << "Max error: " << maxErr << std::endl;


  return 0;
}
