#ifndef _INTEGRATE_POLAR_KERNELS_H_
#define _INTEGRATE_POLAR_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <thrust/complex.h>

using complex = thrust::complex<double>;

namespace integrate_polar
{
  //  struct IntegrateImageDescriptor
  //  {
  //    int dummy;
  //  };

  void gpu_integrate_image(cudaStream_t stream, void **buffers, const char *opaque,
                           std::size_t opaque_len);

} // namespace integrate_polar

#endif