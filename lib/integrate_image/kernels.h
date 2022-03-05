#ifndef _INTEGRATE_POLAR_KERNELS_H_
#define _INTEGRATE_POLAR_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

using complex = thrust::complex<double>;

namespace integrate_polar
{
  struct IntegratePolarDescriptor
  {
    double rmin;
    double theta_min;
    double dr;
    double dtheta;
    int nr;
    int ntheta;
    double rho;
    double a1;
    double a;
    double e1;
    complex source_center;
  };

  void gpu_integrate_polar(cudaStream_t stream, void **buffers, const char *opaque,
                           std::size_t opaque_len);

} // namespace integrate_polar

#endif