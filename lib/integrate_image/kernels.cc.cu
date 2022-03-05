#include "kernel_helpers.h"
#include "kernels.h"
#include <thrust/complex.h>

using complex = thrust::complex<double>;

namespace
{

  namespace
  {

    // CUDA kernel
    __global__ void integrate_polar_kernel(const int N)
    {
      // Compute roots
      // This is a "grid-stride loop" see
      // http://alexminnaar.com/2019/08/02/grid-stride-loops.html

      for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
      {
      }
    }

    void ThrowIfError(cudaError_t error)
    {
      if (error != cudaSuccess)
      {
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }

    inline void apply_integrate_polar(cudaStream_t stream, void **buffers, const char *opaque,
                                      std::size_t opaque_len)
    {
      const IntegratePolarDescriptor &d =
          *UnpackDescriptor<IntegratePolarDescriptor>(opaque, opaque_len);
      const double rmin = d.rmin;
      const double theta_min = d.theta_min;
      const double dr = d.dr;
      const double dtheta = d.dtheta; // theta_min/theta_max need to be exactly divisible by dtheta
      const int nr = d.nr;
      const int ntheta = d.ntheta;
      const double rho = d.rho;
      const double a1 = d.a1;                        // linear limb darkening coefficient
      const double a = d.a;                          // half the separation between lenses
      const double e1 = d.e1;                        // linear limb darkening coefficient
      const complex source_center = d.source_center; // center of the source star

      // Output is the value of the integral
      double *result = reinterpret_cast<double *>(buffers[0]);

      // Preallocate memory for temporary arrays used within the kernel

      const int N = 1000;
      const int block_dim = 512;
      const int grid_dim = std::min<int>(1024, (N + block_dim - 1) / block_dim);

      // Free memory
      //      cudaFree(alpha);
      //      cudaFree(conv);
      //      cudaFree(conv2);
      //      cudaFree(points);
      //      cudaFree(hull);

      cudaError_t cudaerr = cudaDeviceSynchronize();
      if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

      ThrowIfError(cudaGetLastError());
    }

  } // namespace

  void gpu_integrate_polar(cudaStream_t stream, void **buffers, const char *opaque,
                           std::size_t opaque_len)
  {
    apply_integrate_polar(stream, buffers, opaque, opaque_len);
  }

} // namespace
