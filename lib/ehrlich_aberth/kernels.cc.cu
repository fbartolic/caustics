#include "ehrlich_aberth.h"
#include "kernel_helpers.h"
#include "kernels.h"

using complex = thrust::complex<double>;

namespace ehrlich_aberth_jax
{

  namespace
  {

    // CUDA kernel
    __global__ void ehrlich_aberth_kernel(const int N, const int deg, const int itmax,
                                          const complex *coeffs, complex *roots, double *alpha,
                                          bool *conv, point *points, point *hull)
    {
      // Compute roots
      // This is a "grid-stride loop" see
      // http://alexminnaar.com/2019/08/02/grid-stride-loops.html

      for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
      {
        ehrlich_aberth(deg, itmax, coeffs + tid * (deg + 1), roots + tid * deg,
                       alpha + tid * (deg + 1), conv + tid * deg, points + tid * (deg + 1),
                       hull + tid * (deg + 1));
      }
    }

    __global__ void ehrlich_aberth_comp_kernel(const int N, const int deg, const int itmax,
                                               const complex *coeffs, complex *roots, double *alpha,
                                               point_conv *conv, point *points, point *hull)
    {

      for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
      {
        ehrlich_aberth_comp(deg, itmax, coeffs + tid * (deg + 1), roots + tid * deg,
                            alpha + tid * (deg + 1), conv + tid * deg, points + tid * (deg + 1),
                            hull + tid * (deg + 1));
      }
    }

    void ThrowIfError(cudaError_t error)
    {
      if (error != cudaSuccess)
      {
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }

    inline void apply_ehrlich_aberth(cudaStream_t stream, void **buffers, const char *opaque,
                                     std::size_t opaque_len)
    {
      const EhrlichAberthDescriptor &d =
          *UnpackDescriptor<EhrlichAberthDescriptor>(opaque, opaque_len);
      const int N = d.size;
      const int deg = d.deg;
      const int itmax = d.itmax;
      const bool compensated = d.compensated;

      const complex *coeffs = reinterpret_cast<const complex *>(buffers[0]);
      complex *roots = reinterpret_cast<complex *>(buffers[1]);

      // Preallocate memory for temporary arrays used within the kernel
      double *alpha;
      point *points;
      point *hull;
      bool *conv;
      point_conv *conv2;

      cudaMalloc(&alpha, N * (deg + 1) * sizeof(double));
      cudaMalloc(&points, N * (deg + 1) * sizeof(point));
      cudaMalloc(&hull, N * (deg + 1) * sizeof(point));
      cudaMalloc(&conv, N * deg * sizeof(bool));
      cudaMalloc(&conv2, N * deg * sizeof(point_conv));

      const int block_dim = 512;
      const int grid_dim = std::min<int>(1024, (N + block_dim - 1) / block_dim);

      if (compensated)
      {
        ehrlich_aberth_comp_kernel<<<grid_dim, block_dim, 0, stream>>>(N, deg, itmax, coeffs, roots, alpha,
                                                                       conv2, points, hull);
      }
      else
      {
        ehrlich_aberth_kernel<<<grid_dim, block_dim, 0, stream>>>(N, deg, itmax, coeffs, roots, alpha,
                                                                  conv, points, hull);
      }

      // Free memory
      cudaFree(alpha);
      cudaFree(conv);
      cudaFree(conv2);
      cudaFree(points);
      cudaFree(hull);

      cudaError_t cudaerr = cudaDeviceSynchronize();
      if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

      ThrowIfError(cudaGetLastError());
    }

  } // namespace

  void gpu_ehrlich_aberth(cudaStream_t stream, void **buffers, const char *opaque,
                          std::size_t opaque_len)
  {
    apply_ehrlich_aberth(stream, buffers, opaque, opaque_len);
  }

} // namespace ehrlich_aberth_jax
