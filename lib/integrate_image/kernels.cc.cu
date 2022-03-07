#include "kernel_helpers.h"
#include "kernels.h"
#include <iostream>
#include <thrust/device_vector.h>

using complex = thrust::complex<double>;

namespace integrate_polar
{

  namespace
  {

    struct sum_functor
    {
      int R;
      int C;
      double *arr;

      sum_functor(int _R, int _C, double *_arr) : R(_R), C(_C), arr(_arr){};

      __host__ __device__ double
      operator()(int myC)
      {
        double sum = 0;
        for (int i = 0; i < R; i++)
          sum += arr[i * C + myC];
        return sum;
      }
    };

    __host__ __device__ double add_angles(double a, double b)
    {
      double cos_apb = std::cos(a) * std::cos(b) - std::sin(a) * std::sin(b);
      double sin_apb = std::sin(a) * std::cos(b) + std::cos(a) * std::sin(b);
      return std::atan2(sin_apb, cos_apb);
    }

    __host__ __device__ complex lens_eq_binary(complex z, double a, double e1)
    {
      complex zbar = thrust::conj(z);
      return z - e1 / (zbar - a) - (1.0 - e1) / (zbar + a);
    }

    __host__ __device__ double linear_limbdark(double r, double I0, double c)
    {
      return I0 * (1.0 - c * (1.0 - std::sqrt(1.0 - r * r)));
    }

    // CUDA kernel
    __global__ void evaluate_integrand(
        const double *rmin, const double *theta_min, const double *dr, const double *dtheta, const int *nr,
        const int *ntheta, const double *rho, const double *a1, const double *a, const double *e1, const complex *source_center, double *integrand)
    {
      // This is a "grid-stride loop" see
      // http://alexminnaar.com/2019/08/02/grid-stride-loops.html

      for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < *nr * *ntheta; tid += blockDim.x * gridDim.x)
      {
        int i = tid / *ntheta;
        int j = tid % *ntheta;

        double r = *rmin + i * *dr + 0.5 * *dr;
        double theta = add_angles(*theta_min, j * *dtheta + 0.5 * *dtheta);

        complex w = lens_eq_binary(r * thrust::exp(complex(0, theta)), *a, *e1);

        // Inside the image
        double xs = w.real() - (*source_center).real();
        double ys = w.imag() - (*source_center).imag();
        double rs = std::sqrt(xs * xs + ys * ys);
        if (rs < *rho)
        {
          integrand[tid] = r * linear_limbdark(rs, 1.0, *a1);
        }
        else
        {
          integrand[tid] = 0.0;
        }
      }
    }

    void ThrowIfError(cudaError_t error)
    {
      if (error != cudaSuccess)
      {
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }

    inline void apply_integrate_image(cudaStream_t stream, void **buffers, const char *opaque,
                                      std::size_t opaque_len)
    {
      //      const IntegrateImageDescriptor &d =
      //          *UnpackDescriptor<IntegrateImageDescriptor>(opaque, opaque_len);
      //
      const double *rmin = reinterpret_cast<const double *>(buffers[0]);
      const double *theta_min = reinterpret_cast<const double *>(buffers[1]);
      const double *dr = reinterpret_cast<const double *>(buffers[2]);
      const double *dtheta = reinterpret_cast<const double *>(buffers[3]);
      const int *nr = reinterpret_cast<const int *>(buffers[4]);
      const int *ntheta = reinterpret_cast<const int *>(buffers[5]);
      const double *rho = reinterpret_cast<const double *>(buffers[6]);
      const double *a1 = reinterpret_cast<const double *>(buffers[7]);
      const double *a = reinterpret_cast<const double *>(buffers[8]);
      const double *e1 = reinterpret_cast<const double *>(buffers[9]);
      const complex *source_center = reinterpret_cast<const complex *>(buffers[10]);

      // Output is the value of the integral
      double *result = reinterpret_cast<double *>(buffers[11]);

      // Copy nr_d and ntheta_d to host
      int nr_h, ntheta_h;
      cudaMemcpy(&nr_h, nr, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&ntheta_h, ntheta, sizeof(int), cudaMemcpyDeviceToHost);

      // if nr_h or ntheta_h is zero, return 0
      if (nr_h == 0 || ntheta_h == 0)
      {
        double res = 0.0;
        cudaMemcpy(result, &res, sizeof(double), cudaMemcpyHostToDevice);
      }
      else
      {

        int N = nr_h * ntheta_h;

        // Preallocate memory for temporary arrays used within the kernel
        double *integrand;
        // integrand is a row-major matrix of size nr * ntheta
        ThrowIfError(cudaMalloc(&integrand, N * sizeof(double)));

        const int block_dim = 512;
        const int grid_dim = std::min<int>(1024, (N + block_dim - 1) / block_dim);

        evaluate_integrand<<<grid_dim, block_dim, 0, stream>>>(rmin, theta_min, dr, dtheta, nr, ntheta, rho, a1, a, e1,
                                                               source_center, integrand);

        // Sum the columns of integrand matrix, adapted from:
        // https://stackoverflow.com/questions/34093054/cuda-thrust-how-to-sum-the-columns-of-an-interleaved-array
        int C = ntheta_h; // number of columns
        int R = nr_h;     // number of rows
        thrust::device_vector<double> array(integrand, integrand + N);

        // allocate storage for column sums and indices
        thrust::device_vector<double> col_sums(C);
        thrust::device_vector<int> col_indices(C);

        thrust::device_vector<double> fcol_sums(C);
        thrust::sequence(fcol_sums.begin(), fcol_sums.end()); // start with column index
        thrust::transform(fcol_sums.begin(), fcol_sums.end(), fcol_sums.begin(), sum_functor(R, C, thrust::raw_pointer_cast(array.data())));
        double sum = thrust::reduce(fcol_sums.begin(), fcol_sums.end(), (double)0, thrust::plus<double>());
        cudaDeviceSynchronize();

        // Copy nr_d and ntheta_d to host
        // TODO: avoid doing this by directing the output of thrust::reduce from above
        // to device memory directly
        double dr_h, dtheta_h;
        cudaMemcpy(&dr_h, dr, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&dtheta_h, dtheta, sizeof(double), cudaMemcpyDeviceToHost);

        // Assign to result
        double res = sum * dr_h * dtheta_h;
        cudaMemcpy(result, &res, sizeof(double), cudaMemcpyHostToDevice);

        // Free memory
        cudaFree(integrand);

        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
          printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

        ThrowIfError(cudaGetLastError());
      }
    }

  } // namespace

  void gpu_integrate_image(cudaStream_t stream, void **buffers, const char *opaque,
                           std::size_t opaque_len)
  {
    apply_integrate_image(stream, buffers, opaque, opaque_len);
  }

} // namespace
