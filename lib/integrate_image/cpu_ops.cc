// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"
#include <cmath>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

using complex = thrust::complex<double>;

using namespace integrate_polar;

namespace

{

  struct sum_functor
  {
    int R;
    int C;
    double *arr;

    sum_functor(int _R, int _C, double *_arr) : R(_R), C(_C), arr(_arr){};

    double
    operator()(int myC)
    {
      double sum = 0;
      for (int i = 0; i < R; i++)
        sum += arr[i * C + myC];
      return sum;
    }
  };

  double add_angles(double a, double b)
  {
    double cos_apb = std::cos(a) * std::cos(b) - std::sin(a) * std::sin(b);
    double sin_apb = std::sin(a) * std::cos(b) + std::cos(a) * std::sin(b);
    return std::atan2(sin_apb, cos_apb);
  }

  double sub_angles(double a, double b)
  {
    double cos_amb = std::cos(a) * std::cos(b) + std::sin(a) * std::sin(b);
    double sin_amb = std::sin(a) * std::cos(b) - std::cos(a) * std::sin(b);
    return std::atan2(sin_amb, cos_amb);
  }
  double ang_dist(double theta1, double theta2)
  {
    return std::abs(sub_angles(theta1, theta2));
  }

  complex lens_eq_binary(complex z, double a, double e1)
  {
    complex zbar = thrust::conj(z);
    return z - e1 / (zbar - a) - (1.0 - e1) / (zbar + a);
  }

  double linear_limbdark(double r, double I0, double c)
  {
    return I0 * (1.0 - c * (1.0 - std::sqrt(1.0 - r * r)));
  }

  void evaluate_integrand(
      const double rmin, const double theta_min, const double dr, const double dtheta, const int nr,
      const int ntheta, const double rho, const double a1, const double a, const double e1, const complex source_center, double *integrand)
  {
    for (int tid = 0; tid < nr * ntheta; tid++)
    {
      int i = tid / ntheta;
      int j = tid % ntheta;

      double r = rmin + i * dr + 0.5 * dr;
      double theta = add_angles(theta_min, j * dtheta + 0.5 * dtheta);

      complex w = lens_eq_binary(r * thrust::exp(complex(0, theta)), a, e1);

      // Check if point falls inside the source
      double xs = w.real() - source_center.real();
      double ys = w.imag() - source_center.imag();
      double rs = std::sqrt(xs * xs + ys * ys);

      if (rs < rho)
      {
        integrand[tid] = r * linear_limbdark(rs, 1.0, a1);
      }
      else
      {
        integrand[tid] = 0.0;
      }
    }
  }

  double sum_integrand(const int nr, const int ntheta,
                       const double rmin, const double theta_min, const double dr, const double dtheta,
                       const double rho, const double a1, const double a, const double e1, const complex source_center)
  {
    int N = nr * ntheta;

    // Initialize array of size (nr, ntheta)
    double *integrand = new double[N];

    evaluate_integrand(rmin, theta_min, dr, dtheta, nr, ntheta, rho, a1, a, e1,
                       source_center, integrand);

    // Sum the columns of integrand matrix, adapted from:
    // https://stackoverflow.com/questions/34093054/cuda-thrust-how-to-sum-the-columns-of-an-interleaved-array
    int C = ntheta; // number of columns
    int R = nr;     // number of rows
    thrust::host_vector<double> array(integrand, integrand + N);

    // allocate storage for column sums and indices
    thrust::host_vector<double> col_sums(C);
    thrust::host_vector<int> col_indices(C);

    thrust::host_vector<double> fcol_sums(C);
    thrust::sequence(fcol_sums.begin(), fcol_sums.end()); // start with column index
    thrust::transform(fcol_sums.begin(), fcol_sums.end(), fcol_sums.begin(), sum_functor(R, C, thrust::raw_pointer_cast(array.data())));
    double sum = thrust::reduce(fcol_sums.begin(), fcol_sums.end(), (double)0, thrust::plus<double>());

    // Free memory
    delete[] integrand;

    return sum;
  }

  void cpu_integrate_image(void *out, const void **in)
  {
    double rmin = *reinterpret_cast<const double *>(in[0]);
    const double rmax = *reinterpret_cast<const double *>(in[1]);
    const double theta_min = *reinterpret_cast<const double *>(in[2]);
    const double theta_max = *reinterpret_cast<const double *>(in[3]);
    double dr = *reinterpret_cast<const double *>(in[4]);
    double dtheta = *reinterpret_cast<const double *>(in[5]);
    const double rho = *reinterpret_cast<const double *>(in[6]);
    const double a1 = *reinterpret_cast<const double *>(in[7]);
    const double a = *reinterpret_cast<const double *>(in[8]);
    const double e1 = *reinterpret_cast<const double *>(in[9]);
    const complex source_center = *reinterpret_cast<const complex *>(in[10]);
    const double eps_max = *reinterpret_cast<const double *>(in[11]);
    const double f = *reinterpret_cast<const double *>(in[12]);
    const double grid_ratio = *reinterpret_cast<const double *>(in[13]);

    // Output is the value of the integral
    double *final_result = reinterpret_cast<double *>(out);

    // Return 0. if the bounding box is empty
    if (rmax - rmin == 0. || theta_max - theta_min == 0.)
    {

      double result = 0.0;
      *final_result = result;
    }
    else
    {
      double eps = 1.;
      double I_previous = 0.;
      double I_estimate;

      while (eps > eps_max)
      {

        int nr = std::ceil((rmax - rmin) / dr);
        int ntheta = std::ceil(ang_dist(theta_min, theta_max) / dtheta);
        int N = nr * ntheta;

        // Because of memory limitations, we need to compute the integral in chunks
        double sum = 0.0;
        int n_chunks = std::max<int>(int((N / 2.e06)), 1);

        // Iterate over chunks of the grid (split across radius r) and compute add to the sum
        double _rmin = rmin;
        int chunk_size = int(std::ceil(nr / n_chunks));
        int nr_chunk = chunk_size;

        for (int i = 0; i < n_chunks; i++)
        {
          _rmin = rmin + i * chunk_size * dr;

          if (i == (n_chunks - 1))
          {
            nr_chunk = nr - i * chunk_size;
          }
          sum += sum_integrand(nr_chunk, ntheta, _rmin, theta_min, dr, dtheta, rho, a1, a, e1, source_center);
        }

        I_estimate = sum * dr * dtheta;
        eps = std::abs((I_estimate - I_previous) / I_previous);
        I_previous = I_estimate;

        // Update the step size
        dr *= f;
        dtheta = grid_ratio * dr;
      }

      // Store the result
      *final_result = I_estimate;
    }
  }

  pybind11::dict Registrations()
  {
    pybind11::dict dict;
    dict["cpu_integrate_image"] = EncapsulateFunction(cpu_integrate_image);
    return dict;
  }

  PYBIND11_MODULE(integrate_image_cpu_op, m) { m.def("registrations", &Registrations); }

} // namespace
