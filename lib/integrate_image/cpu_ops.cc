// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"
#include <cmath>
#include <thrust/complex.h>

using complex = thrust::complex<double>;

using namespace integrate_polar;

namespace

{

  double add_angles(double a, double b)
  {
    double cos_apb = std::cos(a) * std::cos(b) - std::sin(a) * std::sin(b);
    double sin_apb = std::sin(a) * std::cos(b) + std::cos(a) * std::sin(b);
    return std::atan2(sin_apb, cos_apb);
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

  void cpu_integrate_image(void *out, const void **in)
  {
    // Parse the inputs
    // reinterpret_cast here converts a void* to a pointer to a pointer for a specific type
    const double rmin = *reinterpret_cast<const double *>(in[0]);
    const double theta_min = *reinterpret_cast<const double *>(in[1]);
    const double dr = *reinterpret_cast<const double *>(in[2]);     // rmin/rmax need to be exactly divisible by dr
    const double dtheta = *reinterpret_cast<const double *>(in[3]); // theta_min/theta_max need to be exactly divisible by dtheta
    const int nr = *reinterpret_cast<const int *>(in[4]);
    const int ntheta = *reinterpret_cast<const int *>(in[5]);
    const double rho = *reinterpret_cast<const double *>(in[6]);
    const double a1 = *reinterpret_cast<const double *>(in[7]);               // linear limb darkening coefficient
    const double a = *reinterpret_cast<const double *>(in[8]);                // half the separation between lenses
    const double e1 = *reinterpret_cast<const double *>(in[9]);               // linear limb darkening coefficient
    const complex source_center = *reinterpret_cast<const complex *>(in[10]); // center of the source star

    // Output is the value of the integral
    double *result = reinterpret_cast<double *>(out);

    // Initialize array of size (nr, ntheta)
    double **integrand = new double *[nr];
    for (int i = 0; i < nr; i++)
    {
      integrand[i] = new double[ntheta];
    }

    // Evaluate the integrand at each point, it is zero if the point is outside the image
    // and equal to the source flux if inside the image
    for (int i = 0; i < nr; i++)
    {
      for (int j = 0; j < ntheta; j++)
      {
        double r = rmin + i * dr;
        double theta = theta_min + j * dtheta;

        complex w = lens_eq_binary(r * thrust::exp(complex(0, theta)), a, e1);

        // Inside the image
        double xs = w.real() - source_center.real();
        double ys = w.imag() - source_center.imag();
        double rs = std::sqrt(xs * xs + ys * ys);
        if (rs < rho)
        {
          integrand[i][j] = linear_limbdark(rs, 1.0, a1);
        }
        else
        {
          integrand[i][j] = 0.0;
        }
      }
    }

    // Compute the integral using trapezoidal rule
    double *integral_r = new double[ntheta];

    // Integrate over r
    for (int j = 0; j < ntheta; j++)
    {
      // Trapezoidal rule
      double sum = 0.0;
      for (int i = 1; i < nr - 1; i++)
      {
        double r = rmin + i * dr;
        sum += 2. * r * integrand[i][j];
      }
      double r0 = rmin;
      double rN = rmin + (nr - 1) * dr;
      sum += rmin * integrand[0][j] + rN * integrand[nr - 1][j];
      integral_r[j] = 0.5 * dr * sum;
    }

    // Integrate over theta
    double sum = 0.0;
    for (int i = 1; i < ntheta - 1; i++)
    {
      sum += 2. * integral_r[i];
    }
    sum += integral_r[0] + integral_r[ntheta - 1];
    sum *= 0.5 * dtheta;

    // Store the result
    *result = sum;

    // Free memory
    for (int i = 0; i < nr; i++)
    {
      delete[] integrand[i];
    }
    delete[] integral_r;
  }

  pybind11::dict Registrations()
  {
    pybind11::dict dict;
    dict["cpu_integrate_image"] = EncapsulateFunction(cpu_integrate_image);
    return dict;
  }

  PYBIND11_MODULE(integrate_image_cpu_op, m) { m.def("registrations", &Registrations); }

} // namespace
