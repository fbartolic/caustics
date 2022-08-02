// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "ehrlich_aberth.h"
#include "pybind11_kernel_helpers.h"

using complex = thrust::complex<double>;

using namespace ehrlich_aberth_jax;

namespace
{

  void cpu_ehrlich_aberth(void *out, const void **in)
  {
    // Parse the inputs
    // reinterpret_cast here converts a void* to a pointer to a pointer for a specific type
    const int size =
        *reinterpret_cast<const int *>(in[0]);                       // number of polynomials (size of problem)
    const int deg = *reinterpret_cast<const int *>(in[1]);           // degree of polynomials
    const int itmax = *reinterpret_cast<const int *>(in[2]);         // maxiter
    const bool compensated = *reinterpret_cast<const bool *>(in[3]); // maxiterk
    const bool custom_init = *reinterpret_cast<const bool *>(in[4]); // custom initialization values

    // Flattened polynomial coefficients, shape (deg + 1)*size
    const complex *coeffs = reinterpret_cast<const complex *>(in[5]);

    // Output roots, shape deg*size
    complex *roots = reinterpret_cast<complex *>(out);

    // Flattened initialization values for the roots (shape deg*size)
    const complex *roots_init = reinterpret_cast<const complex *>(in[6]);

    // Allocate memory for temporary arrays
    double *alpha = new double[deg + 1];
    bool *conv = new bool[deg];
    point_conv *conv2 = new point_conv[deg];
    point *points = new point[deg + 1];
    point *hull = new point[deg + 1];

    // Compute roots
    if (compensated)
    {
      for (int idx = 0; idx < size; ++idx)
      {
        if (custom_init){
        ehrlich_aberth_jax::ehrlich_aberth_comp(deg, itmax, custom_init, coeffs + idx * (deg + 1), roots_init + idx*deg,
            roots + idx * deg, alpha, conv, points, hull);
        }
        else {
        ehrlich_aberth_jax::ehrlich_aberth_comp(deg, itmax, custom_init, coeffs + idx * (deg + 1),nullptr,
            roots + idx * deg, alpha, conv, points, hull);

        }
 
      }
    }
    else
    {
      for (int idx = 0; idx < size; ++idx)
      {
        if (custom_init){
        ehrlich_aberth_jax::ehrlich_aberth(deg, itmax, custom_init, coeffs + idx * (deg + 1), roots_init + idx*deg,
            roots + idx * deg, alpha, conv, points, hull);
        }
        else {
        ehrlich_aberth_jax::ehrlich_aberth(deg, itmax, custom_init, coeffs + idx * (deg + 1),nullptr,
            roots + idx * deg, alpha, conv, points, hull);

        }
      }
    }

    // Free memory
    delete[] alpha;
    delete[] conv;
    delete[] conv2;
    delete[] points;
    delete[] hull;
  }

  pybind11::dict Registrations()
  {
    pybind11::dict dict;
    dict["cpu_ehrlich_aberth"] = EncapsulateFunction(cpu_ehrlich_aberth);
    return dict;
  }

  PYBIND11_MODULE(ehrlich_aberth_cpu_op, m) { m.def("registrations", &Registrations); }

} // namespace
