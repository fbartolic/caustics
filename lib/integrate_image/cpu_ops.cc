// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.
// compile with clang++ cpu_ops.cc -Xclang -load -Xclang /home/fb90/opt/Enzyme/enzyme/build/Enzyme/ClangEnzyme-13.so -O2 -flegacy-pass-manager -std=c++14 $(python3 -m pybind11 --includes) -o integrate_image_cpu_op$(python3-config --extension-suffix) --gcc-toolchain="/share/apps/gcc112/" -shared -fPIC

#include "pybind11_kernel_helpers.h"
#include <cmath>
#include <iostream>

using namespace integrate_polar;

namespace

{

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

  double lens_eq_binary_real(double z_real, double z_imag, double a, double e1)
  {
    // e1*(a - re(z))/((a - re(z))**2 + im(z)**2) + (a + re(z))*(e1 - 1.0)/((a + re(z))**2 + im(z)**2) + re(z)
    return e1 * (a - z_real) / ((a - z_real) * (a - z_real) + z_imag * z_imag) + (a + z_real) * (e1 - 1.0) / ((a + z_real) * (a + z_real) + z_imag * z_imag) + z_real;
  }
  double lens_eq_binary_imag(double z_real, double z_imag, double a, double e1)
  {
    // -e1*im(z)/((a - re(z))**2 + im(z)**2) + (e1 - 1.0)*im(z)/((a + re(z))**2 + im(z)**2) + im(z)
    return -e1 * z_imag / ((a - z_real) * (a - z_real) + z_imag * z_imag) + (e1 - 1.0) * z_imag / ((a + z_real) * (a + z_real) + z_imag * z_imag) + z_imag;
  }

  double linear_limbdark(double r, double I0, double c)
  {
    return I0 * (1.0 - c * (1.0 - std::sqrt(1.0 - r * r)));
  }

  void evaluate_integrand(
      const double rmin, const double theta_min, const double dr, const double dtheta, const int nr,
      const int ntheta, const double rho, const double a1, const double a, const double e1, const double w_cent_real,
      const double w_cent_imag, double *integrand)
  {
    for (int tid = 0; tid < nr * ntheta; tid++)
    {
      int i = tid / ntheta;
      int j = tid % ntheta;

      double r = rmin + i * dr + 0.5 * dr;
      double theta = add_angles(theta_min, j * dtheta + 0.5 * dtheta);

      double w_real = lens_eq_binary_real(r * std::cos(theta), r * std::sin(theta), a, e1);
      double w_imag = lens_eq_binary_imag(r * std::cos(theta), r * std::sin(theta), a, e1);

      // Check if point falls inside the source
      double xs = w_real - w_cent_real;
      double ys = w_imag - w_cent_imag;
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
                       const double rho, const double a1, const double a, const double e1, const double w_cent_real,
                       const double w_cent_imag)
  {
    int N = nr * ntheta;

    // Initialize array of size (nr, ntheta)
    double *integrand = new double[N];

    evaluate_integrand(rmin, theta_min, dr, dtheta, nr, ntheta, rho, a1, a, e1,
                       w_cent_real, w_cent_imag, integrand);

    // Sum the columns of integrand matrix, adapted from:
    double *integrand_sum = new double[ntheta];
    for (int j = 0; j < ntheta; j++)
    {
      double sum = 0.0;
      for (int i = 0; i < nr; i++)
      {
        sum += integrand[i * ntheta + j];
      }
      integrand_sum[j] = sum;
    }

    // Sum the rows
    double sum = 0.0;
    for (int i = 0; i < ntheta; i++)
    {
      sum += integrand_sum[i];
    }

    // Free memory
    delete[] integrand_sum;
    delete[] integrand;

    return sum;
  }

  void integrate_image(double *params, double grid_ratio, double f, double eps_max, double *result)
  {

    double rmin = params[0];
    double rmax = params[1];
    double theta_min = params[2];
    double theta_max = params[3];
    double dr = params[4];
    double dtheta = params[5];
    double rho = params[6];
    double a1 = params[7];
    double a = params[8];
    double e1 = params[9];
    double w_cent_real = params[10];
    double w_cent_imag = params[11];

    // Return 0. if the bounding box is empty
    if (rmax - rmin == 0. || theta_max - theta_min == 0.)
    {

      *result = 0.0;
    }
    else
    {
      double eps = 1.;
      double I_previous = 0.;
      double I_estimate;

      //        while (eps > eps_max)
      //        {

      int nr = std::ceil((rmax - rmin) / dr);
      int ntheta = std::ceil(ang_dist(theta_min, theta_max) / dtheta);
      int N = nr * ntheta;

      // Because of memory limitations, we need to compute the integral in chunks
      double sum = 0.0;
      int n_chunks = std::max<int>(int((N / 2.e06)), 1);

      // Iterate over chunks of the grid (split across radius r) and add to the sum
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
        sum += sum_integrand(nr_chunk, ntheta, _rmin, theta_min, dr, dtheta, rho, a1, a, e1, w_cent_real, w_cent_imag);
      }

      I_estimate = sum * dr * dtheta;
      eps = std::abs((I_estimate - I_previous) / I_previous);
      I_previous = I_estimate;

      // Update the step size
      // dr *= f;
      // dtheta = grid_ratio * dr;
      //        }
      *result = I_estimate;
    }
  }

  // Evaluates the integral
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
    const double w_cent_real = *reinterpret_cast<const double *>(in[10]);
    const double w_cent_imag = *reinterpret_cast<const double *>(in[11]);
    const double eps_max = *reinterpret_cast<const double *>(in[12]);
    const double f = *reinterpret_cast<const double *>(in[13]);
    const double grid_ratio = *reinterpret_cast<const double *>(in[14]);

    // Output is the value of the integral
    double *final_result = reinterpret_cast<double *>(out);

    double params[12] = {rmin, rmax, theta_min, theta_max, dr, dtheta, rho, a1, a, e1, w_cent_real, w_cent_imag};

    integrate_image(params, grid_ratio, f, eps_max, final_result);
  }

  void __enzyme_autodiff(void (*)(double *, double, double, double, double *), ...);

  extern "C"
  {
    extern int enzyme_dup;
    extern int enzyme_const;
    extern int enzyme_dupnoneed;
  }

  // evaluates the gradient of the image integral
  void cpu_integrate_image_grad(void *out, const void **in)
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
    const double w_cent_real = *reinterpret_cast<const double *>(in[10]);
    const double w_cent_imag = *reinterpret_cast<const double *>(in[11]);
    const double eps_max = *reinterpret_cast<const double *>(in[12]);
    const double f = *reinterpret_cast<const double *>(in[13]);
    const double grid_ratio = *reinterpret_cast<const double *>(in[14]);

    // Output is the gradient of the integral with respect to the inputs
    double *output = reinterpret_cast<double *>(out);

    // Function parameters
    double params[12] = {rmin, rmax, theta_min, theta_max, dr, dtheta, rho, a1, a, e1, w_cent_real, w_cent_imag};

    // Initialize gradient
    double grad_params[12] = {0.};

    double result = 0.0;
    double dresult = 1.0;

    // Evaluate gradient with enzyme
    __enzyme_autodiff(
        integrate_image,
        enzyme_dupnoneed, params,
        grad_params,
        enzyme_const, grid_ratio,
        enzyme_const, f,
        enzyme_const, eps_max,
        enzyme_dup, &result,
        &dresult);

    output[0] = result;
    for (int i = 0; i < 12; i++)
    {
      output[i + 1] = grad_params[i];
    }
  }

  pybind11::dict Registrations()
  {
    pybind11::dict dict;
    dict["cpu_integrate_image"] = EncapsulateFunction(cpu_integrate_image);
    dict["cpu_integrate_image_grad"] = EncapsulateFunction(cpu_integrate_image_grad);
    return dict;
  }

  PYBIND11_MODULE(integrate_image_cpu_op, m) { m.def("registrations", &Registrations); }

} // namespace
