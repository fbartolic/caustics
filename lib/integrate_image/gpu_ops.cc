// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_integrate_polar_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace integrate_polar;

namespace
{
  pybind11::dict Registrations()
  {
    pybind11::dict dict;
    dict["gpu_integrate_polar"] = EncapsulateFunction(gpu_integrate_polar);
    return dict;
  }

  PYBIND11_MODULE(gpu_ops, m)
  {
    m.def("registrations", &Registrations);
    m.def("build_integrate_polar_descriptor", [](
                                                  double rmin,
                                                  double theta_min,
                                                  double dr,
                                                  double dtheta,
                                                  int nr,
                                                  int ntheta,
                                                  double rho,
                                                  double a1,
                                                  double a,
                                                  double e1,
                                                  complex source_center)
          { return PackDescriptor(IntegratePolarDescriptor{rmin, theta_min, dr, dtheta, nr, ntheta, rho, a1, a, e1, source_center}); });
  }
} // namespace
