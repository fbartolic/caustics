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
    dict["gpu_integrate_image"] = EncapsulateFunction(gpu_integrate_image);
    return dict;
  }

  PYBIND11_MODULE(integrate_image_gpu_op, m)
  {
    m.def("registrations", &Registrations);
    //    m.def("build_integrate_image_descriptor", [](int dummy)
    //          { return PackDescriptor(IntegrateImageDescriptor{nr, ntheta}); });
  }
} // namespace
