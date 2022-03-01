#ifndef _EHRLICH_ABERT_JAX_KERNELS_H_
#define _EHRLICH_ABERT_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ehrlich_aberth_jax
{
  struct EhrlichAberthDescriptor
  {
    int size;
    int deg;
    int itmax;
    bool compensated;
  };

  void gpu_ehrlich_aberth(cudaStream_t stream, void **buffers, const char *opaque,
                          std::size_t opaque_len);

} // namespace ehrlich_aberth_jax

#endif