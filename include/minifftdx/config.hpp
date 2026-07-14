#pragma once
// Host/device annotation macros. The same headers compile with a plain C++
// compiler (CPU backend) and with nvcc (CUDA backend).
#if defined(__CUDACC__)
#define MFX_HD __host__ __device__
#define MFX_INLINE __forceinline__
#else
#define MFX_HD
#define MFX_INLINE inline
#endif
