#pragma once
// CUDA backend: one thread block computes one FFT entirely in shared memory,
// using the very same fft_block<> routine as the CPU backend. Compiled only by
// nvcc (this header is empty for host-only compilers).
#if defined(__CUDACC__)

#include <cuda_runtime.h>

#include <cstddef>

#include "../fusion.hpp"

namespace minifftdx {
namespace cuda {

namespace detail {

__device__ __forceinline__ void sincospi_t(float x, float* s, float* c) { sincospif(x, s, c); }
__device__ __forceinline__ void sincospi_t(double x, double* s, double* c) { sincospi(x, s, c); }

// Dynamic shared memory, declared with one fixed type so that every template
// instantiation agrees on the symbol.
extern __shared__ __align__(16) unsigned char mfx_smem[];

}  // namespace detail

// Threads per block: a quarter of the points (one radix-4 butterfly each),
// clamped to [1, 1024].
template <unsigned N>
constexpr unsigned threads_per_block() {
    return N / 4 < 1 ? 1 : (N / 4 > 1024 ? 1024 : N / 4);
}

template <class T, unsigned N>
struct block_context {
    __device__ unsigned tid() const { return threadIdx.x; }
    __device__ unsigned nthreads() const { return blockDim.x; }
    __device__ void sync() const { __syncthreads(); }
    // exp(-2*pi*i*k/N); 2k/N is exactly representable, sincospi is ~1 ulp.
    __device__ complex<T> twiddle(unsigned k) const {
        T s, c;
        detail::sincospi_t(static_cast<T>(-2.0 * k / N), &s, &c);
        return complex<T>(c, s);
    }
};

template <class Desc>
__global__ void fft_kernel(const typename Desc::complex_type* in, typename Desc::complex_type* out) {
    using C = typename Desc::complex_type;
    constexpr unsigned N = Desc::size;
    C* smem = reinterpret_cast<C*>(detail::mfx_smem);
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * N;
    for (unsigned i = threadIdx.x; i < N; i += blockDim.x) smem[i] = in[base + i];
    __syncthreads();
    fft_block<Desc>(smem, block_context<typename Desc::value_type, N>{});
    for (unsigned i = threadIdx.x; i < N; i += blockDim.x) out[base + i] = smem[i];
}

// Fused: global load -> FFT -> op -> IFFT -> global store; one kernel, the
// intermediate spectrum lives only in shared memory.
template <class Fwd, class Inv, class Op>
__global__ void fused_kernel(const typename Fwd::complex_type* in, typename Fwd::complex_type* out, Op op) {
    using C = typename Fwd::complex_type;
    constexpr unsigned N = Fwd::size;
    C* smem = reinterpret_cast<C*>(detail::mfx_smem);
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * N;
    for (unsigned i = threadIdx.x; i < N; i += blockDim.x) smem[i] = in[base + i];
    __syncthreads();
    fused_fft_op_ifft_block<Fwd, Inv>(smem, op, block_context<typename Fwd::value_type, N>{});
    for (unsigned i = threadIdx.x; i < N; i += blockDim.x) out[base + i] = smem[i];
}

// Standalone pointwise kernel (the unfused middle step, via global memory).
template <class Desc, class Op>
__global__ void pointwise_kernel(typename Desc::complex_type* data, Op op) {
    constexpr unsigned N = Desc::size;
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * N;
    for (unsigned k = threadIdx.x; k < N; k += blockDim.x) data[base + k] = op(data[base + k], k);
}

namespace detail {
template <class Kernel>
cudaError_t allow_smem(Kernel kernel, std::size_t bytes) {
    if (bytes <= 48 * 1024) return cudaSuccess;  // default per-block limit
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
}
}  // namespace detail

// Device pointers; `in` and `out` may alias. Asynchronous on `stream`.
template <class Desc>
cudaError_t execute(const typename Desc::complex_type* d_in, typename Desc::complex_type* d_out,
                    std::size_t batch, cudaStream_t stream = 0) {
    if (batch == 0) return cudaSuccess;
    cudaError_t e = detail::allow_smem(fft_kernel<Desc>, Desc::storage_bytes);
    if (e != cudaSuccess) return e;
    fft_kernel<Desc><<<static_cast<unsigned>(batch), threads_per_block<Desc::size>(), Desc::storage_bytes, stream>>>(d_in, d_out);
    return cudaGetLastError();
}

template <class Fwd, class Inv, class Op>
cudaError_t execute_fused(const typename Fwd::complex_type* d_in, typename Fwd::complex_type* d_out,
                          std::size_t batch, Op op, cudaStream_t stream = 0) {
    if (batch == 0) return cudaSuccess;
    cudaError_t e = detail::allow_smem(fused_kernel<Fwd, Inv, Op>, Fwd::storage_bytes);
    if (e != cudaSuccess) return e;
    fused_kernel<Fwd, Inv, Op><<<static_cast<unsigned>(batch), threads_per_block<Fwd::size>(), Fwd::storage_bytes, stream>>>(d_in, d_out, op);
    return cudaGetLastError();
}

// Three kernels with global-memory round trips through d_tmp (batch*N elements).
template <class Fwd, class Inv, class Op>
cudaError_t execute_unfused(const typename Fwd::complex_type* d_in, typename Fwd::complex_type* d_out,
                            typename Fwd::complex_type* d_tmp, std::size_t batch, Op op, cudaStream_t stream = 0) {
    if (batch == 0) return cudaSuccess;
    cudaError_t e = execute<Fwd>(d_in, d_tmp, batch, stream);
    if (e != cudaSuccess) return e;
    pointwise_kernel<Fwd, Op><<<static_cast<unsigned>(batch), threads_per_block<Fwd::size>(), 0, stream>>>(d_tmp, op);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    return execute<Inv>(d_tmp, d_out, batch, stream);
}

}  // namespace cuda
}  // namespace minifftdx

#endif  // __CUDACC__
