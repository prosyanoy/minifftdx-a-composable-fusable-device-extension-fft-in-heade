#pragma once
// Backend-agnostic block FFT.
//
// The algorithm is written exactly once and is parameterised on an execution
// context `Ctx` supplied by the backend:
//
//   unsigned tid() const          index of this worker inside the block
//   unsigned nthreads() const     number of cooperating workers
//   void     sync() const         barrier between dependent phases
//   complex<T> twiddle(k) const   exp(-2*pi*i*k/N) for 0 <= k < N/2
//
// CUDA: tid/nthreads = threadIdx.x/blockDim.x, sync = __syncthreads().
// CPU : one worker per FFT (nthreads = 1, sync is a no-op); parallelism comes
//       from running many independent FFTs through OpenMP.
//
// The transform is in place on `x`, which holds Desc::size elements (shared
// memory on the GPU, a thread-local buffer on the CPU). Output is in natural
// order. The inverse transform is unnormalised (as in cuFFT / cuFFTDx), i.e.
// inverse(forward(x)) == N * x.
#include "descriptor.hpp"

namespace minifftdx {

namespace detail {

MFX_HD MFX_INLINE unsigned bit_reverse(unsigned v, unsigned bits) {
    unsigned r = 0;
    for (unsigned b = 0; b < bits; ++b) {
        r = (r << 1) | (v & 1u);
        v >>= 1;
    }
    return r;
}

// Twiddle for the requested direction (forward table, conjugated for inverse).
template <class Desc, class Ctx>
MFX_HD MFX_INLINE typename Desc::complex_type directed_twiddle(const Ctx& ctx, unsigned k) {
    auto w = ctx.twiddle(k);
    if (Desc::direction == fft_direction::inverse) w.im = -w.im;
    return w;
}

// Multiply by -i (forward) or +i (inverse): the radix-4 "trivial" rotation.
template <class Desc>
MFX_HD MFX_INLINE typename Desc::complex_type rotate_quarter(typename Desc::complex_type v) {
    if (Desc::direction == fft_direction::forward) return {v.im, -v.re};
    return {-v.im, v.re};
}

}  // namespace detail

// In-place mixed-radix (2/4) decimation-in-time FFT of one block.
//
// Phase 0: bit-reversal permutation.
// Phase 1: if log2(N) is odd, one radix-2 stage (twiddle-free).
// Phase 2: radix-4 stages; each fuses two radix-2 stages into a single pass
//          over the data, halving the number of barriers and shared-memory
//          sweeps compared to pure radix-2.
template <class Desc, class Ctx>
MFX_HD void fft_block(typename Desc::complex_type* x, const Ctx& ctx) {
    constexpr unsigned N = Desc::size;
    const unsigned tid = ctx.tid();
    const unsigned nt = ctx.nthreads();

    for (unsigned i = tid; i < N; i += nt) {
        const unsigned j = detail::bit_reverse(i, Desc::log2_size);
        if (i < j) {
            const auto t = x[i];
            x[i] = x[j];
            x[j] = t;
        }
    }
    ctx.sync();

    unsigned h = 1;  // half-size of the first butterfly stage handled below
    if (Desc::radix2_passes) {
        for (unsigned t = tid; t < N / 2; t += nt) {
            const auto a = x[2 * t];
            const auto b = x[2 * t + 1];
            x[2 * t] = a + b;
            x[2 * t + 1] = a - b;
        }
        ctx.sync();
        h = 2;
    }

    for (unsigned pass = 0; pass < Desc::radix4_passes; ++pass, h *= 4) {
        const unsigned s1 = N / (2 * h);  // table stride of W_{2h}
        const unsigned s2 = N / (4 * h);  // table stride of W_{4h}
        for (unsigned t = tid; t < N / 4; t += nt) {
            const unsigned j = t & (h - 1);
            const unsigned i0 = (t - j) * 4 + j;
            const unsigned i1 = i0 + h, i2 = i1 + h, i3 = i2 + h;
            const auto w1 = detail::directed_twiddle<Desc>(ctx, j * s1);
            const auto w2 = detail::directed_twiddle<Desc>(ctx, j * s2);

            // First fused stage (half-size h), twiddle W_{2h}^j.
            const auto a0 = x[i0];
            const auto a1 = x[i1] * w1;
            const auto a2 = x[i2];
            const auto a3 = x[i3] * w1;
            const auto b0 = a0 + a1, b1 = a0 - a1;
            const auto b2 = a2 + a3, b3 = a2 - a3;

            // Second fused stage (half-size 2h): W_{4h}^j and W_{4h}^{j+h} = W_{4h}^j * (-+i).
            const auto c2 = b2 * w2;
            const auto c3 = detail::rotate_quarter<Desc>(b3 * w2);
            x[i0] = b0 + c2;
            x[i2] = b0 - c2;
            x[i1] = b1 + c3;
            x[i3] = b1 - c3;
        }
        ctx.sync();
    }
}

}  // namespace minifftdx
