#pragma once
// Fusion: compose an FFT, a user functor, and an inverse FFT inside one block
// so the data never leaves fast memory (CUDA shared memory / CPU L1) between
// the stages.
#include <type_traits>

#include "fft.hpp"

namespace minifftdx {

// Apply `op(value, index) -> value` to every element of the block, in place.
template <class Desc, class Op, class Ctx>
MFX_HD void apply_pointwise(typename Desc::complex_type* x, Op op, const Ctx& ctx) {
    for (unsigned k = ctx.tid(); k < Desc::size; k += ctx.nthreads()) x[k] = op(x[k], k);
    ctx.sync();
}

// forward FFT -> op(spectrum bin, bin index) -> inverse FFT, all in place.
// The inverse is unnormalised; fold any 1/N into `op` (see spectral_multiply).
template <class Fwd, class Inv, class Op, class Ctx>
MFX_HD void fused_fft_op_ifft_block(typename Fwd::complex_type* x, Op op, const Ctx& ctx) {
    static_assert(Fwd::direction == fft_direction::forward, "minifftdx: first descriptor of a fused pipeline must be forward");
    static_assert(Inv::direction == fft_direction::inverse, "minifftdx: second descriptor of a fused pipeline must be inverse");
    static_assert(Fwd::size == Inv::size, "minifftdx: fused FFT descriptors must have the same size");
    static_assert(std::is_same<typename Fwd::value_type, typename Inv::value_type>::value,
                  "minifftdx: fused FFT descriptors must have the same precision");
    fft_block<Fwd>(x, ctx);  // fft_block ends with a barrier
    apply_pointwise<Fwd>(x, op, ctx);
    fft_block<Inv>(x, ctx);
}

// ---- Ready-made functors ---------------------------------------------------

// X[k] -> X[k] * H[k] * scale. With scale = 1/N this turns the fused pipeline
// into a circular (FFT) convolution with the signal whose spectrum is H.
template <class T>
struct spectral_multiply {
    const complex<T>* H;
    T scale;
    MFX_HD complex<T> operator()(complex<T> v, unsigned k) const { return (v * H[k]) * scale; }
};

// X[k] -> X[k] * scale (e.g. the 1/N normalisation of a round trip).
template <class T>
struct scale_by {
    T scale;
    MFX_HD complex<T> operator()(complex<T> v, unsigned) const { return v * scale; }
};

}  // namespace minifftdx
