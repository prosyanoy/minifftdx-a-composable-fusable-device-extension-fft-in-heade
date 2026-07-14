#pragma once
#include <cstddef>
#include <type_traits>

#include "complex.hpp"

namespace minifftdx {

// ---- Descriptor operators (cuFFTDx style) ---------------------------------

enum class fft_direction { forward, inverse };

template <unsigned N>
struct Size { static constexpr unsigned value = N; };

template <class T>
struct Precision { using type = T; };

template <fft_direction D>
struct Direction { static constexpr fft_direction value = D; };

// Execution scope: one FFT is computed cooperatively by one "block" -
// a CUDA thread block on the GPU, one OpenMP iteration on the CPU.
struct Block {};

// Convenience aliases.
using Forward = Direction<fft_direction::forward>;
using Inverse = Direction<fft_direction::inverse>;

namespace detail {

template <class> struct is_size : std::false_type {};
template <unsigned N> struct is_size<Size<N>> : std::true_type {};

template <class> struct is_precision : std::false_type {};
template <class T> struct is_precision<Precision<T>> : std::true_type {};

template <class> struct is_direction : std::false_type {};
template <fft_direction D> struct is_direction<Direction<D>> : std::true_type {};

// Extractors that degrade gracefully, so a wrong template argument yields one
// readable static_assert message instead of a cascade of errors.
template <class S> struct size_of { static constexpr unsigned value = 0; };
template <unsigned N> struct size_of<Size<N>> { static constexpr unsigned value = N; };

template <class P> struct precision_of { using type = float; };
template <class T> struct precision_of<Precision<T>> { using type = T; };

template <class D> struct direction_of { static constexpr fft_direction value = fft_direction::forward; };
template <fft_direction D> struct direction_of<Direction<D>> { static constexpr fft_direction value = D; };

constexpr bool is_pow2(unsigned n) { return n != 0 && (n & (n - 1)) == 0; }
constexpr unsigned ilog2(unsigned n) { unsigned l = 0; while (n > 1) { n >>= 1; ++l; } return l; }

}  // namespace detail

// Supported transform sizes (powers of two).
inline constexpr unsigned min_fft_size = 2;
inline constexpr unsigned max_fft_size = 16384;

// Compile-time description of an FFT. All invalid configurations are rejected
// with a static_assert at the point of instantiation.
template <class SizeOp, class PrecisionOp, class DirectionOp, class ExecOp = Block>
struct fft_descriptor {
    static_assert(detail::is_size<SizeOp>::value, "minifftdx: first argument must be Size<N>");
    static_assert(detail::is_precision<PrecisionOp>::value, "minifftdx: second argument must be Precision<float|double>");
    static_assert(detail::is_direction<DirectionOp>::value, "minifftdx: third argument must be Direction<fft_direction::forward|inverse>");
    static_assert(std::is_same<ExecOp, Block>::value, "minifftdx: only the Block execution scope is supported");

    using value_type = typename detail::precision_of<PrecisionOp>::type;
    static_assert(std::is_same<value_type, float>::value || std::is_same<value_type, double>::value,
                  "minifftdx: precision must be float or double");

    static constexpr unsigned size = detail::size_of<SizeOp>::value;
    static_assert(detail::is_pow2(size), "minifftdx: FFT size must be a power of two");
    static_assert(size >= min_fft_size && size <= max_fft_size,
                  "minifftdx: FFT size must be in the range [2, 16384]");

    static constexpr fft_direction direction = detail::direction_of<DirectionOp>::value;
    static constexpr unsigned log2_size = detail::ilog2(size);
    // Number of radix-4 passes (two radix-2 stages fused) and radix-2 passes.
    static constexpr unsigned radix2_passes = log2_size & 1u;
    static constexpr unsigned radix4_passes = log2_size / 2;

    using complex_type = complex<value_type>;
    using size_type = SizeOp;
    using precision_type = PrecisionOp;
    using direction_type = DirectionOp;
    using execution_type = ExecOp;

    // Bytes of working memory (CUDA shared memory) one FFT needs.
    static constexpr std::size_t storage_bytes = sizeof(complex_type) * size;
    // Conventional 5 N log2 N flop count of a complex FFT.
    static constexpr double flops = 5.0 * size * log2_size;
};

// The descriptor with the opposite direction and everything else unchanged.
template <class Desc>
using opposite_t = fft_descriptor<typename Desc::size_type, typename Desc::precision_type,
                                  Direction<Desc::direction == fft_direction::forward ? fft_direction::inverse
                                                                                       : fft_direction::forward>,
                                  typename Desc::execution_type>;

}  // namespace minifftdx
