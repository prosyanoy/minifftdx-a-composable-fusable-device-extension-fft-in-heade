#pragma once
// CPU backend: runs the shared block algorithm once per FFT; independent FFTs
// of a batch are distributed over threads with OpenMP (if available).
#include <cmath>
#include <cstddef>
#include <vector>

#include "../fusion.hpp"

namespace minifftdx {
namespace cpu {

namespace detail {

// exp(-2*pi*i*k/N), k in [0, N/2), computed in long double and rounded once.
template <class T, unsigned N>
const complex<T>* twiddle_table() {
    static const std::vector<complex<T>> table = [] {
        std::vector<complex<T>> t(N / 2 > 0 ? N / 2 : 1);
        const long double two_pi = 6.283185307179586476925286766559005768L;
        for (unsigned k = 0; k < N / 2; ++k) {
            const long double a = two_pi * static_cast<long double>(k) / static_cast<long double>(N);
            t[k] = complex<T>(static_cast<T>(std::cos(a)), static_cast<T>(-std::sin(a)));
        }
        return t;
    }();
    return table.data();
}

}  // namespace detail

// Execution context for a single FFT on one CPU thread.
template <class T, unsigned N>
struct serial_context {
    const complex<T>* table = detail::twiddle_table<T, N>();
    unsigned tid() const { return 0; }
    unsigned nthreads() const { return 1; }
    void sync() const {}
    complex<T> twiddle(unsigned k) const { return table[k]; }
};

template <class Desc>
using context_for = serial_context<typename Desc::value_type, Desc::size>;

// out[b] = FFT(in[b]) for b in [0, batch). `in` and `out` may alias.
template <class Desc>
void execute(const typename Desc::complex_type* in, typename Desc::complex_type* out, std::size_t batch) {
    constexpr unsigned N = Desc::size;
    const context_for<Desc> ctx;
    const long long nb = static_cast<long long>(batch);
#pragma omp parallel
    {
        std::vector<typename Desc::complex_type> buf(N);
#pragma omp for schedule(static)
        for (long long b = 0; b < nb; ++b) {
            const auto* src = in + static_cast<std::size_t>(b) * N;
            auto* dst = out + static_cast<std::size_t>(b) * N;
            for (unsigned i = 0; i < N; ++i) buf[i] = src[i];
            fft_block<Desc>(buf.data(), ctx);
            for (unsigned i = 0; i < N; ++i) dst[i] = buf[i];
        }
    }
}

// Standalone pointwise pass over `batch` blocks of Desc::size elements (the
// "unfused" middle step: one extra full sweep over memory).
template <class Desc, class Op>
void apply_pointwise_pass(typename Desc::complex_type* data, std::size_t batch, Op op) {
    constexpr unsigned N = Desc::size;
    const long long nb = static_cast<long long>(batch);
#pragma omp parallel for schedule(static)
    for (long long b = 0; b < nb; ++b) {
        auto* p = data + static_cast<std::size_t>(b) * N;
        for (unsigned k = 0; k < N; ++k) p[k] = op(p[k], k);
    }
}

// Fused forward FFT -> op -> inverse FFT: each signal is loaded once, kept in a
// thread-local buffer for all three phases, and stored once.
template <class Fwd, class Inv, class Op>
void execute_fused(const typename Fwd::complex_type* in, typename Fwd::complex_type* out, std::size_t batch, Op op) {
    constexpr unsigned N = Fwd::size;
    const context_for<Fwd> ctx;
    const long long nb = static_cast<long long>(batch);
#pragma omp parallel
    {
        std::vector<typename Fwd::complex_type> buf(N);
#pragma omp for schedule(static)
        for (long long b = 0; b < nb; ++b) {
            const auto* src = in + static_cast<std::size_t>(b) * N;
            auto* dst = out + static_cast<std::size_t>(b) * N;
            for (unsigned i = 0; i < N; ++i) buf[i] = src[i];
            fused_fft_op_ifft_block<Fwd, Inv>(buf.data(), op, ctx);
            for (unsigned i = 0; i < N; ++i) dst[i] = buf[i];
        }
    }
}

// The same computation as three separate passes through `tmp` (which must hold
// batch * N elements); `out` may alias `in`. Numerically identical to
// execute_fused, but with a global-memory round trip between stages.
template <class Fwd, class Inv, class Op>
void execute_unfused(const typename Fwd::complex_type* in, typename Fwd::complex_type* out,
                     typename Fwd::complex_type* tmp, std::size_t batch, Op op) {
    execute<Fwd>(in, tmp, batch);
    apply_pointwise_pass<Fwd>(tmp, batch, op);
    execute<Inv>(tmp, out, batch);
}

}  // namespace cpu
}  // namespace minifftdx
