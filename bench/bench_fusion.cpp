// Fused vs. unfused FFT convolution on the CPU backend.
// Working set (2^22 complex points) is far larger than cache, so the unfused
// version pays real memory traffic for its two intermediate round trips.
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "minifftdx.hpp"

using namespace minifftdx;

namespace {

constexpr std::size_t kTotalPoints = std::size_t(1) << 22;

template <class T, unsigned N>
struct Fixture {
    using F = fft_descriptor<Size<N>, Precision<T>, Forward>;
    using I = opposite_t<F>;
    std::size_t batch = kTotalPoints / N;
    std::vector<complex<T>> in, out, tmp, H;
    Fixture() : in(kTotalPoints), out(kTotalPoints), tmp(kTotalPoints), H(N) {
        std::mt19937 g(1);
        std::uniform_real_distribution<float> d(-1, 1);
        for (auto& c : in) c = {T(d(g)), T(d(g))};
        for (auto& c : H) c = {T(d(g)), T(d(g))};
    }
};

template <class T, unsigned N, bool Fused>
void BM_Conv(benchmark::State& st) {
    Fixture<T, N> f;
    const spectral_multiply<T> op{f.H.data(), T(1) / T(N)};
    for (auto _ : st) {
        if (Fused) cpu::execute_fused<typename Fixture<T, N>::F, typename Fixture<T, N>::I>(f.in.data(), f.out.data(), f.batch, op);
        else cpu::execute_unfused<typename Fixture<T, N>::F, typename Fixture<T, N>::I>(f.in.data(), f.out.data(), f.tmp.data(), f.batch, op);
        benchmark::DoNotOptimize(f.out.data());
        benchmark::ClobberMemory();
    }
    st.SetItemsProcessed(static_cast<int64_t>(st.iterations() * f.batch));  // convolutions
    st.counters["GFLOPS"] = benchmark::Counter(
        2.0 * Fixture<T, N>::F::flops * f.batch * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
}

#define REG(T, N)                                                                        \
    BENCHMARK_TEMPLATE(BM_Conv, T, N, true)->Name("conv_fused/" #T "/" #N)->UseRealTime(); \
    BENCHMARK_TEMPLATE(BM_Conv, T, N, false)->Name("conv_unfused/" #T "/" #N)->UseRealTime();
#define REG_ALL(T) REG(T, 64) REG(T, 256) REG(T, 1024) REG(T, 4096)
REG_ALL(float)
REG_ALL(double)

}  // namespace

BENCHMARK_MAIN();
