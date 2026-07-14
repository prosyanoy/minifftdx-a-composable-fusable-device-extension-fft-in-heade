// Fused vs. unfused FFT convolution on the CUDA backend (same JSON names as the CPU bench).
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <vector>

#include "minifftdx.hpp"
#include "minifftdx/backend/cuda.hpp"

using namespace minifftdx;

namespace {

constexpr std::size_t kTotalPoints = std::size_t(1) << 24;

template <class T, unsigned N, bool Fused>
void BM_Conv(benchmark::State& st) {
    using F = fft_descriptor<Size<N>, Precision<T>, Forward>;
    using I = opposite_t<F>;
    const std::size_t batch = kTotalPoints / N;
    std::vector<complex<T>> h(N, complex<T>(0.5, 0.25));
    complex<T> *d_in, *d_out, *d_tmp, *d_H;
    cudaMalloc(&d_in, kTotalPoints * sizeof(complex<T>));
    cudaMalloc(&d_out, kTotalPoints * sizeof(complex<T>));
    cudaMalloc(&d_tmp, kTotalPoints * sizeof(complex<T>));
    cudaMalloc(&d_H, N * sizeof(complex<T>));
    cudaMemset(d_in, 0, kTotalPoints * sizeof(complex<T>));
    cudaMemcpy(d_H, h.data(), N * sizeof(complex<T>), cudaMemcpyHostToDevice);
    const spectral_multiply<T> op{d_H, T(1) / T(N)};
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (auto _ : st) {
        cudaEventRecord(a);
        if (Fused) cuda::execute_fused<F, I>(d_in, d_out, batch, op);
        else cuda::execute_unfused<F, I>(d_in, d_out, d_tmp, batch, op);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        st.SetIterationTime(ms * 1e-3);
    }
    st.SetItemsProcessed(static_cast<int64_t>(st.iterations() * batch));
    st.counters["GFLOPS"] = benchmark::Counter(2.0 * F::flops * batch * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); cudaFree(d_H);
}

#define REG(T, N)                                                                                \
    BENCHMARK_TEMPLATE(BM_Conv, T, N, true)->Name("conv_fused/" #T "/" #N)->UseManualTime();     \
    BENCHMARK_TEMPLATE(BM_Conv, T, N, false)->Name("conv_unfused/" #T "/" #N)->UseManualTime();
#define REG_ALL(T) REG(T, 64) REG(T, 256) REG(T, 1024) REG(T, 4096)
REG_ALL(float)
REG_ALL(double)

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::fprintf(stderr, "no CUDA device; skipping GPU benchmarks\n");
        return 0;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
