// GPU vs CPU-backend agreement (built only when nvcc is found; skips without a device).
#include <gtest/gtest.h>

#include "minifftdx/backend/cuda.hpp"
#include "reference.hpp"

using namespace minifftdx;
using namespace mfxtest;

namespace {

bool have_device() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

template <class T, unsigned N>
void check() {
    using F = fft_descriptor<Size<N>, Precision<T>, Forward>;
    using I = opposite_t<F>;
    constexpr size_t B = 8;
    auto x = random_signal<T>(N * B, N);
    auto h = random_signal<T>(N, 2 * N);
    std::vector<complex<T>> H(N), cpu_out(N * B), gpu_out(N * B), gpu_unfused(N * B), gpu_fft(N * B), cpu_fft(N * B);
    cpu::execute<F>(h.data(), H.data(), 1);
    const T scale = T(1) / T(N);
    cpu::execute_fused<F, I>(x.data(), cpu_out.data(), B, spectral_multiply<T>{H.data(), scale});
    cpu::execute<F>(x.data(), cpu_fft.data(), B);

    complex<T> *d_in, *d_out, *d_tmp, *d_H;
    cudaMalloc(&d_in, N * B * sizeof(complex<T>));
    cudaMalloc(&d_out, N * B * sizeof(complex<T>));
    cudaMalloc(&d_tmp, N * B * sizeof(complex<T>));
    cudaMalloc(&d_H, N * sizeof(complex<T>));
    cudaMemcpy(d_in, x.data(), N * B * sizeof(complex<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, H.data(), N * sizeof(complex<T>), cudaMemcpyHostToDevice);
    const spectral_multiply<T> op{d_H, scale};

    ASSERT_EQ(cuda::execute<F>(d_in, d_out, B), cudaSuccess);
    cudaMemcpy(gpu_fft.data(), d_out, N * B * sizeof(complex<T>), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cuda::execute_fused<F, I>(d_in, d_out, B, op), cudaSuccess);
    cudaMemcpy(gpu_out.data(), d_out, N * B * sizeof(complex<T>), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cuda::execute_unfused<F, I>(d_in, d_out, d_tmp, B, op), cudaSuccess);
    cudaMemcpy(gpu_unfused.data(), d_out, N * B * sizeof(complex<T>), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); cudaFree(d_H);

    // Same algorithm, different libm for twiddles: agree to a few eps, fused == unfused on the GPU.
    const double tol = 8 * rel_error_bound<T>(F::log2_size);
    auto rel = [&](const std::vector<complex<T>>& a, const std::vector<complex<T>>& b) {
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            num += double(a[i].re - b[i].re) * (a[i].re - b[i].re) + double(a[i].im - b[i].im) * (a[i].im - b[i].im);
            den += double(b[i].re) * b[i].re + double(b[i].im) * b[i].im;
        }
        return std::sqrt(num / den);
    };
    EXPECT_LT(rel(gpu_fft, cpu_fft), tol) << "N=" << N;
    EXPECT_LT(rel(gpu_out, cpu_out), tol) << "N=" << N;
    EXPECT_LT(rel(gpu_unfused, gpu_out), tol) << "N=" << N;
}

}  // namespace

TEST(Cuda, MatchesCpuBackendFloat) {
    if (!have_device()) GTEST_SKIP() << "no CUDA device";
    check<float, 8>(); check<float, 64>(); check<float, 256>(); check<float, 1024>(); check<float, 4096>();
}
TEST(Cuda, MatchesCpuBackendDouble) {
    if (!have_device()) GTEST_SKIP() << "no CUDA device";
    check<double, 16>(); check<double, 128>(); check<double, 2048>(); check<double, 4096>();
}
