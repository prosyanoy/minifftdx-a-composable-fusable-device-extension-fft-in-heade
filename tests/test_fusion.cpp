#include <gtest/gtest.h>

#include "reference.hpp"

using namespace minifftdx;
using namespace mfxtest;

namespace {

// Direct O(N^2) circular convolution in long double.
template <class T>
std::vector<cld> circular_convolution(const std::vector<complex<T>>& x, const std::vector<complex<T>>& h) {
    const size_t N = x.size();
    std::vector<cld> y(N);
    for (size_t k = 0; k < N; ++k) {
        cld acc = 0;
        for (size_t n = 0; n < N; ++n) acc += cld(x[n].re, x[n].im) * cld(h[(k + N - n) % N].re, h[(k + N - n) % N].im);
        y[k] = acc;
    }
    return y;
}

template <class T>
void fused_convolution_matches_direct() {
    for_sizes(
        [&](auto sz) {
            constexpr unsigned N = decltype(sz)::value;
            using F = fft_descriptor<Size<N>, Precision<T>, Forward>;
            using I = opposite_t<F>;
            auto x = random_signal<T>(N, 21 + N);
            auto h = random_signal<T>(N, 77 + N);
            std::vector<complex<T>> H(N), y(N);
            cpu::execute<F>(h.data(), H.data(), 1);
            cpu::execute_fused<F, I>(x.data(), y.data(), 1, spectral_multiply<T>{H.data(), T(1) / T(N)});
            const Stats s = compare<T>(y, circular_convolution<T>(x, h));
            rows().push_back({"fused", std::is_same<T, float>::value ? "float" : "double", "conv", N, s});
            EXPECT_LT(s.rel_l2, 4 * rel_error_bound<T>(F::log2_size)) << "N=" << N;
        },
        std::integer_sequence<unsigned, 2, 8, 16, 64, 256, 1024, 4096>{});
}

}  // namespace

TEST(Fusion, ConvolutionFloat) { fused_convolution_matches_direct<float>(); }
TEST(Fusion, ConvolutionDouble) { fused_convolution_matches_direct<double>(); }

TEST(Fusion, FusedIsBitIdenticalToUnfused) {
    constexpr unsigned N = 512;
    constexpr size_t B = 20;
    using F = fft_descriptor<Size<N>, Precision<float>, Forward>;
    using I = opposite_t<F>;
    auto x = random_signal<float>(N * B, 4);
    auto h = random_signal<float>(N, 5);
    std::vector<complex<float>> H(N), fused(N * B), unfused(N * B), tmp(N * B);
    cpu::execute<F>(h.data(), H.data(), 1);
    const spectral_multiply<float> op{H.data(), 1.f / N};
    cpu::execute_fused<F, I>(x.data(), fused.data(), B, op);
    cpu::execute_unfused<F, I>(x.data(), unfused.data(), tmp.data(), B, op);
    for (size_t i = 0; i < fused.size(); ++i) ASSERT_EQ(fused[i], unfused[i]) << i;
}

TEST(Fusion, LambdaFilterZeroesHighFrequencies) {
    constexpr unsigned N = 256, K = 4, HI = 100;
    using F = fft_descriptor<Size<N>, Precision<double>, Forward>;
    using I = opposite_t<F>;
    std::vector<complex<double>> x(N), y(N);
    for (unsigned n = 0; n < N; ++n) {
        const double lo = 2 * M_PI * K * n / N, hi = 2 * M_PI * HI * n / N;
        x[n] = {std::cos(lo) + std::cos(hi), 0.0};
    }
    // Band-limit: keep |frequency| <= 10 bins (bins near 0 and N), fold in 1/N.
    auto lowpass = [](complex<double> v, unsigned k) {
        return (k <= 10 || k >= N - 10) ? v * (1.0 / N) : complex<double>{};
    };
    cpu::execute_fused<F, I>(x.data(), y.data(), 1, lowpass);
    for (unsigned n = 0; n < N; ++n) EXPECT_NEAR(y[n].re, std::cos(2 * M_PI * K * n / N), 1e-12);
}

TEST(Fusion, ScaleFunctorRoundTripIsIdentity) {
    constexpr unsigned N = 1024;
    using F = fft_descriptor<Size<N>, Precision<double>, Forward>;
    using I = opposite_t<F>;
    auto x = random_signal<double>(N, 8);
    std::vector<complex<double>> y(N);
    cpu::execute_fused<F, I>(x.data(), y.data(), 1, scale_by<double>{1.0 / N});
    for (unsigned i = 0; i < N; ++i) { EXPECT_NEAR(y[i].re, x[i].re, 1e-13); EXPECT_NEAR(y[i].im, x[i].im, 1e-13); }
}
