// FFT vs. a naive long-double DFT across sizes 2..4096, float and double,
// forward and inverse, with relative-error / ULP reporting.
#include <gtest/gtest.h>

#include "reference.hpp"

using namespace minifftdx;
using namespace mfxtest;

namespace {

template <class T, fft_direction D>
void run_against_naive() {
    for_sizes(
        [&](auto sz) {
            constexpr unsigned N = decltype(sz)::value;
            using Desc = fft_descriptor<Size<N>, Precision<T>, Direction<D>, Block>;
            auto x = random_signal<T>(N, 1234 + N);
            std::vector<complex<T>> y(N);
            cpu::execute<Desc>(x.data(), y.data(), 1);
            const auto ref = naive_dft<T>(x, D == fft_direction::forward ? -1 : +1);
            const Stats s = compare<T>(y, ref);
            rows().push_back({"naive", std::is_same<T, float>::value ? "float" : "double",
                              D == fft_direction::forward ? "forward" : "inverse", N, s});
            EXPECT_LT(s.rel_l2, rel_error_bound<T>(Desc::log2_size)) << "N=" << N;
        },
        AllSizes{});
}

}  // namespace

TEST(AccuracyNaive, FloatForward) { run_against_naive<float, fft_direction::forward>(); }
TEST(AccuracyNaive, FloatInverse) { run_against_naive<float, fft_direction::inverse>(); }
TEST(AccuracyNaive, DoubleForward) { run_against_naive<double, fft_direction::forward>(); }
TEST(AccuracyNaive, DoubleInverse) { run_against_naive<double, fft_direction::inverse>(); }

TEST(Accuracy, LargestSizes) {
    using D = fft_descriptor<Size<16384>, Precision<double>, Forward>;
    auto x = random_signal<double>(16384, 7);
    std::vector<complex<double>> y(16384);
    cpu::execute<D>(x.data(), y.data(), 1);
    // Check a handful of bins against the direct sum instead of the full O(N^2) DFT.
    const ld two_pi = 6.283185307179586476925286766559005768L;
    for (size_t k : {0u, 1u, 5000u, 8192u, 16383u}) {
        cld acc = 0;
        for (size_t n = 0; n < 16384; ++n) {
            const ld a = two_pi * static_cast<ld>((k * n) % 16384) / 16384.0L;
            acc += cld(x[n].re, x[n].im) * cld(std::cos(a), -std::sin(a));
        }
        EXPECT_NEAR(y[k].re, static_cast<double>(acc.real()), 1e-9);
        EXPECT_NEAR(y[k].im, static_cast<double>(acc.imag()), 1e-9);
    }
}

TEST(Structure, ImpulseGivesFlatSpectrum) {
    using D = fft_descriptor<Size<64>, Precision<float>, Forward>;
    std::vector<complex<float>> x(64), y(64);
    x[0] = {1.f, 0.f};
    cpu::execute<D>(x.data(), y.data(), 1);
    for (auto v : y) { EXPECT_FLOAT_EQ(v.re, 1.f); EXPECT_FLOAT_EQ(v.im, 0.f); }
}

TEST(Structure, ToneLandsInOneBin) {
    constexpr unsigned N = 256, K = 17;
    using D = fft_descriptor<Size<N>, Precision<double>, Forward>;
    std::vector<complex<double>> x(N), y(N);
    for (unsigned n = 0; n < N; ++n) {
        const double a = 2.0 * M_PI * K * n / N;
        x[n] = {std::cos(a), std::sin(a)};
    }
    cpu::execute<D>(x.data(), y.data(), 1);
    for (unsigned k = 0; k < N; ++k) {
        EXPECT_NEAR(y[k].re, k == K ? double(N) : 0.0, 1e-10);
        EXPECT_NEAR(y[k].im, 0.0, 1e-10);
    }
}

TEST(Structure, RoundTripScalesByN) {
    for_sizes(
        [&](auto sz) {
            constexpr unsigned N = decltype(sz)::value;
            using F = fft_descriptor<Size<N>, Precision<double>, Forward>;
            using I = opposite_t<F>;
            auto x = random_signal<double>(N, 99 + N);
            std::vector<complex<double>> y(N), z(N);
            cpu::execute<F>(x.data(), y.data(), 1);
            cpu::execute<I>(y.data(), z.data(), 1);
            for (unsigned i = 0; i < N; ++i) {
                EXPECT_NEAR(z[i].re / N, x[i].re, 1e-12) << "N=" << N;
                EXPECT_NEAR(z[i].im / N, x[i].im, 1e-12) << "N=" << N;
            }
        },
        AllSizes{});
}

TEST(Structure, ParsevalHolds) {
    constexpr unsigned N = 1024;
    using F = fft_descriptor<Size<N>, Precision<double>, Forward>;
    auto x = random_signal<double>(N, 5);
    std::vector<complex<double>> y(N);
    cpu::execute<F>(x.data(), y.data(), 1);
    double ex = 0, ey = 0;
    for (unsigned i = 0; i < N; ++i) { ex += x[i].re * x[i].re + x[i].im * x[i].im; ey += y[i].re * y[i].re + y[i].im * y[i].im; }
    EXPECT_NEAR(ey / N, ex, 1e-9 * ex);
}

TEST(Helpers, UlpDistance) {
    EXPECT_EQ(ulp_distance<float>(1.f, 1.f), 0u);
    EXPECT_EQ(ulp_distance<float>(1.f, std::nextafter(1.f, 2.f)), 1u);
    EXPECT_EQ(ulp_distance<double>(-0.0, 0.0), 0u);
    EXPECT_EQ(ulp_distance<double>(-std::nextafter(0.0, 1.0), std::nextafter(0.0, 1.0)), 2u);
}
