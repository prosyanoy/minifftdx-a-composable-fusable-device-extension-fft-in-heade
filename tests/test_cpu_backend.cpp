#include <gtest/gtest.h>

#include "reference.hpp"

using namespace minifftdx;
using namespace mfxtest;

TEST(CpuBackend, BatchMatchesSingleTransforms) {
    constexpr unsigned N = 128;
    constexpr size_t B = 37;
    using D = fft_descriptor<Size<N>, Precision<float>, Forward>;
    auto x = random_signal<float>(N * B, 3);
    std::vector<complex<float>> batched(N * B);
    cpu::execute<D>(x.data(), batched.data(), B);
    for (size_t b = 0; b < B; ++b) {
        std::vector<complex<float>> one(N);
        cpu::execute<D>(x.data() + b * N, one.data(), 1);
        for (unsigned i = 0; i < N; ++i) EXPECT_EQ(one[i], batched[b * N + i]);
    }
}

TEST(CpuBackend, InPlaceAliasing) {
    constexpr unsigned N = 64;
    using D = fft_descriptor<Size<N>, Precision<double>, Forward>;
    auto x = random_signal<double>(N * 4, 11);
    std::vector<complex<double>> out(N * 4), inplace = x;
    cpu::execute<D>(x.data(), out.data(), 4);
    cpu::execute<D>(inplace.data(), inplace.data(), 4);
    ASSERT_EQ(out.size(), inplace.size());
    for (size_t i = 0; i < out.size(); ++i) EXPECT_EQ(out[i], inplace[i]) << "in-place result must match out-of-place";
}

TEST(CpuBackend, ZeroBatchIsNoop) {
    using D = fft_descriptor<Size<8>, Precision<float>, Forward>;
    cpu::execute<D>(nullptr, nullptr, 0);
    SUCCEED();
}
