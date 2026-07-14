#include <gtest/gtest.h>

#include <type_traits>

#include "minifftdx.hpp"

using namespace minifftdx;

using F1024 = fft_descriptor<Size<1024>, Precision<float>, Direction<fft_direction::forward>, Block>;
using I8 = fft_descriptor<Size<8>, Precision<double>, Inverse, Block>;

static_assert(F1024::size == 1024 && F1024::log2_size == 10);
static_assert(F1024::radix2_passes == 0 && F1024::radix4_passes == 5);
static_assert(I8::radix2_passes == 1 && I8::radix4_passes == 1);
static_assert(std::is_same<F1024::value_type, float>::value);
static_assert(I8::direction == fft_direction::inverse);
static_assert(F1024::storage_bytes == 1024 * 8);
static_assert(std::is_same<opposite_t<F1024>::value_type, float>::value);
static_assert(opposite_t<F1024>::direction == fft_direction::inverse);
static_assert(opposite_t<opposite_t<I8>>::direction == fft_direction::inverse);

TEST(Descriptor, CompileTimeProperties) {
    EXPECT_EQ(F1024::size, 1024u);
    EXPECT_DOUBLE_EQ(F1024::flops, 5.0 * 1024 * 10);
    EXPECT_EQ(sizeof(F1024::complex_type), 8u);
}

TEST(Descriptor, SupportedRangeBoundaries) {
    using Min = fft_descriptor<Size<2>, Precision<float>, Forward>;
    using Max = fft_descriptor<Size<16384>, Precision<double>, Inverse>;
    EXPECT_EQ(Min::size, 2u);
    EXPECT_EQ(Max::size, 16384u);
}
