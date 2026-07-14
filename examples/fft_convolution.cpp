// FFT convolution: forward FFT -> multiply by a filter spectrum -> inverse FFT,
// fused into a single pass per signal.
#include <cstdio>
#include <vector>

#include "minifftdx.hpp"

using namespace minifftdx;

int main() {
    constexpr unsigned N = 1024;
    using FFT  = fft_descriptor<Size<N>, Precision<float>, Direction<fft_direction::forward>, Block>;
    using IFFT = fft_descriptor<Size<N>, Precision<float>, Direction<fft_direction::inverse>, Block>;

    std::vector<complex<float>> x(N), h(N), H(N), y(N);
    x[3] = {1.f, 0.f};                          // impulse at n = 3
    h[0] = {0.5f, 0.f}; h[1] = {0.25f, 0.f};    // 2-tap smoothing filter
    cpu::execute<FFT>(h.data(), H.data(), 1);   // filter spectrum, computed once

    cpu::execute_fused<FFT, IFFT>(x.data(), y.data(), 1, spectral_multiply<float>{H.data(), 1.f / N});

    std::printf("y[2..5] = %.3f %.3f %.3f %.3f  (expected 0 0.5 0.25 0)\n", y[2].re, y[3].re, y[4].re, y[5].re);
    return (y[3].re > 0.49f && y[3].re < 0.51f) ? 0 : 1;
}
