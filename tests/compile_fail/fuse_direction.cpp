#include "minifftdx.hpp"
using namespace minifftdx;
using F = fft_descriptor<Size<64>, Precision<float>, Inverse>; using I = fft_descriptor<Size<64>, Precision<float>, Inverse>; void f(complex<float>* x) { cpu::serial_context<float,64> c; fused_fft_op_ifft_block<F, I>(x, scale_by<float>{1.f}, c); }
int main() {}
