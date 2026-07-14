#include "minifftdx.hpp"
using namespace minifftdx;
struct Warp {}; using Bad = fft_descriptor<Size<64>, Precision<float>, Forward, Warp>; static constexpr unsigned s = Bad::size;
int main() {}
