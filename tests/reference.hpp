#pragma once
// Test helpers: naive DFT, error metrics (relative L2, ULP) and a report table.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "minifftdx.hpp"

namespace mfxtest {

using ld = long double;
using cld = std::complex<long double>;

// O(N^2) DFT in long double. sign = -1 forward, +1 inverse (unnormalised).
template <class T>
std::vector<cld> naive_dft(const std::vector<minifftdx::complex<T>>& x, int sign) {
    const size_t N = x.size();
    const ld two_pi = 6.283185307179586476925286766559005768L;
    std::vector<cld> w(N);
    for (size_t m = 0; m < N; ++m) {
        const ld a = two_pi * static_cast<ld>(m) / static_cast<ld>(N);
        w[m] = cld(std::cos(a), sign * std::sin(a));
    }
    std::vector<cld> y(N);
    for (size_t k = 0; k < N; ++k) {
        cld acc = 0;
        for (size_t n = 0; n < N; ++n) acc += cld(x[n].re, x[n].im) * w[(k * n) % N];
        y[k] = acc;
    }
    return y;
}

template <class T>
std::vector<minifftdx::complex<T>> random_signal(size_t n, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<minifftdx::complex<T>> v(n);
    for (auto& c : v) { c.re = static_cast<T>(d(g)); c.im = static_cast<T>(d(g)); }
    return v;
}

// Distance in representable values between a and b (same type).
template <class T>
std::uint64_t ulp_distance(T a, T b) {
    using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    using S = std::conditional_t<sizeof(T) == 4, std::int32_t, std::int64_t>;
    auto ordered = [](T v) {
        U u;
        std::memcpy(&u, &v, sizeof(T));
        const U sign = U(1) << (sizeof(T) * 8 - 1);
        return (u & sign) ? static_cast<S>(-static_cast<S>(u & ~sign)) : static_cast<S>(u);
    };
    const S x = ordered(a), y = ordered(b);
    return x > y ? static_cast<std::uint64_t>(x) - static_cast<std::uint64_t>(y)
                 : static_cast<std::uint64_t>(y) - static_cast<std::uint64_t>(x);
}

struct Stats {
    double rel_l2 = 0;      // ||got - ref||_2 / ||ref||_2
    double max_ulp_norm = 0;  // max |err| / ulp(max|ref|): normwise error in ULPs
    std::uint64_t max_ulp_elem = 0;  // max per-component ULP distance over significant components
};

// `ref` is the high-precision reference (long double). Per-component ULP is only
// taken over components with |ref| >= 1% of the peak, where it is meaningful
// (near-zero components of a sum have unbounded relative error by nature).
template <class T>
Stats compare(const std::vector<minifftdx::complex<T>>& got, const std::vector<cld>& ref) {
    ld num = 0, den = 0, peak = 0, maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const cld g(got[i].re, got[i].im);
        num += std::norm(g - ref[i]);
        den += std::norm(ref[i]);
        peak = std::max({peak, std::fabs(ref[i].real()), std::fabs(ref[i].imag())});
        maxerr = std::max({maxerr, std::fabs(g.real() - ref[i].real()), std::fabs(g.imag() - ref[i].imag())});
    }
    Stats s;
    s.rel_l2 = static_cast<double>(std::sqrt(num / (den > 0 ? den : 1)));
    const T pk = static_cast<T>(peak);
    const ld ulp = static_cast<ld>(std::nextafter(pk, std::numeric_limits<T>::infinity())) - static_cast<ld>(pk);
    s.max_ulp_norm = ulp > 0 ? static_cast<double>(maxerr / ulp) : 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const T rr = static_cast<T>(ref[i].real()), ri = static_cast<T>(ref[i].imag());
        if (std::fabs(ref[i].real()) >= 0.01L * peak) s.max_ulp_elem = std::max(s.max_ulp_elem, ulp_distance<T>(got[i].re, rr));
        if (std::fabs(ref[i].imag()) >= 0.01L * peak) s.max_ulp_elem = std::max(s.max_ulp_elem, ulp_distance<T>(got[i].im, ri));
    }
    return s;
}

// FFTs accumulate roughly eps * O(log N) relative L2 error; a bug gives >= 1e-3.
template <class T>
double rel_error_bound(unsigned log2n) {
    return 2.0 * std::numeric_limits<T>::epsilon() * (log2n + 1);
}

struct Row { std::string ref, type, dir; unsigned n; Stats s; };
inline std::vector<Row>& rows() { static std::vector<Row> r; return r; }

inline void print_report() {
    if (rows().empty()) return;
    std::printf("\n%-8s %-7s %-8s %6s %12s %14s %14s\n", "ref", "type", "dir", "N", "rel L2 err", "max err [ulp]", "max elem ulp");
    for (const auto& r : rows())
        std::printf("%-8s %-7s %-8s %6u %12.3e %14.2f %14llu\n", r.ref.c_str(), r.type.c_str(), r.dir.c_str(), r.n,
                    r.s.rel_l2, r.s.max_ulp_norm, static_cast<unsigned long long>(r.s.max_ulp_elem));
}

// Invoke f(Size<N>{}) for each size.
template <unsigned... Ns, class F>
void for_sizes(F&& f, std::integer_sequence<unsigned, Ns...>) { (f(minifftdx::Size<Ns>{}), ...); }
using AllSizes = std::integer_sequence<unsigned, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096>;

}  // namespace mfxtest
