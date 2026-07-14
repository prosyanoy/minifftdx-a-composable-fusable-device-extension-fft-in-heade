#pragma once
#include "config.hpp"

namespace minifftdx {

// Minimal POD complex type usable in host and device code (std::complex is not
// guaranteed to be device-callable).
template <class T>
struct complex {
    using value_type = T;
    T re;
    T im;

    MFX_HD constexpr complex() : re(0), im(0) {}
    MFX_HD constexpr complex(T r, T i = 0) : re(r), im(i) {}

    MFX_HD constexpr complex& operator+=(const complex& o) { re += o.re; im += o.im; return *this; }
    MFX_HD constexpr complex& operator-=(const complex& o) { re -= o.re; im -= o.im; return *this; }
};

template <class T>
MFX_HD constexpr complex<T> operator+(complex<T> a, complex<T> b) { return {a.re + b.re, a.im + b.im}; }
template <class T>
MFX_HD constexpr complex<T> operator-(complex<T> a, complex<T> b) { return {a.re - b.re, a.im - b.im}; }
template <class T>
MFX_HD constexpr complex<T> operator*(complex<T> a, complex<T> b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <class T>
MFX_HD constexpr complex<T> operator*(complex<T> a, T s) { return {a.re * s, a.im * s}; }
template <class T>
MFX_HD constexpr complex<T> operator*(T s, complex<T> a) { return {a.re * s, a.im * s}; }
template <class T>
MFX_HD constexpr bool operator==(complex<T> a, complex<T> b) { return a.re == b.re && a.im == b.im; }
template <class T>
MFX_HD constexpr bool operator!=(complex<T> a, complex<T> b) { return !(a == b); }

template <class T>
MFX_HD constexpr complex<T> conj(complex<T> a) { return {a.re, -a.im}; }

}  // namespace minifftdx
