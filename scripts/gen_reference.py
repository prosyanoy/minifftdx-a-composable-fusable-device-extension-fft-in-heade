#!/usr/bin/env python3
"""Write NumPy FFT reference vectors consumed by tests/test_numpy_reference.cpp.

ref_<N>.bin = float64 little-endian: input (2N) | np.fft.fft (2N) | N*np.fft.ifft (2N),
complex values interleaved (re, im). Inputs are float32-representable so that the
float64 NumPy result is an exact-input reference for both precisions.
"""
import sys
from pathlib import Path

import numpy as np


def main(outdir: str) -> int:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261027)
    n = 2
    while n <= 4096:
        x = (rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)).astype(np.complex64).astype(np.complex128)
        fwd = np.fft.fft(x)
        inv = np.fft.ifft(x) * n  # unnormalised inverse, like cuFFT / MiniFFTDx

        def flat(a):
            return np.column_stack([a.real, a.imag]).ravel().astype("<f8")

        np.concatenate([flat(x), flat(fwd), flat(inv)]).tofile(out / f"ref_{n}.bin")
        n *= 2
    print(f"wrote references for N=2..4096 to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "numpy_ref"))
