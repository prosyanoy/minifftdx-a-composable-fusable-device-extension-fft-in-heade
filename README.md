# MiniFFTDx — a composable, fusable device-extension FFT in header-only C++

A small header-only C++17 library in the style of **cuFFTDx**: you describe an FFT at
compile time (size, precision, direction, execution scope) and can **fuse** it with
pointwise operations inside one kernel / pass (FFT → filter → IFFT convolution). The
same templated radix-2/4 algorithm runs on a **CUDA block-level backend** and on an
**OpenMP CPU backend**, so everything builds and tests without a GPU.

## Features

- **Descriptor API** — `fft_descriptor<Size<N>, Precision<float|double>, Direction<fwd|inv>, Block>`.
  Invalid configs (non-power-of-two or out-of-range size, `Precision<int>`, wrong operator
  types, unsupported scope, mismatched fused pair) fail with a readable `static_assert`.
  Eight compile-fail cases are part of the test suite.
- **Mixed-radix 2/4 FFT, written once** (`include/minifftdx/fft.hpp`) against a tiny
  execution-context concept; the CUDA and CPU backends only supply the context.
- **Fusion** — `fused_fft_op_ifft_block<Fwd, Inv>(data, op, ctx)` applies any functor
  between the forward and inverse transforms. On CUDA the spectrum never leaves shared
  memory (one kernel, one global load, one global store). Ready-made `spectral_multiply`
  (FFT convolution) and `scale_by` functors; lambdas work too.
- **Tests** — vs a long-double naive DFT and vs NumPy, sizes 2…4096 (the spec's 8–4096 plus 2/4),
  float/double, forward/inverse, with relative-L2 and ULP error reporting; fusion,
  batching, in-place, structure (impulse, tone, Parseval, round trip) tests.
- **Benchmarks** — Google Benchmark + `scripts/run_benchmarks.py` → table (md/CSV) and plot.
  CPU numbers always; GPU numbers when built with nvcc and a device is present.

## Usage (the 10-line version)

```cpp
#include "minifftdx.hpp"
using namespace minifftdx;

constexpr unsigned N = 1024;
using FFT  = fft_descriptor<Size<N>, Precision<float>, Direction<fft_direction::forward>, Block>;
using IFFT = fft_descriptor<Size<N>, Precision<float>, Direction<fft_direction::inverse>, Block>;

// x, y: batch * N elements; H = FFT of the filter (computed once with cpu::execute<FFT>)
cpu::execute_fused<FFT, IFFT>(x, y, batch, spectral_multiply<float>{H, 1.f / N});  // FFT -> *H -> IFFT, one pass
// CUDA: cuda::execute_fused<FFT, IFFT>(d_x, d_y, batch, spectral_multiply<float>{d_H, 1.f / N});
```

A complete program is in `examples/fft_convolution.cpp`. Conventions: forward is
`exp(-2πi nk/N)`, the inverse is **unnormalised** (like cuFFT/cuFFTDx: `ifft(fft(x)) = N·x`),
so fold `1/N` into your fused functor.

## Architecture

```
            fft_descriptor<Size, Precision, Direction, Block>      descriptor.hpp  (compile-time config + static_asserts)
                                   │
                                   ▼
      fft_block<Desc>(x, ctx)  ── radix-2/4 DIT, in place ──────   fft.hpp         (written ONCE)
      fused_fft_op_ifft_block  ── fwd → op → inv, in place ─────   fusion.hpp
                    │  ctx = { tid(), nthreads(), sync(), twiddle(k) }
        ┌───────────┴─────────────────────────┐
        ▼                                     ▼
 backend/cpu.hpp                        backend/cuda.hpp   (only compiled by nvcc)
 ctx: 1 worker, no-op sync,             ctx: threadIdx.x / blockDim.x, __syncthreads(),
      twiddle table                          sincospi twiddles
 batch loop: #pragma omp for            1 block per FFT, data in dynamic shared memory;
 thread-local buffer per FFT            fft_kernel / fused_kernel / pointwise_kernel
```

Layout: `include/minifftdx/` (library), `tests/` (GoogleTest, incl. `compile_fail/`),
`bench/` (Google Benchmark, CPU + CUDA), `scripts/` (NumPy reference generator, benchmark
runner, demo), `examples/`.

## Setup

Requirements: a C++17 compiler, CMake ≥ 3.18. Optional: OpenMP, nvcc, Python 3 with NumPy
(+ matplotlib for the plot). GoogleTest and Google Benchmark are used if installed
(e.g. `apt install libgtest-dev libbenchmark-dev`), otherwise downloaded by CMake
(`-DMINIFFTDX_FETCH_DEPS=OFF` to forbid that).

```bash
pip install -r requirements.txt            # NumPy (tests) + matplotlib (plot); optional
cmake -S . -B build                         # CUDA is enabled automatically iff nvcc is found
cmake --build build -j
```

Options: `-DMINIFFTDX_ENABLE_CUDA=AUTO|ON|OFF`, `-DMINIFFTDX_BUILD_TESTS/BENCH/EXAMPLES=ON|OFF`.
No configuration or secrets are required; `.env.example` lists the optional knobs
(`OMP_NUM_THREADS`, `MINIFFTDX_BENCH_MIN_TIME`).

## Run, test, benchmark

```bash
build/fft_convolution                       # example
ctest --test-dir build --output-on-failure  # all tests (30 on a CPU-only box)
build/test_accuracy                         # prints the error table (rel. L2, ULP) vs the naive DFT
MINIFFTDX_REFDIR=build/numpy_ref build/test_numpy_reference   # same vs NumPy (after ctest generated the data)
python3 scripts/run_benchmarks.py --build-dir build           # table + plot in bench/results/
scripts/demo.sh                             # everything in ~1 minute (record this as the terminal GIF)
```

## Results

Measured in the dev container (14 cores, GCC 13, OpenMP, **CPU only — no GPU was
available, so the CUDA path is written but has not been compiled or run**).
2²² complex points per case (batch = 2²²/N), so the working set exceeds cache.

| backend | precision | N | fused [Mconv/s] | unfused [Mconv/s] | speedup | fused GFLOP/s | unfused GFLOP/s |
|---|---|---|---|---|---|---|---|
| cpu | double | 64 | 19.537 | 13.896 | 1.41x | 75.0 | 53.4 |
| cpu | double | 256 | 4.007 | 2.955 | 1.36x | 82.1 | 60.5 |
| cpu | double | 1024 | 0.727 | 0.600 | 1.21x | 74.4 | 61.5 |
| cpu | double | 4096 | 0.146 | 0.114 | 1.28x | 71.7 | 56.0 |
| cpu | float | 64 | 20.236 | 18.245 | 1.11x | 77.7 | 70.1 |
| cpu | float | 256 | 3.709 | 3.375 | 1.10x | 76.0 | 69.1 |
| cpu | float | 1024 | 0.807 | 0.726 | 1.11x | 82.6 | 74.3 |
| cpu | float | 4096 | 0.159 | 0.143 | 1.11x | 78.3 | 70.4 |

Fusion removes two full read+write sweeps over memory (spectrum out, pointwise pass,
spectrum back in). The gain on CPU is moderate (≈1.1–1.4×) because the transforms are
compute-bound at 70–80 GFLOP/s on this machine; on a GPU the intermediate round trips go
to DRAM and the gap is expected to be larger — run `bench_fusion_cuda` to measure it.

**Accuracy** (vs. a long-double naive DFT and NumPy float64, N = 2…4096): relative L2 error
≤ 1.3·10⁻⁷ for float and ≤ 3.0·10⁻¹⁶ for double at N = 4096 (≈ 1 ε), and the
max absolute error is ≤ ~2 ULP of the peak output value (`max err [ulp]` column). The
`max elem ulp` column is the per-component ULP distance over components ≥ 1 % of the peak
(up to ~100 ULP at N = 4096; individual small components of a sum naturally lose relative
precision). Tests assert `rel L2 < 2ε(log₂N+1)`.

CV line: *Built a header-only C++/CUDA FFT library with compile-time descriptors and kernel
fusion; 1.1–1.4× faster than the unfused baseline on the CPU backend, max error ≈ 1.6 ULP
(normwise) vs. a long-double reference.* (Replace with your GPU number after running on a GPU.)

## How to present it

- README with an architecture diagram, a 10-line usage snippet, and a benchmark table
  (fused vs. unfused, error vs. reference) — all above.
- CV line: *"Built header-only C++/CUDA FFT library with compile-time descriptors and kernel
  fusion; X× faster than unfused baseline, max error Y ULP."* — fill X and Y from your own
  `scripts/run_benchmarks.py` and `build/test_accuracy` output.
- Include a 60-second terminal GIF running tests and benchmarks: `scripts/demo.sh` runs
  exactly that sequence; record it with asciinema/vhs and link the GIF here.
- Optional profiling: `ncu --set full build/bench_fusion_cuda --benchmark_filter=fused/float/1024`
  to show shared-memory-only traffic in the fused kernel (Nsight Compute, GPU box).

## Key design decisions

- **One algorithm, two backends.** `fft_block` is parameterised on an execution context
  rather than duplicated; CPU = 1 worker per FFT with parallelism across the batch,
  CUDA = a block of threads per FFT with `__syncthreads()` between stages.
- **Radix-4 by fusing two radix-2 stages**, preceded by one radix-2 stage when log₂N is odd:
  half the passes and barriers, 4-point butterflies per thread, any power of two supported.
- **Twiddles**: CPU table computed in long double and rounded once (accurate); CUDA uses
  `sincospi(f)` with an exactly representable argument — no table, no constant-memory setup.
- **Unnormalised inverse**, matching cuFFT/cuFFTDx; normalisation belongs in the fused functor.
- **Fused and unfused CPU paths are bit-identical** (same arithmetic; tested), so the
  benchmark isolates memory traffic.
- **Sizes 2…16384**, bounded by CUDA shared memory (the launcher raises the dynamic
  shared-memory limit above 48 KB; very large double-precision sizes need a GPU that allows it).
- Errors are reported normwise (relative L2, ULPs of the peak) because per-component
  relative error is meaningless near cancellation zeros; per-component ULPs are shown for
  significant components.
- **Scope / limitations**: batched 1-D complex-to-complex, power-of-two sizes, one FFT per
  block, filter shared across the batch. The CUDA backend (`tests/test_cuda.cu`,
  `bench/bench_fusion.cu`) is untested here for lack of nvcc/GPU; it reuses the verified
  core, so only launch/shared-memory plumbing is unverified.
