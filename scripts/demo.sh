#!/usr/bin/env bash
# ~60 second terminal demo: build, run the tests (with the error table), run the
# benchmarks. Record it with e.g. `asciinema rec -c scripts/demo.sh` or `vhs`.
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build >/dev/null && cmake --build build -j >/dev/null
echo "== tests =="; ctest --test-dir build --output-on-failure -j4 | tail -8
echo "== accuracy vs naive DFT / NumPy =="
(cd build && ./test_accuracy | grep -E "^(ref|naive)" | awk 'NR==1 || $4==8 || $4==4096')
echo "== example =="; build/fft_convolution
echo "== benchmarks (fused vs unfused) =="; python3 scripts/run_benchmarks.py --build-dir build --min-time 0.1
