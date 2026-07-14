#!/usr/bin/env python3
"""Run the fused-vs-unfused benchmarks and emit a table (markdown + CSV) and a plot.

CPU numbers are always produced (bench_fusion_cpu). GPU numbers are added when the
project was built with CUDA (bench_fusion_cuda exists) and a device is present.

    python3 scripts/run_benchmarks.py --build-dir build [--min-time 0.2] [--out-dir bench/results]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def run(binary: Path, min_time: str) -> list:
    cmd = [str(binary), "--benchmark_format=json", f"--benchmark_min_time={min_time}s"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"{binary.name} failed:\n{res.stderr}")
    if not res.stdout.strip():
        return []  # e.g. CUDA binary with no device
    return json.loads(res.stdout)["benchmarks"]


def collect(backend: str, benches: list) -> dict:
    """-> {(backend, precision, N): {'fused': conv/s, 'unfused': conv/s, 'gflops_f', 'gflops_u'}}"""
    out = {}
    for b in benches:
        kind, prec, n = b["name"].split("/")[:3]
        key = (backend, prec, int(n))
        side = "fused" if kind == "conv_fused" else "unfused"
        d = out.setdefault(key, {})
        d[side] = b["items_per_second"]
        d["gflops_" + side[0]] = b.get("GFLOPS", float("nan"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--min-time", default=os.environ.get("MINIFFTDX_BENCH_MIN_TIME", "0.2"))
    ap.add_argument("--out-dir", default="bench/results")
    args = ap.parse_args()
    bdir, odir = Path(args.build_dir), Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    results = {}
    cpu = bdir / "bench_fusion_cpu"
    if not cpu.exists():
        sys.exit(f"{cpu} not found - build first (cmake -S . -B build && cmake --build build)")
    results.update(collect("cpu", run(cpu, args.min_time)))
    gpu = bdir / "bench_fusion_cuda"
    if gpu.exists():
        results.update(collect("gpu", run(gpu, args.min_time)))
    else:
        print("note: no bench_fusion_cuda (built without nvcc) - CPU numbers only", file=sys.stderr)

    rows = []
    for (backend, prec, n), r in sorted(results.items()):
        rows.append([backend, prec, n, r["fused"] / 1e6, r["unfused"] / 1e6, r["fused"] / r["unfused"],
                     r["gflops_f"], r["gflops_u"]])
    header = ["backend", "precision", "N", "fused [Mconv/s]", "unfused [Mconv/s]", "speedup", "fused GFLOP/s", "unfused GFLOP/s"]

    with open(odir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[*r[:3], *(f"{v:.4f}" for v in r[3:])] for r in rows])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.2f}x | {r[6]:.1f} | {r[7]:.1f} |")
    table = "\n".join(lines)
    (odir / "results.md").write_text(table + "\n")
    print(table)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot (pip install matplotlib)", file=sys.stderr)
        return 0
    series = sorted({(r[0], r[1]) for r in rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for i, (backend, prec) in enumerate(series):
        sel = [r for r in rows if r[0] == backend and r[1] == prec]
        ns = [r[2] for r in sel]
        c = colors[i % len(colors)]
        a1.plot(ns, [r[3] for r in sel], "-o", color=c, label=f"{backend} {prec} fused")
        a1.plot(ns, [r[4] for r in sel], "--s", color=c, alpha=0.6, label=f"{backend} {prec} unfused")
        a2.plot(ns, [r[5] for r in sel], "-o", color=c, label=f"{backend} {prec}")
    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.set_xlabel("FFT size N")
        a.grid(alpha=0.3)
    a1.set_yscale("log")
    a1.set_ylabel("FFT convolutions / s  (millions)")
    a1.set_title("Throughput")
    a1.legend(fontsize=7)
    a2.axhline(1.0, color="gray", lw=1)
    a2.set_ylabel("fused / unfused")
    a2.set_title("Fusion speedup")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(odir / "results.png", dpi=130)
    print(f"\nwrote {odir}/results.md, results.csv, results.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
