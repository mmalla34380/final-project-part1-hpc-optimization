

"""
hpc_vectorization_demo.py

Demonstrates an HPC-relevant optimization: reducing Python-level iteration
by using NumPy vectorization.

Three implementations of the same elementwise operation:
  1) python_loop         - pure Python for-loop over lists
  2) list_comprehension  - Python list comprehension
  3) numpy_vectorized    - NumPy vectorized expression (recommended)

Operation computed:  c[i] = a[i] * b[i] + sin(a[i])

Dependencies:
  numpy, pandas
Run: python hpc_vectorization_demo.py
"""

import time
import math
import numpy as np
import pandas as pd

def python_loop(a_list, b_list):
    """Pure Python loop using math.sin."""
    out = [0.0] * len(a_list)
    for i in range(len(a_list)):
        out[i] = a_list[i] * b_list[i] + math.sin(a_list[i])
    return out

def list_comprehension(a_list, b_list):
    """List comprehension (still Python-level iteration)."""
    return [ai * bi + math.sin(ai) for ai, bi in zip(a_list, b_list)]

def numpy_vectorized(a_np, b_np):
    """NumPy vectorized version (runs in compiled C loops)."""
    return a_np * b_np + np.sin(a_np)

def timeit(func, *args, repeat=3):
    """Return best time (min of repeat runs) and the last result."""
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), result

def max_abs_diff(x, y):
    """Max absolute difference between two sequences/arrays (numpy-friendly)."""
    xa = np.array(x)
    ya = np.array(y)
    return float(np.max(np.abs(xa - ya)))

def main():
    # Input size
    N = 1_000_000  # adjust up/down depending on machine
    print(f"Preparing inputs (N = {N})...")

    rng = np.random.default_rng(42)
    a_np = rng.random(N).astype(np.float64) * 10.0
    b_np = rng.random(N).astype(np.float64) * 10.0

    # Pure-Python inputs
    a_list = a_np.tolist()
    b_list = b_np.tolist()

    # Warm-up (helps avoid one-time overhead affecting measured times)
    _ = math.sin(0.5)
    _ = numpy_vectorized(a_np[:10], b_np[:10])

    # Benchmarks
    t_loop, res_loop = timeit(python_loop, a_list, b_list, repeat=3)
    t_lc,   res_lc   = timeit(list_comprehension, a_list, b_list, repeat=3)
    t_np,   res_np   = timeit(numpy_vectorized, a_np, b_np, repeat=3)

    # Verify correctness (small sample and numeric closeness)
    diff_loop_vs_np = max_abs_diff(res_loop[:1000], res_np[:1000])
    diff_lc_vs_np   = max_abs_diff(res_lc[:1000], res_np[:1000])

    # Summary table using pandas (optional)
    df = pd.DataFrame([
        ("python_loop", t_loop),
        ("list_comprehension", t_lc),
        ("numpy_vectorized", t_np),
    ], columns=["implementation", "time_seconds"])
    df["speedup_vs_python_loop"] = df["time_seconds"].iloc[0] / df["time_seconds"]
    df = df.sort_values("time_seconds").reset_index(drop=True)

    print("\nBenchmark results (best of 3 runs):")
    print(df.to_string(index=False, float_format="{:0.6f}".format))

    print("\nMax absolute differences (sample of first 1000):")
    print(f"python_loop vs numpy: {diff_loop_vs_np:.6e}")
    print(f"list_comp  vs numpy: {diff_lc_vs_np:.6e}")

    print("\nObservations:")
    print("- NumPy vectorized implementation avoids Python-level per-element overhead")
    print("- It typically gives large speedups for numeric array workloads")
    print("- Caveat: avoid unnecessary conversions between lists and numpy arrays; keep data as numpy arrays when possible")
    print("- For patterns not easily vectorizable, consider Numba or C/C++ extensions")

if __name__ == "__main__":
    main()

