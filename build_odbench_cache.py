"""Build the ODBench curve cache datasets/cfbo_benchmarks/odbench_cfbo_test_curves.npz from the CFBO data archive's
cfbo_ref/data/od_datasets.json (see DATA.md); the source file's SHA-256 is checked.

    python build_odbench_cache.py
"""
from cfbo_benchmark_io import build_cache

if __name__ == "__main__":
    print(build_cache("odbench"))
