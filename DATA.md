# Data

The benchmark data come from their public releases and are not redistributed here. Download each source, build the
two curve caches, then build the replay arrays; every builder checks its source files' SHA-256, and
`tests/test_replay_arrays.py` compares the built arrays with their reference SHA-256.

| Benchmark | Download to | Source |
|---|---|---|
| LCBench | `datasets/lcbench/data_2k_lw.json` | LCBench (Zimmer et al., 2021), https://github.com/automl/LCBench (CC BY 4.0) |
| PD1 | `datasets/pd1/pd1_matched_phase1_results.jsonl.gz` | PD1 (Wang et al., 2024), https://github.com/google-research/hyperbo (CC BY 4.0) |
| ODBench | `cfbo_ref/data/od_datasets.json` | the data archive released with CFBO (Lee et al., 2025), https://drive.google.com/file/d/1SWhf1v6QGFDvYvPxXv7yVZiTfOCD0oAs/view |

```bash
python build_pd1_cache.py                      # datasets/pd1/pd1_all_curves_15tasks.npz
python build_odbench_cache.py                  # datasets/cfbo_benchmarks/odbench_cfbo_test_curves.npz
python build_shared_temporal_replay_arrays.py  # staging/shared_temporal_gp/v1/replay_arrays/
```
