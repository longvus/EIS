# Cost-Aware Stopping for Freeze–Thaw Bayesian Optimization

Code for the paper *Cost-Aware Stopping for Freeze–Thaw Bayesian Optimization: When Does One-Step Lookahead Suffice?*
([arXiv:2610.42085](https://arxiv.org/abs/2610.42085)).
It runs EI stopping and H2 stopping under the paper's three Gaussian learning-curve surrogates on recorded
hyperparameter-optimization curves from LCBench, PD1 and ODBench.

## Setup

```bash
conda env create -f environment.yml
conda activate eis
```

Download the benchmark data and build the curve caches and replay arrays as described in [DATA.md](DATA.md).
Optionally build the compiled EI kernel; a NumPy fallback is used otherwise:

```bash
python build_compiled_ei.py
```

## Run

```bash
python run_episode.py --benchmark pd1 --task cifar100_wide_resnet_batch_size_256 --seed 1826701615 \
    --cost 0.001 --surrogate gp-atj --stopping h2
```

`--surrogate` is `gp-a`, `gp-at` or `gp-atj` (Section 5 of the paper); `--stopping` is `ei` or `h2` (Section 4).
Each benchmark, task and seed fixes a pool of 32 configurations. The episode prints as JSON: utility, best score,
increments and actions.

## Tests

```bash
pytest tests
```

`tests/test_replay_arrays.py` checks the built replay arrays against their expected SHA-256; `tests/test_episode_parity.py`
checks that `run_episode.py` reproduces reference episodes under all three surrogates and both stopping rules.

## Code

| Component | Files |
|---|---|
| Surrogates GP-a, GP-at, GP-atJ | `shared_temporal_gp.py`, `shared_temporal_design.py`, `gp_types.py` |
| EI stopping and H2 stopping | `shared_temporal_policy.py`, `shared_temporal_h2.py`, `depth2_estimate.py` |
| Certificate with directed rounding | `shared_temporal_certificate.py`, `mpfr_certificate.py`, `mpfr_truncated_ei.py`, `shared_temporal_mpfr.py`, `interval_linalg.py`, `mpfr_interval.py` |
| Episodes on recorded curves | `run_shared_temporal_replay.py`, `pool_binding.py`, `shared_temporal_data.py` |

## License

The code is released under the MIT License (`LICENSE`). The benchmark data keep the licenses of their sources
(see `DATA.md`).
