"""Run one episode of EI stopping or H2 stopping under a Gaussian learning-curve surrogate on a recorded benchmark.

    python run_episode.py --benchmark pd1 --task cifar100_wide_resnet_batch_size_256 --seed 1826701615 \
        --cost 0.001 --surrogate gp-atj --stopping h2

Surrogates (Section 5): gp-a couples asymptotes only, gp-at also couples transients through the hyperparameter kernel,
and gp-atj adds a task-wide transient. Stopping: ei stops once no expected improvement exceeds the cost; h2 adds the
certified two-step gate at every EI stop. Each benchmark, task and seed fixes a pool of 32 configurations.
Prints the episode as JSON: utility, best score, increments and actions.
"""
from __future__ import annotations

import argparse
import json
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

MODES = {"gp-a": "b_asymptote", "gp-at": "b_hp_temporal", "gp-atj": "b_pool_temporal"}


def use_fast_intervals() -> None:
    """The validated fast interval product the certificate uses in every reported H2 episode."""
    import shared_temporal_mpfr
    from fast_interval_product import interval_product
    shared_temporal_mpfr.interval_product = interval_product


def episode(benchmark: str, task: str, seed: int, cost: float, surrogate: str, stopping: str) -> dict:
    from run_shared_temporal_replay import pool_for, run_episode
    use_fast_intervals()
    mode = MODES[surrogate]
    record = run_episode(pool_for(benchmark, task, seed, mode), mode, stopping, cost)
    return dict(benchmark=benchmark, task=task, seed=seed, cost=cost, surrogate=surrogate, stopping=stopping,
                utility=record["utility"], best_score=record["incumbent"], increments=record["allocations"],
                actions=record["actions"], gate_events=len(record["boundary_events"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=("lcbench", "pd1", "odbench"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cost", type=float, required=True)
    parser.add_argument("--surrogate", required=True, choices=sorted(MODES))
    parser.add_argument("--stopping", required=True, choices=("ei", "h2"))
    args = parser.parse_args()
    print(json.dumps(episode(args.benchmark, args.task, args.seed, args.cost, args.surrogate, args.stopping)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
