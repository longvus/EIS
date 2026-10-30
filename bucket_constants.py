"""
bucket_constants.py
─────────────────────────────────────────────────────────────────────────────
Constants and synthetic-curve generators salvaged from the deleted
`real_data_lcpfn_v2.py`.

The custom PFN it housed has been removed; FT-PFN (via `surrogate_api.
FTPFNSurrogate`) is now the only surrogate. These four exports are still
needed across the rest of the codebase:

  - `N_BUCKETS`           bucket count for the discretized PPD
  - `make_bucket_edges`   helper to build [0, 1] bucket edges
  - `LCBENCH_HELD_OUT_TASKS` set of task names held out from FT-PFN training
  - `generate_synthetic_horizon_randomized_curves`
                          synthetic curve generator used by
                          `train_scheduler.py --benchmark synthetic` and
                          `--benchmark ftpfn_prior` (mixed-horizon variant)

The cosine-curve helper `generate_synthetic_cosine_curves` is included as a
private dependency of the horizon-randomized generator.
"""

from __future__ import annotations

LCBENCH_HELD_OUT_TASKS = {
    "segment", "shuttle", "sylvine", "vehicle", "volkert",
    "higgs", "jannis", "jasmine",
    "jungle_chess_2pcs_raw_endgame_complete", "kc1", "kr-vs-kp",
    "mfeat-factors", "nomao", "numerai28.6", "phoneme",
}
