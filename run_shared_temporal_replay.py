"""EI and H2 stopping episodes on recorded curves, with no fitting to scores."""

from __future__ import annotations

import time

from shared_temporal_data import binding, model_snapshot

from shared_temporal_gp import SharedTemporalPosterior

from shared_temporal_policy import SharedTemporalPolicy, advance

def run_episode(pool,mode,controller,cost):
    if pool.snapshot.mode!=mode:
        raise ValueError("pool snapshot mode differs")
    post=SharedTemporalPosterior(pool.snapshot,pool.candidate_ids,pool.hps,[[] for _ in pool.candidate_ids])
    policy=SharedTemporalPolicy(controller,cost,(pool.curve_len,)*len(pool.candidate_ids))
    actions,trajectory=[],[]
    q=0.
    context_s=0.
    started=time.perf_counter()
    for _ in range(pool.curve_len*len(pool.candidate_ids)+1):
        action=policy.choose(post)
        if action is None:
            break
        # Only now does the evaluator reveal the chosen next curve score.
        y=float(pool.curves[action,len(post.histories[action])])
        actions.append(action)
        q=max(q,y)
        trajectory.append(q)
        start=time.perf_counter()
        post=advance(post,action,y)
        context_s+=time.perf_counter()-start
    else:
        raise RuntimeError("missing explicit finite-cap stop")
    return dict(status="complete",actions=actions,trajectory=trajectory,allocations=len(actions),incumbent=q,
        utility=q-cost*len(actions),boundary_events=policy.events,decision_counts=dict(policy.counts),
        terminal_state_sha256=post.state_sha256(policy.caps,sum(policy.caps)-len(actions)),
        runtime_seconds=time.perf_counter()-started,context_seconds=context_s,ei_seconds=policy.ei_seconds,
        execution_failure=None)

def pool_for(benchmark,task,seed,mode):
    from types import SimpleNamespace
    value=binding(benchmark,task,seed)
    return SimpleNamespace(candidate_ids=value.candidate_ids,hps=value.hps,curves=value.curves,
        row_ids=value.row_ids,pool_sha256=value.pool_sha256,curve_len=value.curve_len,
        snapshot=model_snapshot(benchmark,mode))
