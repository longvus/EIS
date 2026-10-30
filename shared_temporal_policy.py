"""History-only EI stopping and certified H2 stopping action selection.
"""

from collections import Counter

import math

import time

from gp_types import SurrogateQuery as Query

from shared_temporal_h2 import feasible_queries, estimate_pooled_h2

from shared_temporal_certificate import certify_pooled_h2

class SharedTemporalPolicy:
    def __init__(self,controller,cost,caps):
        if controller not in ("ei","h2"):
            raise ValueError("unknown controller")
        self.controller,self.cost,self.caps=controller,float(cost),tuple(caps)
        if not math.isfinite(self.cost) or self.cost<=0 or any(type(c) is not int or c<0 for c in self.caps):
            raise ValueError("invalid cost or caps")
        self.counts=Counter()
        self.events=[]
        self.ei_seconds=0.

    def choose(self,post,*,common_cache=None):
        started=time.perf_counter()
        steps=sum(map(len,post.histories))
        remaining=sum(self.caps)-steps
        arms,queries=feasible_queries(post,self.caps,remaining)
        if not arms:
            self.counts["exhaustion"]+=1
            return None
        q=max((y for h in post.histories for y in h),default=0.)
        distributions=post.predict(queries)
        eis=[d.ei(q) for d in distributions]
        best=min(range(len(arms)),key=lambda k:(-eis[k],post.candidate_ids[arms[k]]))
        self.ei_seconds+=time.perf_counter()-started
        if eis[best]>self.cost:
            self.counts["ordinary_ei_action"]+=1
            return arms[best]
        state_hash=post.state_sha256(self.caps,remaining)
        event=dict(state_sha256=state_hash,step=steps,history_lengths=list(map(len,post.histories)),
                   incumbent=q,root_ei=eis,active_candidate_ids=[post.candidate_ids[a] for a in arms],
                   root_predictions=[dict(candidate_id=post.candidate_ids[a],mu=d.mu,sigma=d.sigma) for a,d in zip(arms,distributions)],
                   ei_candidate_id=post.candidate_ids[arms[best]],ei_max=eis[best],classification="ei_boundary_stop")
        action=None
        if self.controller=="h2":
            cache_key=(state_hash,self.cost.hex())
            cached=None if common_cache is None else common_cache.get(cache_key)
            if cached is None:
                point_start=time.perf_counter()
                estimate=estimate_pooled_h2(post,self.cost,caps=self.caps,remaining_allocations=remaining)
                point_seconds=time.perf_counter()-point_start
                cert=certify_pooled_h2(post,estimate,caps=self.caps,remaining_allocations=remaining)
                cached=(estimate.to_dict(),cert,point_seconds)
                if common_cache is not None:
                    common_cache[cache_key]=cached
            point_value,cert,point_seconds=cached
            event.update(point_estimate=point_value,certificate=cert,point_seconds=point_seconds)
            if cert["accepted"]:
                action=post.candidate_ids.index(cert["selected_candidate_id"])
                event["classification"]="certified_continuation"
            elif cert["numerical_errors"]:
                event["classification"]="certificate_numerical_unknown"
            elif cert["selected_root_bound"] and cert["selected_root_bound"]["gate_status"]=="PASS":
                event["classification"]="selected_gate_nonpositive"
            else:
                event["classification"]="certificate_unknown_stop"
        event["action"]=action
        self.events.append(event)
        self.counts[event["classification"]]+=1
        return action

def advance(post,arm,value):
    epoch=post.epochs[arm][-1]+1 if post.epochs[arm] else 1.
    return post.condition(Query(post.candidate_ids[arm],epoch),value)
