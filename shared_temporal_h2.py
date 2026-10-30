"""All-root, all-feasible-child H2 for the fixed shared-temporal Gaussian law."""

import math

import numpy as np

from gp_types import SurrogateQuery as Query, AffineFantasy

from depth2_estimate import SharedDepth2Estimate, SharedDepth2RootEstimate, _quadrature_rule

from shared_temporal_gp import SharedTemporalPosterior, readonly

from surrogate_api import TruncGaussDist

def feasible_queries(post, caps, remaining):
    caps = tuple(caps)
    if len(caps) != len(post.histories) or any(type(c) is not int or c < len(h) for c, h in zip(caps, post.histories)):
        raise ValueError("invalid caps")
    if type(remaining) is not int or remaining < 0:
        raise ValueError("invalid remaining allocation count")
    arms = [a for a, (h,c) in enumerate(zip(post.histories,caps)) if remaining and len(h)<c]
    return arms, tuple(Query(post.candidate_ids[a], post.epochs[a][-1]+1 if post.epochs[a] else 1.) for a in arms)

def estimate_pooled_h2(post, cost, *, caps, remaining_allocations, quadrature_orders=(32,64,128), backend="compiled"):
    if not isinstance(post, SharedTemporalPosterior):
        raise TypeError("SharedTemporalPosterior required")
    cost = float(cost)
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("positive finite cost required")
    orders = tuple(quadrature_orders)
    if len(orders)<2 or any(type(o) is not int or o<2 for o in orders) or any(b<=a for a,b in zip(orders,orders[1:])):
        raise ValueError("invalid quadrature orders")
    if backend == "compiled":
        from compiled_ei import child_ei_batch
    elif backend == "reference":
        from depth2_estimate import _child_ei_batch as child_ei_batch
    else:
        raise ValueError("unknown child EI backend")
    arms, root_queries = feasible_queries(post,caps,remaining_allocations)
    if not arms:
        raise ValueError("no feasible H2 root")
    q=max((y for h in post.histories for y in h),default=0.)
    extra={a:Query(post.candidate_ids[a], query.epoch+1) for a,query in zip(arms,root_queries)
           if len(post.histories[a])+1<caps[a] and remaining_allocations>=2}
    all_queries=(*root_queries,*extra.values())
    joint=post.joint_parameters(all_queries)
    positions={a:k for k,a in enumerate(arms)}
    extra_positions={a:len(arms)+k for k,a in enumerate(extra)}
    estimates=[]
    for i,a in enumerate(arms):
        children=[b for b in arms if remaining_allocations>=2 and (b!=a or a in extra)]
        js=[extra_positions[a] if b==a else positions[b] for b in children]
        variance=float(joint.observation_covariance[i,i])
        cross=joint.latent_covariance[js,i]
        delta=np.outer(cross,cross)/variance
        root=TruncGaussDist(mu=float(joint.mean[i]),sigma=math.sqrt(variance))
        aff=AffineFantasy(root_queries[i],tuple(all_queries[j] for j in js),root,variance,
            float(joint.mean[i]),readonly(joint.mean[js]),readonly(cross/variance),
            readonly(joint.latent_covariance[np.ix_(js,js)]-delta),
            readonly(joint.observation_covariance[np.ix_(js,js)]-delta),"observed",post.snapshot.sigma_floor,0.,1.)
        values=[]
        for order in orders:
            levels, weights=_quadrature_rule(order)
            ei=child_ei_batch(aff,root.quantiles(levels),q)
            pooled=ei.max(axis=1) if children else np.zeros(order)
            same=ei[:,children.index(a)] if a in children else np.zeros(order)
            values.append((order,float(weights@np.maximum(pooled-cost,0)),float(weights@np.maximum(same-cost,0))))
        rei=root.ei(q)
        net=rei-cost
        pp,sp=values[-1][1:]
        pv=[v[1] for v in values]
        sv=[v[2] for v in values]
        estimates.append(SharedDepth2RootEstimate(post.candidate_ids[a],a,len(post.histories[a]),bool(children),
            tuple(post.candidate_ids[b] for b in children),rei,net,pp,sp,net+pp,net+sp,
            net+max(0.,min(pv)-1e-10),net+max(pv)+1e-10,
            net+max(0.,min(sv)-1e-10),net+max(sv)+1e-10,tuple(values)))
    selected=min(estimates,key=lambda r:(-r.pooled_advantage,r.candidate_id))
    ei_best=min(estimates,key=lambda r:(-r.root_ei,r.candidate_id))
    lower=max(0.,max(r.pooled_lower_stability_endpoint for r in estimates))
    upper=max(0.,max(r.pooled_upper_stability_endpoint for r in estimates))
    boundary=ei_best.root_ei<=cost
    return SharedDepth2Estimate(cost,q,ei_best.root_ei,ei_best.candidate_id,boundary,
        selected.candidate_id,max(0.,selected.pooled_advantage),max(0.,selected.same_arm_advantage),
        lower,upper,boundary and lower>0,boundary and upper<=0,tuple(estimates),
        dict(schema="shared-temporal-point-h2/v1",snapshot_sha256=post.snapshot.sha256(),
             state_sha256=post.state_sha256(caps,remaining_allocations),caps=list(caps),
             remaining_allocations=remaining_allocations,quadrature_orders=list(orders),
             numerical_atol=1e-10,backend=backend,full_child_roster=True,
             arms_are_autonomous=post.arms_are_autonomous,stability_interval_is_certified=False))
