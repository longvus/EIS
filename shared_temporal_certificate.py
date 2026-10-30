"""Selected-root pooled-H2 certificate for exact-input shared-temporal laws."""

from dataclasses import asdict

import math

import time

from gp_types import SurrogateQuery as Query

from mpfr_certificate import SharedMPFRCertificateConfig, _CompiledSelectedRoot, _MPFRChildLaw, _bound_selected_root, _selected_bound_for_eta

from mpfr_truncated_ei import bound_truncated_normal_ei, IntervalRecord, NumericalUnresolved

from shared_temporal_design import sha

from shared_temporal_h2 import feasible_queries

from shared_temporal_mpfr import SharedTemporalMPFRPosterior, point

ALGORITHM = "shared-temporal-verified-residual-slope-endpoint/v1"

def interval_record(v):
    return asdict(IntervalRecord.from_interval(v))

def sign(net):
    return "FAIL" if net.lo > 0 else "PASS" if net.hi <= 0 else "UNKNOWN"

def certify_pooled_h2(post,estimate,*,caps,remaining_allocations,config=None):
    config=SharedMPFRCertificateConfig() if config is None else config
    if not isinstance(config,SharedMPFRCertificateConfig):
        raise TypeError("invalid certificate configuration")
    started=time.perf_counter()
    arms,queries=feasible_queries(post,caps,remaining_allocations)
    cost=float(estimate.lambda_)
    q=max((y for h in post.histories for y in h),default=0.)
    if not math.isfinite(cost) or cost<=0 or estimate.incumbent!=q:
        raise ValueError("inconsistent point target")
    state_hash=post.state_sha256(caps,remaining_allocations)
    if estimate.metadata.get("state_sha256")!=state_hash or estimate.metadata.get("snapshot_sha256")!=post.snapshot.sha256():
        raise ValueError("point selector state/snapshot mismatch")
    if tuple(r.candidate_id for r in estimate.root_estimates)!=tuple(post.candidate_ids[a] for a in arms):
        raise ValueError("point root roster mismatch")
    selected=min(estimate.root_estimates,key=lambda r:(-r.pooled_advantage,r.candidate_id))
    if selected.candidate_id!=estimate.selected_candidate_id or not all(math.isfinite(r.pooled_advantage) for r in estimate.root_estimates):
        raise ValueError("invalid point selector")
    inputs=dict(algorithm=ALGORITHM,state_sha256=state_hash,selector_sha256=sha(estimate.to_dict()),
                cost_hex=cost.hex(),eta_hex=0.0.hex(),config=asdict(config))
    result=dict(schema="shared-temporal-certificate/v1",**inputs,input_sha256=sha(inputs),
        selected_candidate_id=selected.candidate_id,accepted=False,boundary_status="UNKNOWN",
        selected_root_bound=None,root_ei_certificates=[],numerical_errors=[],
        scope="local selected-root rollout; no all-root or full-horizon stopping claim")
    p=config.mpfr_precision_bits
    try:
        mpfr=SharedTemporalMPFRPosterior(post,p)
        result["posterior_diagnostics"]=mpfr.diagnostics
        joint=mpfr.joint_parameters(queries)
        enclosures=[]
        for query,mean,row in zip(queries,joint.means,joint.observation_covariance):
            i=len(enclosures)
            variance=row[i]
            if variance.lo < point(post.snapshot.sigma_floor,p).square().hi:
                raise NumericalUnresolved("root scale floor inactivity unresolved")
            sigma=variance.sqrt()
            ei=bound_truncated_normal_ei(mean,sigma,point(q,p),tolerance=config.root_ei_tolerance,max_cells=config.max_root_ei_cells)
            status=sign(ei.value-point(cost,p))
            result["root_ei_certificates"].append(dict(candidate_id=query.candidate_id,mean=interval_record(mean),
                sigma=interval_record(sigma),ei=interval_record(ei.value),status=status,cells=ei.cells,tolerance_met=ei.resolved))
            enclosures.append(ei)
        statuses=[r["status"] for r in result["root_ei_certificates"]]
        boundary="FAIL" if "FAIL" in statuses else "PASS" if all(v=="PASS" for v in statuses) else "UNKNOWN"
        result["boundary_status"]=boundary
        if boundary=="PASS":
            root_arm=post.candidate_ids.index(selected.candidate_id)
            root_pos=arms.index(root_arm)
            root=queries[root_pos]
            children=tuple(Query(post.candidate_ids[a],root.epoch+1 if a==root_arm else queries[arms.index(a)].epoch)
                for a in arms if remaining_allocations>=2 and (a!=root_arm or len(post.histories[a])+1<caps[a]))
            values=mpfr.joint_parameters((root,*children))
            root_mean,root_variance=values.means[0],values.observation_covariance[0][0]
            child_laws=[]
            for k,query in enumerate(children,start=1):
                cross=values.latent_covariance[k][0]
                var=values.observation_covariance[k][k]-cross.square()/root_variance
                if var.lo < point(post.snapshot.sigma_floor,p).square().hi:
                    raise NumericalUnresolved("child scale floor inactivity unresolved")
                child_laws.append(_MPFRChildLaw(query.candidate_id,query.epoch,values.means[k],cross/root_variance,var,var.sqrt()))
            coefficient_hash=sha(dict(root_mean=interval_record(root_mean),root_variance=interval_record(root_variance),
                children=[dict(candidate_id=c.candidate_id,mean=interval_record(c.base_mean),slope=interval_record(c.mean_slope),
                               variance=interval_record(c.observation_variance)) for c in child_laws]))
            compiled=_CompiledSelectedRoot(root.candidate_id,root.epoch,root_mean,root_variance,root_variance.sqrt(),tuple(child_laws),coefficient_hash)
            raw=_bound_selected_root(compiled,enclosures[root_pos],q,cost,config)
            bound=_selected_bound_for_eta(raw,cost,0.,p)
            result["selected_root_bound"]=bound.to_dict()
            result["integration_diagnostics"]=dict(raw.diagnostics)
            result["accepted"]=bound.gate_status.value=="FAIL"
    except (ArithmeticError, NumericalUnresolved) as error:
        result["numerical_errors"].append(dict(stage="shared_temporal_mpfr",type=type(error).__name__,message=str(error)))
        result["accepted"]=False
        result["boundary_status"]="UNKNOWN"
    result["runtime_seconds"]=time.perf_counter()-started
    return result
