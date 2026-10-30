"""Rigorous coefficient-space posterior of the Gaussian surrogates via MPFR residual enclosures.
"""

from functools import lru_cache

import gmpy2 as g

import numpy as np

from mpfr_interval import MPInterval as IV, _contexts

from interval_linalg import verified_cholesky, solve_spd

from shared_temporal_design import WEIGHTS

from mpfr_certificate import MPFRJointParameters

def point(value, precision):
    return IV.point(float(value), precision)

def basis_iv(t, snapshot, precision):
    time=point(t,precision)
    one=point(1,precision)
    return (one, *((point(s,precision)+one)/(point(s,precision)+time) for s in snapshot.basis_scales))

def noise_iv(t,snapshot,precision):
    return point(snapshot.noise_base,precision)+point(snapshot.noise_decay,precision)/(point(t,precision)+1).square()+point(snapshot.jitter,precision)

def endpoints(matrix):
    return (np.array([[v.lo for v in row] for row in matrix],dtype=object),
            np.array([[v.hi for v in row] for row in matrix],dtype=object))

def interval_product(a,b,precision):
    """Enclose interval matrix products by sign-split point midpoint + radius.

    The midpoint can be any point. Its MPFR rounding is included by computing
    the residual radius against both true endpoints with upward rounding.
    """
    alo,ahi=a
    blo,bhi=b
    down,up=_contexts(precision)
    with g.context(precision=precision):
        am=(alo+ahi)/g.mpfr(2)
        bm=(blo+bhi)/g.mpfr(2)
    with g.context(up):
        ar=np.maximum(am-alo,ahi-am)
        br=np.maximum(bm-blo,bhi-bm)
        error=np.abs(am)@br+ar@np.abs(bm)+ar@br
    with g.context(down):
        lo=am@bm-error
    with g.context(up):
        hi=am@bm+error
    return lo,hi

def infinity_upper(bounds,precision):
    with g.context(_contexts(precision)[1]):
        return max(np.sum(np.maximum(np.abs(bounds[0]),np.abs(bounds[1])),axis=1),default=g.mpfr(0))

class VerifiedInverse:
    """||I-R A||_inf < 1 proves A^-1 encloses R +- scalar error.

    A^-1=(I-E)^-1 R, E=I-R A, hence
    ||A^-1-R||_inf <= ||E||_inf ||R||_inf/(1-||E||_inf).
    Every matrix entry is enclosed by that norm error. Symmetry/PD of A follows
    separately from the fixed Gaussian model and is not inferred from R.
    """
    def __init__(self,matrix,preconditioner,precision):
        self.precision=precision
        r=np.asarray(preconditioner,dtype=float)
        if r.shape!=(len(matrix),len(matrix)) or not np.isfinite(r).all():
            raise ArithmeticError("invalid preconditioner")
        self.r=np.array([[point(v,precision).lo for v in row] for row in r],dtype=object)
        self.a=endpoints(matrix)
        lo,hi=interval_product((self.r,self.r),self.a,precision)
        identity=np.array([[g.mpfr(int(i==j)) for j in range(len(r))] for i in range(len(r))],dtype=object)
        down,up=_contexts(precision)
        with g.context(down):
            elo=identity-hi
        with g.context(up):
            ehi=identity-lo
        rho=infinity_upper((elo,ehi),precision)
        if not g.is_finite(rho) or rho>=1:
            raise ArithmeticError("preconditioned enclosure is noncontractive")
        self.rho=rho
        radius=point(1,precision)-IV(rho,rho,precision)
        rnorm=infinity_upper((self.r,self.r),precision)
        err=(IV(rho,rho,precision)*IV(rnorm,rnorm,precision)/radius).hi
        with g.context(down):
            lower=self.r-err
        with g.context(up):
            upper=self.r+err
        self.inverse=(lower,upper)

    def solve(self,right,estimate):
        p=self.precision
        x=np.array([[point(v,p).lo] for v in estimate],dtype=object)
        b=endpoints([[v] for v in right])
        ax=interval_product(self.a,(x,x),p)
        with g.context(_contexts(p)[0]):
            rlo=b[0]-ax[1]
        with g.context(_contexts(p)[1]):
            rhi=b[1]-ax[0]
        residual=interval_product((self.r,self.r),(rlo,rhi),p)
        bound=infinity_upper(residual,p)
        error=(IV(bound,bound,p)/(point(1,p)-IV(self.rho,self.rho,p))).hi
        with g.context(_contexts(p)[0]):
            lo=x-error
        with g.context(_contexts(p)[1]):
            hi=x+error
        return lo,hi

@lru_cache(maxsize=4)
def prior_precision(snapshot,feature_rows,precision):
    n=len(feature_rows)
    zero=point(0,precision)
    matern=[[zero for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i,n):
            r2=zero
            for a,b,ell in zip(feature_rows[i],feature_rows[j],snapshot.lengthscales):
                r2=r2+((point(a,precision)-point(b,precision))/point(ell,precision)).square()
            z=(point(5,precision)*r2).sqrt()
            val=(1+z+z.square()/3)*(-z).exp()
            matern[i][j]=matern[j][i]=val
    inverses=[]
    for weights in WEIGHTS[snapshot.mode]:
        mat=[[point(weights[0]*int(i==j)+weights[2],precision)+point(weights[1],precision)*matern[i][j]
              for j in range(n)] for i in range(n)]
        factor=verified_cholesky(mat)
        columns=[solve_spd(factor,[point(int(i==j),precision) for i in range(n)]) for j in range(n)]
        inverses.append([[columns[j][i] for j in range(n)] for i in range(n)])
    result=[[zero for _ in range(5*n)] for _ in range(5*n)]
    for r in range(5):
        variance=point(snapshot.asymptote_variance if r==0 else snapshot.temporal_variance/4,precision)
        for i in range(n):
            for j in range(n):
                result[i*5+r][j*5+r]=inverses[r!=0][i][j]/variance
    return tuple(map(tuple,result))

@lru_cache(maxsize=16)
def temporal_information(epochs,snapshot,precision):
    zero=point(0,precision)
    info=[[zero for _ in range(5)] for _ in range(5)]
    weighted=[]
    for t in epochs:
        phi=basis_iv(t,snapshot,precision)
        inv_noise=1/noise_iv(t,snapshot,precision)
        row=tuple(v*inv_noise for v in phi)
        weighted.append(row)
        for i in range(5):
            for j in range(i,5):
                info[i][j]=info[i][j]+row[i]*phi[j]
    for i in range(5):
        for j in range(i):
            info[i][j]=info[j][i]
    return tuple(map(tuple,info)),tuple(weighted)

class SharedTemporalMPFRPosterior:
    def __init__(self,post,precision=256,*,solver="verified_residual"):
        self.post,self.precision=post,precision
        _contexts(precision)
        p=precision
        matrix=[list(row) for row in prior_precision(post.snapshot,tuple(map(tuple,post.features)),p)]
        rhs=[point(0,p) for _ in matrix]
        for arm,(epochs,history) in enumerate(zip(post.epochs,post.histories)):
            info,weighted=temporal_information(epochs,post.snapshot,p)
            for i in range(5):
                for j in range(5):
                    matrix[5*arm+i][5*arm+j]=matrix[5*arm+i][5*arm+j]+info[i][j]
                rhs[5*arm+i]=sum((w[i]*(point(y,p)-point(post.snapshot.mean,p)) for w,y in zip(weighted,history)),point(0,p))
        self.matrix,self.rhs=matrix,rhs
        if solver=="verified_residual":
            verified=VerifiedInverse(matrix,post.coefficient_covariance,p)
            self.covariance=verified.inverse
            self.mean=verified.solve(rhs,post.coefficient_mean)
            self.diagnostics=dict(solver=solver,coefficient_dimension=len(matrix),contraction_upper=float(verified.rho))
        elif solver=="interval_cholesky":
            factor=verified_cholesky(matrix)
            columns=[solve_spd(factor,[point(int(i==j),p) for i in range(len(matrix))]) for j in range(len(matrix))]
            self.covariance=endpoints([[columns[j][i] for j in range(len(matrix))] for i in range(len(matrix))])
            self.mean=endpoints([[v] for v in solve_spd(factor,rhs)])
            self.diagnostics=dict(solver=solver,coefficient_dimension=len(matrix))
        else:
            raise ValueError("unknown rigorous solver")

    def joint_parameters(self,queries):
        queries=tuple(queries)
        p=self.precision
        n=len(self.post.candidate_ids)*5
        zero=point(0,p)
        rows=[]
        for q in queries:
            self.post.query_features((q,))
            arm=self.post.candidate_ids.index(q.candidate_id)
            row=[zero]*n
            row[5*arm:5*arm+5]=basis_iv(q.epoch,self.post.snapshot,p)
            rows.append(row)
        if not rows:
            return MPFRJointParameters((),(),(),())
        h=endpoints(rows)
        lo,hi=interval_product(h,self.mean,p)
        means=tuple(IV(lo[i,0],hi[i,0],p)+point(self.post.snapshot.mean,p) for i in range(len(rows)))
        mid=interval_product(h,self.covariance,p)
        clo,chi=interval_product(mid,(h[0].T,h[1].T),p)
        if any(chi[i,i] < 0 for i in range(len(rows))):
            raise ArithmeticError("wholly negative latent variance enclosure")
        latent=tuple(tuple(IV(clo[i,j],chi[i,j],p).maximum(0) if i==j else IV(clo[i,j],chi[i,j],p)
                           for j in range(len(rows))) for i in range(len(rows)))
        observed=tuple(tuple(v+noise_iv(queries[i].epoch,self.post.snapshot,p) if i==j else v
                            for j,v in enumerate(row)) for i,row in enumerate(latent))
        return MPFRJointParameters(queries,means,latent,observed)
