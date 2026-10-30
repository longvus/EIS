"""Validated fast interval product used by the certificate's MPFR enclosures."""

import gmpy2 as g

import numpy as np

from mpfr_interval import _contexts

def sparse_dot(left,right):
    nonzero_left=left!=0
    nonzero_right=right!=0
    shape=(left.shape[0],right.shape[1])
    if not np.any(nonzero_left) or not np.any(nonzero_right):
        return np.full(shape,g.mpfr(0),dtype=object)
    # Sparsity dispatch is based solely on current exact matrix entries.
    row_density=np.count_nonzero(nonzero_left)/left.size
    col_density=np.count_nonzero(nonzero_right)/right.size
    if min(row_density,col_density)>.4:
        return left@right
    result=np.full(shape,g.mpfr(0),dtype=object)
    if row_density<=col_density:
        for row in range(shape[0]):
            indices=np.flatnonzero(nonzero_left[row])
            if len(indices):
                result[row,:]=left[row,indices]@right[indices,:]
    else:
        for column in range(shape[1]):
            indices=np.flatnonzero(nonzero_right[:,column])
            if len(indices):
                result[:,column]=left[:,indices]@right[indices,column]
    return result

def interval_product(left,right,precision):
    alo,ahi=left
    blo,bhi=right
    down,up=_contexts(precision)
    with g.context(precision=precision):
        am=(alo+ahi)/g.mpfr(2)
        bm=(blo+bhi)/g.mpfr(2)
    with g.context(up):
        ar=np.maximum(am-alo,ahi-am)
        br=np.maximum(bm-blo,bhi-bm)
        error=sparse_dot(np.abs(am),br)+sparse_dot(ar,np.abs(bm))+sparse_dot(ar,br)
    with g.context(down):
        lo=sparse_dot(am,bm)-error
    with g.context(up):
        hi=sparse_dot(am,bm)+error
    return lo,hi
