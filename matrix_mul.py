import numpy as np
import numba

@numba.njit(fastmath=True, cache=True, parallel=True)

def matmul_f2(m1, m2):
    mr = np.empty((m1.shape[0], m2.shape[1]), dtype=np.bool_)
    for i in numba.prange(mr.shape[0]):
        for j in range(mr.shape[1]):
            acc = False
            for k in range(m2.shape[0]):
                acc ^= m1[i, k] & m2[k, j]
            mr[i, j] = acc
    return mr.astype(np.uint8)