'''
Parallelized version with Numba.
Generate all global cyclic shifts of a QC-LDPC parity check vector,
where each block (size p) is shifted independently mod p,
but all blocks are shifted simultaneously.

Parameters
----------
vec : 1D numpy array of {0,1}, length n
p : int, block size

Returns
-------
shifts : 2D numpy array of shape (p, n)
         shifts[s, :] = vector after shift by s
'''
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def qc_global_cyclic_shifts_numba(vec, p):
    n = vec.shape[0] #n = len(vec)
    if n % p != 0:
        raise ValueError("Vector length must be multiple of block size p")
    num_blocks = n // p

    shifts = np.zeros((p, n), dtype=np.uint8)
    for s in prange(p):  # shift amount
        for b in range(num_blocks):  # block index
            for j in range(p):  # inside block
                src_idx = b * p + j
                dst_idx = b * p + ((j + s) % p)
                shifts[s, dst_idx] = vec[src_idx]

    return shifts

# small test example
if __name__ == "__main__1":
    p = 4
    n = 4 * p
    vec = np.zeros(n, dtype=np.uint8)
    vec[2] = 1   # block 0, pos=0
    vec[4] = 1   # block 1, pos=0
    vec[9] = 1   # block 2, pos=1

    shifts = qc_global_cyclic_shifts_numba(vec, p)

    print("Original:", vec)
    for i in range(p):
        print(f"Shift {i}:", shifts[i])


