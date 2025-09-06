import numpy as np
import numba
from formatter import *

'''
Sample submatrix of M 
Input) M: matrix , k: number of databits
Out) Sampled submatrix according to sampling rule

'''

def sample_row_col_indices(M,ms,ns):
    m,n = M.shape

    row_survive_idx = np.array([True] * ms + [False] * (m-ms))
    row_indicies = np.arange(m)

    col_survive_idx = np.array([True] * ns + [False] * (n-ns))
    col_indicies = np.arange(n)

    sample_row_idx = row_indicies[np.random.permutation(row_survive_idx)]
    sample_col_idx = col_indicies[np.random.permutation(col_survive_idx)]
    # print(ms, ns)
    # print(sample_row_idx,sample_col_idx)
    return sample_row_idx, sample_col_idx


def sample_col_indices(M,ns, always_sample_parity_bits = 0):
    m,n = M.shape
    if always_sample_parity_bits>0: # parity bit number given
        p = always_sample_parity_bits
        col_survive_idx = np.array([True] * ns + [False] * (n - p - ns))
        col_index = np.arange(n - p)
        parity_index = np.arange(n - p, n)

        permute_index = col_index[np.random.permutation(col_survive_idx)]
        sample_col_idx = np.concatenate((permute_index, parity_index))
        return sample_col_idx

    col_survive_idx = np.array([True] * ns + [False] * (n-ns))
    col_indicies = np.arange(n)

    sample_col_idx = col_indicies[np.random.permutation(col_survive_idx)]
    return sample_col_idx


def sample_submatrix(M,sample_row_idx,sample_col_idx):
    submatrix = M[np.ix_(sample_row_idx,sample_col_idx)]

    return submatrix





