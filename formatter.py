import numpy as np
from gauss_elim import gf2elim

# pretty print
def print_arr(M):
    if len(list(M.shape))==1: # 1D array
        print(' '.join(map(str, M)))
        return

    m, n = M.shape
    for i in range(m):
        print(' '.join(map(str, M[i])))


'''
Format given matrix to H matrix form
which has identity matrix at the last n-k columns
'''
def diag_format(H, databit_num):
    p,n = H.shape
    H_before = np.concatenate((H[:, databit_num:],H[:,:databit_num]), axis=1)
    H_gauss = gf2elim(H_before)
    result = np.concatenate(( H_gauss[:, n-databit_num:],H_gauss[:, :n-databit_num]), axis=1)  #
    result = result[~np.all(result == 0, axis=1)] # remove zero rows
    return result
