import numpy as np
from formatter import diag_format
from variables import density, databit_num, codeword_len
import random
from numba import njit, prange

'''
This is an implementation of algorithm 1 in paper "Progressive reconstruction of qc ldpc ..."

Hyper parameter (not shown in the paper)
W_th : weight threshold - for QC LDPC code with unkown w, how can we determine w value?
d_s  : how strictly to bucket the codes
N_iter (can be approximately calculated)

문제 => 답이 아닌 vector가 많이 찾아진다. 
column swap을 해도 되는것인가!

col swap을 안해도 너무 많이 찾아짐... 하하


더비너만 numba parallelism이 구현 안되어있네! 그래서 느림!
'''
 #
def get_w_th():
    global density, databit_num
    return (density*databit_num)  # weight threshold

def get_N_iter():
    global databit_num, codeword_len
    return 100#codeword_len-databit_num # approximate to n-k

def sparsify(Hs, ms,ns, column_swap = True):
    ds = round(ms/2) # hashing 강도
    w_th= 12 #get_w_th()
    Niter = get_N_iter()
    # if not column_swap:
    #     Niter = 1

    Hr = []

    # if a vector is sparse enough from the beginning, take it
    for i in range(ns-ms): # for each row in Hs
        s = Hs[i]
        s_test = s.astype(np.uint64)
        if 0 < sum(s_test) <= w_th:
            Hr.append(s)  # append sparse vectors

    for t in range(Niter):
        bucket = []
        hash = np.random.choice([0, 1], size=(ms,), p=[1./2, 1./2])

        hash_survive_idx = np.array([True] * ds + [False] * (ms-ds))
        hash_indicies = np.arange(ms)

        hash_idx = hash_indicies[np.random.permutation(hash_survive_idx)]
        for i in range(ns-ms):
            h = Hs[i] # get i th row of Hs
            databit_h = h[:ms]
            # print(databit_h)
            # print(hash)

            if (databit_h[hash_idx] == hash[hash_idx]).all(): # check hash of databits
                bucket.append(h)

        bn = len(bucket)
        # print("collisions: ",bn)
        for i in range(bn-1):
            for j in range(i+1, bn):
                s = bucket[i] ^ bucket[j]
                s_test = s.astype(np.uint64)
                if 0 < sum(s_test) <= w_th:# overflow error? why? uint8 -> 헐 255까지야? 아 그럼 그럴 수 있지... 헐... 생각보다 작네 ㅇㅇ
                    Hr.append(s) # append sparse vectors

        if column_swap: # 이게 True면 시간 많이 걸림
            j = random.randint(0,ms-1)
            k = random.randint(ms,ns-1)
            # swap col of Hc
            temp = np.copy(Hs[:,j])
            Hs[:, j] = Hs[:, k]
            Hs[:, k] = temp
            # format Hs into H matrix format
            Hs = diag_format(Hs, ms)
    return np.array(Hr)


########## parallel version ###########
@njit(parallel=True)
def sparsify_numba(Hs, ms, ns, ds, w_th=12, Niter=100):
    max_candidates = Niter * 100  # rough upper bound (need tuning)
    Hr_tmp = np.zeros((max_candidates, ns), dtype=np.uint8)
    Hr_count = 0

    # if a vector is sparse enough from the beginning, take it
    for i in range(ns-ms): # for each row in Hs
        s = Hs[i]
        s_test = s.astype(np.uint64)
        if 0 < sum(s_test) <= w_th:
            if Hr_count < max_candidates: # append sparse vectors
                Hr_tmp[Hr_count] = s
                Hr_count += 1

    for t in prange(Niter):
        # get filter (to get similar vectors in one bucket)
        hash_vec = np.random.randint(0, 2, size=ms)  # random hash
        hash_idx = np.arange(ms)
        np.random.shuffle(hash_idx)
        hash_idx = hash_idx[:ds]

        bucket = []
        for i in range(ns - ms):
            h = Hs[i]
            databit_h = h[:ms]
            match = True
            for idx in hash_idx: # hash index에 대해서만 hash랑 값이 같은지 체크하는듯
                if databit_h[idx] != hash_vec[idx]:
                    match = False
                    break
            if match:
                bucket.append(h)

        bn = len(bucket)
        for i in range(bn - 1):
            for j in range(i + 1, bn):
                s = bucket[i] ^ bucket[j]
                s_test = s.astype(np.uint64)
                weight = np.sum(s_test)
                if 0 < weight <= w_th:
                    if Hr_count < max_candidates:
                        Hr_tmp[Hr_count] = s
                        Hr_count += 1
    # without col swap

    return Hr_tmp[:Hr_count]






