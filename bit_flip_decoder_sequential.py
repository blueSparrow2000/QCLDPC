import numpy as np
from variables import *
import time
from verifier import compare_matrix
from numba import njit, prange

@njit(parallel=True)
def ldpc_bitflip_seqdecode_numba(H, codewords, max_iter=100, weight_tolerance=2):
    """
    Parallel Gallager-B sequential bit-flip LDPC decoder (Numba, prange).
    Flips one bit at a time: the variable with the strongest error evidence.
    Rollback rules:
      - if final Hamming weight > original + tolerance, revert
      - if decoding fails, revert
    """

    C = codewords.copy()
    B, n = C.shape
    m, nH = H.shape

    decoded = np.empty((B, n), dtype=np.uint8)
    success = np.zeros(B, dtype=np.bool_)
    iters = np.zeros(B, dtype=np.uint8)
    syndromes = np.zeros((B, m), dtype=np.uint8)
    flips = np.zeros(B, dtype=np.uint8)

    # Precompute deg_v and tau
    deg_v = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        sdeg = 0
        for i in range(m):
            if H[i, j] != 0:
                sdeg += 1
        deg_v[j] = sdeg
    tau = deg_v // 2

    for b in prange(B):
        w = C[b].copy()
        orig_w = C[b].copy()
        total_flips = 0
        finished = False

        for it in range(1, max_iter + 1):
            # compute syndrome s
            s = np.zeros(m, dtype=np.uint8)
            nonzero = 0
            for i in range(m):
                acc = 0
                for j in range(n):
                    if H[i, j] != 0:
                        acc ^= (w[j] & 1)
                s[i] = acc
                if acc != 0:
                    nonzero += 1

            if nonzero == 0:  # success
                if np.sum(w) > np.sum(orig_w) + weight_tolerance:
                    decoded[b, :] = orig_w
                    success[b] = False
                else:
                    decoded[b, :] = w
                    success[b] = True
                iters[b] = it - 1
                syndromes[b, :] = s
                flips[b] = total_flips
                finished = True
                break

            # compute unsatisfied counts u
            u = np.zeros(n, dtype=np.uint8)
            for j in range(n):
                if deg_v[j] > 0:
                    acc = 0
                    for i in range(m):
                        if H[i, j] != 0 and s[i] != 0:
                            acc += 1
                    u[j] = acc

            # choose the "most error" variable to flip
            umax = 0
            chosen = -1
            for j in range(n):
                if u[j] > umax and u[j] > tau[j]:
                    umax = u[j]
                    chosen = j

            if chosen >= 0:
                w[chosen] = 1 - w[chosen]
                total_flips += 1
            else:
                # stuck → rollback and exit
                decoded[b, :] = orig_w
                success[b] = False
                iters[b] = it
                syndromes[b, :] = s
                flips[b] = total_flips
                finished = True
                break

            # reached max_iter
            if it == max_iter:
                s = np.zeros(m, dtype=np.uint8)
                nonzero = 0
                for i in range(m):
                    acc = 0
                    for j in range(n):
                        if H[i, j] != 0:
                            acc ^= (w[j] & 1)
                    s[i] = acc
                    if acc != 0:
                        nonzero += 1
                success[b] = (nonzero == 0)
                iters[b] = it
                syndromes[b, :] = s
                if success[b] and (np.sum(w) <= np.sum(orig_w) + weight_tolerance):
                    decoded[b, :] = w
                else:
                    decoded[b, :] = orig_w
                    success[b] = False
                flips[b] = total_flips
                finished = True
                break

        if not finished:
            decoded[b, :] = orig_w
            success[b] = False
            iters[b] = max_iter
            # recompute final syndrome
            s = np.zeros(m, dtype=np.uint8)
            for i in range(m):
                acc = 0
                for j in range(n):
                    if H[i, j] != 0:
                        acc ^= (w[j] & 1)
                s[i] = acc
            syndromes[b, :] = s
            flips[b] = total_flips

    return decoded, success, iters, syndromes, flips

