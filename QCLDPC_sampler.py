"""
nand_qc_ldpc_gen.py

Generate NAND-style QC-LDPC parity-check matrices (protograph + QC lifting).

Defaults are conservative for experimentation; tune mb, nb, Z for your target rate/blocklength.

Dependencies: numpy, scipy (for sparse option)


 ######## How to use / tune

mb, nb: set to match the desired base rate ~ 1 - mb/nb. Example: mb=10, nb=100 → rate 0.90.

Z: lifting factor. NAND controllers use large block sizes (≥ 256) in production; for experiments pick 64–512. Larger Z helps shift assignment and performance.

avg_col_w: tweak to increase/decrease average column weight. Larger column weight -> stronger code but more decoder complexity.

sparse=True: recommended for realistic sizes. If SciPy is not installed or you set sparse=False, the generator will return a dense NumPy array (only do this for very small examples).

Final notes

This generator approximates NAND-style designs (irregular QC-LDPC, high rate). Actual vendor matrices are proprietary and optimized to flash channel statistics and hardware constraints.

If you tell me a target n (final code length) and target rate, I can show you concrete mb, nb, Z choices and produce a sample H you can test with your decoder.
"""
import numpy as np
from gauss_elim import *
from matrix_mul import matmul_f2
from variables import NOISE_PROB
from util import *
from formatter import diag_format

# Optional: SciPy for sparse matrix output
try:
    import scipy.sparse as sp

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# -------------------------
# Helper: make irregular column degree list
# -------------------------
def make_nand_like_col_degrees(nb, mb, target_rate=0.93, avg_col_w=None, rng=None):
    """
    Produce an irregular column-degree list length nb resembling NAND LDPC:
      - many deg=2 or 3, some deg=4..8
      - ensure each degree <= mb (no parallel edges in base)
    Parameters
    ----------
    nb : int
        Number of base variable nodes (columns).
    mb : int
        Number of base check nodes (rows). Used to cap max degree.
    target_rate : float
        Desired code rate approx 1 - mb/nb (informational).
    avg_col_w : float or None
        If set, try to make average column weight ~ avg_col_w. Otherwise pick sensible default ~3.
    rng : int or numpy.random.Generator or None
        RNG seed.
    """
    rng = np.random.default_rng(rng)
    if avg_col_w is None:
        avg_col_w = 3.0  # NAND often uses low-weight variable nodes but some heavier ones

    # We'll create a mixture distribution:
    #   - 50-70% columns weight 2
    #   - 20-40% columns weight 3
    #   - 5-15% columns weight 4-8
    p2 = 0.6
    p3 = 0.3
    phigh = 0.1

    degs = []
    for _ in range(nb):
        r = rng.random()
        if r < p2:
            d = 2
        elif r < p2 + p3:
            d = 3
        else:
            # heavy tail 4..min(8,mb)
            d = rng.integers(4, min(8, mb) + 1)
        # cap by mb
        d = min(d, mb)
        degs.append(int(d))

    # adjust average if too far from avg_col_w
    cur_avg = sum(degs) / nb
    if cur_avg < avg_col_w:
        # increase some degrees randomly
        deficit = int(round((avg_col_w - cur_avg) * nb))
        for i in rng.choice(nb, size=min(deficit, nb), replace=False):
            if degs[i] < mb:
                degs[i] += 1
    elif cur_avg > avg_col_w:
        # try to reduce some heavy ones
        heavies = [i for i, d in enumerate(degs) if d > 2]
        for i in rng.choice(heavies, size=min(len(heavies), int(round((cur_avg - avg_col_w) * nb))), replace=False):
            if degs[i] > 2:
                degs[i] -= 1

    # Final safety cap
    degs = [min(d, mb) for d in degs]
    return degs


# -------------------------
# PEG protograph generator (no parallel edges)
# -------------------------
def peg_protograph(mb, nb, col_degrees, rng=None):
    rng = np.random.default_rng(rng)
    B = np.zeros((mb, nb), dtype=np.uint8)
    row_deg = np.zeros(mb, dtype=int)

    var_neighbors = [set() for _ in range(nb)]
    chk_neighbors = [set() for _ in range(mb)]

    from collections import deque
    def bfs_unreached_checks(start_var):
        reached_chk = set()
        q = deque()
        for c in var_neighbors[start_var]:
            if c not in reached_chk:
                reached_chk.add(c)
                q.append(('chk', c))
        reached_var = {start_var}
        while q:
            kind, node = q.popleft()
            if kind == 'chk':
                for v in chk_neighbors[node]:
                    if v not in reached_var:
                        reached_var.add(v)
                        q.append(('var', v))
            else:
                for c in var_neighbors[node]:
                    if c not in reached_chk:
                        reached_chk.add(c)
                        q.append(('chk', c))
        return reached_chk

    for v in range(nb):
        dv = int(col_degrees[v])
        if dv < 0:
            raise ValueError("col_degrees must be nonnegative")
        if dv > mb:
            raise ValueError(f"col_degrees[{v}] = {dv} exceeds mb = {mb} (no parallel edges allowed)")
        for e in range(dv):
            candidates = [r for r in range(mb) if B[r, v] == 0]
            if not candidates:
                raise ValueError(f"No candidate rows left for column {v} during PEG. Increase mb or lower degree.")
            if len(var_neighbors[v]) == 0:
                min_deg = np.min(row_deg[candidates])
                best = [r for r in candidates if row_deg[r] == min_deg]
                rsel = rng.choice(best)
            else:
                reached = bfs_unreached_checks(v)
                unreached = [r for r in candidates if r not in reached]
                pool = unreached if len(unreached) > 0 else candidates
                min_deg = np.min(row_deg[pool])
                best = [r for r in pool if row_deg[r] == min_deg]
                rsel = rng.choice(best)
            B[rsel, v] = 1
            row_deg[rsel] += 1
            var_neighbors[v].add(rsel)
            chk_neighbors[rsel].add(v)
    return B


# -------------------------
# Assign circulant shifts (avoid 4-cycles using difference constraint)
# -------------------------
def assign_circulant_shifts(B, Z, rng=None, max_tries=2000):
    rng = np.random.default_rng(rng)
    B = np.array(B, dtype=np.uint8)
    mb, nb = B.shape
    S = -np.ones_like(B, dtype=int)

    ones_in_row = [np.flatnonzero(B[r, :]) for r in range(mb)]
    ones_in_col = [np.flatnonzero(B[:, c]) for c in range(nb)]

    edges = [(r, c) for r in range(mb) for c in range(nb) if B[r, c] == 1]
    rng.shuffle(edges)

    for attempt in range(max_tries):
        if attempt > 0:
            S.fill(-1)
            rng.shuffle(edges)
        ok = True
        for (r, c) in edges:
            candidates = rng.permutation(Z)
            placed = False
            for s in candidates:
                valid = True
                # check all 2x2 submatrices that include (r,c)
                for c2 in ones_in_row[r]:
                    if c2 == c or S[r, c2] < 0:
                        continue
                    for r2 in ones_in_col[c]:
                        if r2 == r or S[r2, c] < 0 or B[r2, c2] == 0 or S[r2, c2] < 0:
                            continue
                        diff = (s - S[r, c2] - S[r2, c] + S[r2, c2]) % Z
                        if diff == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S[r, c] = int(s)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return S
    raise RuntimeError("Failed to assign circulant shifts without 4-cycles; try larger Z or different B.")


# -------------------------
# Build H (sparse or dense)
# -------------------------
def circulant_perm_sparse(Z, shift):
    rows = np.arange(Z, dtype=int)
    cols = (rows + shift) % Z
    data = np.ones(Z, dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=(Z, Z))


def circulant_perm_dense(Z, shift):
    I = np.eye(Z, dtype=np.uint8)
    return np.roll(I, int(shift) % Z, axis=1)


# ---------- helper: circulant ----------
def circulant_perm_array(Z, shift):
    I = np.eye(Z, dtype=np.uint8)
    return np.roll(I, int(shift) % Z, axis=1)


# ---------- build QC-LDPC H (numpy array) ----------
def build_qc_ldpc_array(B, S, Z):
    """
    Build QC-LDPC H matrix (NumPy ndarray of 0/1).
    B : base matrix (0/1 ndarray)
    S : shift matrix (integers, -1 means no edge)
    Z : lifting size
    Returns: H (ndarray, dtype=uint8)
    """
    mb, nb = B.shape
    M, N = mb * Z, nb * Z
    H = np.zeros((M, N), dtype=np.uint8)
    for r in range(mb):
        for c in range(nb):
            if B[r, c] == 1:
                H[r * Z:(r + 1) * Z, c * Z:(c + 1) * Z] = circulant_perm_array(Z, S[r, c])
    return H


# -------------------------
# High-level generator
# -------------------------
def generate_nand_qc_ldpc(
        mb=None, nb=None, Z=128, target_rate=0.93, nb_auto=50, mb_auto=None,
        avg_col_w=3.0, rng=None, sparse=True, col_degrees=None
):
    """
    Generate a NAND-like QC-LDPC.

    Parameters
    ----------
    mb, nb : int or None
        Base rows/cols. If None, they'll be chosen from nb_auto and target_rate.
    Z : int
        Lifting factor (circulant size).
    target_rate : float
        Desired approximate code rate (1 - mb/nb).
    nb_auto : int
        Default base columns if nb is None.
    mb_auto : int or None
        Default base rows = round(nb_auto*(1 - target_rate)) if None.
    avg_col_w : float
        Target average column weight in base graph.
    rng : seed or Generator
    sparse : bool
        Return SciPy CSR matrix if True and SciPy available, else dense numpy array.
    col_degrees : list or None
        If provided, use this list (length nb). Otherwise generate NAND-like degrees.

    Returns
    -------
    H, B, S
      H : parity-check matrix (sparse CSR or dense numpy)
      B : base 0/1 matrix (mb x nb) numpy
      S : shift matrix (mb x nb) ints where B==1, -1 where B==0
    """
    rng = np.random.default_rng(rng)
    if nb is None:
        nb = nb_auto
    if mb is None:
        if mb_auto is None:
            mb = max(1, int(round(nb * (1 - target_rate))))
        else:
            mb = mb_auto

    if col_degrees is None:
        col_degrees = make_nand_like_col_degrees(nb, mb, target_rate=target_rate, avg_col_w=avg_col_w, rng=rng)
    else:
        if len(col_degrees) != nb:
            raise ValueError("col_degrees length must equal nb")

    # Safety: every col degree <= mb
    if any(d > mb for d in col_degrees):
        raise ValueError("Each col degree must be <= mb (no parallel edges in base)")

    B = peg_protograph(mb, nb, col_degrees, rng=rng)
    S = assign_circulant_shifts(B, Z, rng=rng)

    H = build_qc_ldpc_array(B, S, Z)
    return H, B, S


def get_codewords(generator, n, k, pooling_factor=2, noise_level=0, save_noise_free=False):
    global NOISE_PROB
    # sample message bits - each row is a message bit
    M = round(pooling_factor * n)  # 10 messages
    message_bits = np.random.choice([0, 1], size=(M, k), p=[1. / 2, 1. / 2])  # np.identity(k, dtype=int)

    ################################## matmul needs to be faster only here ###############################################
    code_words = matmul_f2(message_bits, generator)  # message_bits@generator
    ################################## matmul needs to be faster only here ###############################################

    message_rank = np.linalg.matrix_rank(message_bits)
    code_rank = np.linalg.matrix_rank(code_words)
    if k > code_rank:
        if k > message_rank:
            print(message_rank)
            print("[WARNING] message rank too small! need more data")
        else:
            print(code_rank)
            print("[WARNING] generator matrix produces degenerate code!")

    if save_noise_free:
        save_matrix(code_words, filename='error_free_codeword')

    if noise_level == 1:  # add one bit noise
        code_words[0, -1] = not code_words[0, -1]  # add noise to only the last bit of the first row
        # code_words[-1, - 1] = not code_words[-1,- 1]  # add noise to only the last bit of the last row
    elif noise_level == 2:  # add two bit noise
        code_words[0, -1] = not code_words[0, -1]  # add noise to only the last bit of the first row
        code_words[-1, - 1] = not code_words[-1, - 1]  # add noise to only the last bit of the last row
    elif noise_level == 10:  # add gaussian noise to code_words matrix
        G_noise = sp.random(M, n, density=NOISE_PROB, data_rvs=np.ones).toarray().astype(np.uint8)
        # print_arr(G_noise)
        num_of_error_bits = (G_noise == 1).sum()
        print("Number of error bits: %d" % num_of_error_bits)
        code_words = code_words ^ G_noise

    return code_words


def get_generator(H, k):
    I = np.identity(k, dtype=np.uint8)
    H_diag = diag_format(H, k)  # ECO 에서 degenerate가 나오면, 즉 0인 row가 나오면 없애버리기 때문에 이래된다
    if (H.shape[0] != H_diag.shape[0]):
        raise Exception("H has degenerate rows! There are zero rows after applying ECO to H!")
    P = H_diag[:, :k]  # front part
    # print(P.shape)
    generator = np.concatenate((I, P.T), axis=1)  # generator mat
    return generator


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__1":
    # Example: produce a small-ish NAND-like code for experimentation
    mb = 4  # base checks
    nb = 16  # base vars -> base rate ~ 0.9
    Z = 64  # lifting factor (choose 64,128,256,...)
    '''
    About size of Z
    https://www.sciencedirect.com/topics/computer-science/block-size-b

    block size Z 는 6144 ~ 8192정도가 좋으며(nand flash) 보통 power of 2 (2^n)으로 한다
    '''
    target_rate = 1 - mb / nb

    H, B, S = generate_nand_qc_ldpc(mb=mb, nb=nb, Z=Z, target_rate=target_rate, rng=1234, sparse=True)
    n = nb * Z
    k = n - mb * Z

    print("Base size (mb x nb):", B.shape)
    print("Lifting Z:", Z)
    print("Full H shape:", (mb * Z, nb * Z))
    # Show small slices of B and S
    print("B (first 8 rows/cols):")
    print(B[:8, :16])
    print("S (first 8 rows/cols):")
    print(S[:8, :16])

    save_image_data(H, filename="n_{}_k_{}".format(n, k))


