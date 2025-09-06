from QCLDPC_sampler import *
from submatrix_sampler import *
from dubiner_sparsifier import sparsify, sparsify_numba
from block_recover import *
from verifier import *
from formatter import *
from util import *
from bit_flip_decoder_sequential import ldpc_bitflip_seqdecode_numba


import time
start_time = time.time()

'''
Implementation of LDPC PCM recovering method of paper:
Progressive reconstruction of QC LDPC matrix in a noisy channel

더 개선할 점
1) Do a sanity check? 
=> Check amount of error a dual vector detects from the codeword 
=> if too many, it means it is a wrong dual vector(even when there is noise), so reject

SANITY CHECK STEPS
1. grab one dual vector 
2. get block shifts of a dual vector
3. with all block shifts, check how much error it detects
4. If detected error is higher than 2*error rate, reject the dual vector 
(a dual vector may only detect erros less than real error rate even when there are noise. Hence, if there are reasonably many errors, it is fake one)


2) Check rank before adding
block shift 한 뒤에 새로 얻은 dual vector가 이전의 dual vector에 속한다면 추가할 필요가 없음 
linearly independent한 놈을 쓸때 의미가 있는거다

3) dubiner에서 col swap하기 
왜 하는건지 모르겠는데, 이걸 하면 성능이 좋아지나? 틀린 답을 너무 많이 찾아낼 것 같은데

4) how to reduce false alarm? 
즉 dual vector가 아니면서 찾아지는 벡터를 최대한 없도록 하고싶다
1. dubiner sparsification에서 더 strict하게 버킷을 검사? 
2. 더 많은 col을 검사 수행

'''

skip_generation = True
error_free = False
get_all_block_shifts = True
SHOW_TIME = False

if error_free:
    noise_level = 0
H_current_rank = 0 # initialize rank variable

if skip_generation:
    H = read_matrix("H_true")
    A = read_matrix("noisy_codeword")
    A_error_free = read_matrix('error_free_codeword')
else:
    # 1. sample LDPC code word
    H, B, S = generate_nand_qc_ldpc(mb=mb, nb=nb, Z=Z, target_rate=target_rate, rng=1234, sparse=True) #
    save_matrix(H, 'H_true')
    H_diag = diag_format(H, databit_num)
    generator = get_generator(H,databit_num)
    # print(generator.shape)

    A = get_codewords(generator,codeword_len, databit_num,pooling_factor = pooling_factor,noise_level = noise_level,save_noise_free=True)
    A_error_free = read_matrix('error_free_codeword')
    save_matrix(A, filename='noisy_codeword')

    # print("Base size (mb x nb):", B.shape)
    # print("Lifting Z:", Z)
    # print("Full H shape:", (mb * Z, nb * Z))
    save_image_data(H, filename="n_{}_k_{}".format(codeword_len, databit_num))

if not LARGE_CODE:
    print("H matrix: ")
    print_arr(H)
    print("Code word matrix: ")
    print_arr(A)
else:
    print("Code word generated")
total_codewords = pooling_factor * codeword_len
correct_codewords = compare_matrix(A_error_free, A)
print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))


print("Elapsed time: %s seconds" % round(time.time() - start_time,3))
#############################################################################################
# Iterative decoding

decoding_codeword_matrix = np.copy(A)

### mesh test ####
init_header(["col_factor","row_factor","found vectors","correct vectors" ])
mesh_test = []
mesh_step = 0.1 # 0.05
for i in range(7):
    for j in range(7):
        mesh = (0.25 + mesh_step * i, 0.25 + mesh_step * j)
        mesh_test.append(mesh)
tx, ty = H.shape
### mesh test ####

for mesh_unit in mesh_test:
    col_factor,row_factor = mesh_unit
    print('#' * 30)
    print("Mesh: ",mesh_unit)
    print('#' * 30)
    # target H matrix
    H_final = None

    decode_num = 20 # loop N times
    for i in range(decode_num):
        print()
        print('#'*30)
        print("Iterative Decoding Loop ", i+1)

        # 2. submatrix sampling : progressive reconstruction of ldpc code 논문의 fig2 참조
        ns = round(col_factor * codeword_len)  # number of sampled col
        ms = round(row_factor * ns)  # number of sampled row

        row_indices, col_indices = sample_row_col_indices(A, ms, ns)
        if error_free:
            sub = sample_submatrix(A, row_indices, col_indices)
        else:
            sub = sample_submatrix(decoding_codeword_matrix, row_indices, col_indices)
        if SHOW_TIME:
            print("Submatrix sampled", end='')
            print(" %s seconds" % round(time.time() - start_time, 3))

        # 3. gaussian elimination 으로 복원하는 방법을 사용 - 이거 말고 (검증된) ECO 방식을 써볼까?
        Gs = gf2elim(sub)
        P = Gs[:, ms:]
        P_t = np.transpose(P)
        Hs = np.concatenate((P_t, np.identity(ns - ms, dtype=np.uint8)), axis=1)
        if SHOW_TIME:
            print("Gauss elimination", end='')
            print(" %s seconds" % round(time.time() - start_time, 3))

        # 4.dubiner sparsification
        # Hr = sparsify(Hs, ms, ns, column_swap=False) # ################################### col swap을 true로 하고 실험함. False로 바꿔주기
        ds = round(ms / 2)  # hashing 강도
        Hr = sparsify_numba(Hs, ms, ns, ds)

        if Hr.shape[0]==0: # no vector found
            print("No sparse vector found")
            continue
        mr, nr = Hr.shape
        if SHOW_TIME:
            print("Dubiner sparsification", end='')
            print(" %s seconds" % round(time.time() - start_time, 3))

        # 5. pad Hs to be length n dual vectors, which is a format of H
        H_recovered = np.zeros((mr, codeword_len),
                               dtype=np.uint8)
        H_recovered[:, col_indices] = Hr

        # check block distances => there should be only one 1 per row in a block -> hence there should not be 1's in the same block range
        H_candidate = []
        for i in range(len(H_recovered)):
            add_vector = True
            h = H_recovered[i]
            # check whether there exists multiple 1's in a block, if so, delete it
            num_blocks_per_row = int(codeword_len/Z)
            for p in range(num_blocks_per_row):
                row_block = h[p*Z:(p+1)*Z]
                if sum(row_block) > 1: # invalid vector!
                    add_vector = False
            if add_vector:
                H_candidate.append(h)
        if SHOW_TIME:
            print("block constraint checking", end='')
            print(" %s seconds" % round(time.time() - start_time, 3))

        H_request = [] # Linearly independent (w.r.t H_final) candidate from H_candidate
        # check whether new recovered vector in H_candidate is L.I with H_final
        if H_final is None:
            H_current_rank = np.linalg.matrix_rank(H_candidate)
            H_request = H_candidate
        else:
            for i in range(len(H_candidate)):
                h = H_candidate[i]
                Htemp = np.append(H_final, [h], axis=0)
                if np.linalg.matrix_rank(Htemp) > H_current_rank:  # increase rank
                    H_request.append(h)
                    H_current_rank += 1
        if SHOW_TIME:
            print("New vector's rank checking", end='')
            print(" %s seconds" % round(time.time() - start_time, 3))

        if not H_request:
            print("No candidate dual vector found: Either not in QC format or Linearly dependent")
            continue

        H_request = np.array(H_request)
        # 6. get all the block shifts of blocks
        if get_all_block_shifts:
            for dual_vector in H_request: # for dual_vector in H_candidate: - sample L.I ones
                shifts = qc_global_cyclic_shifts_numba(dual_vector, Z)  # shift to get whole block (block size is given, Z)
                if H_final is None:
                    H_final = np.array(shifts)
                else:
                    H_final = np.concatenate((H_final, shifts), axis=0)
            if SHOW_TIME:
                print("Get block shift", end='')
                print(" %s seconds" % round(time.time() - start_time, 3))
        else: # no shift
            if H_final is None:
                H_final = H_request
            else:
                H_final = np.concatenate((H_final, H_request), axis=0)

        if not error_free:
            # 7. decoding using hard decision bit flip
            decoded_codeword_matrix, ok, _, _, _ = ldpc_bitflip_seqdecode_numba(H_final, A, max_iter=50)
            if SHOW_TIME:
                print("Decoding complete", end=' - ')
                print(" %s seconds" % round(time.time() - start_time, 3))
            correct_codewords = compare_matrix(A_error_free, decoded_codeword_matrix)
            print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))
            decoding_codeword_matrix = np.copy(decoded_codeword_matrix)

        print("Success?: ", check_success(H, H_final))
        print("Total elapsed time: %s seconds" % round(time.time() - start_time, 3))

    found_vectors = 0
    correct_guess = 0
    if not (H_final is None): # if not None
        correct_guess = compare_matrix(H, H_final)
        found_vectors, _ = H_final.shape

    datarow = [col_factor,row_factor, found_vectors ,correct_guess]
    save_recovery_data_row_csv(datarow)




