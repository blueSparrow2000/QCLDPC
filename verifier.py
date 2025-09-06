import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compare_matrix(target, source):
    tx, ty = target.shape
    sx, sy = source.shape
    correct_guess = 0

    for i in prange(tx):  # parallel over target rows
        found = False
        for j in range(sx):
            match = True
            for k in range(ty):
                if target[i, k] != source[j, k]:
                    match = False
                    break
            if match:
                found = True
                break
        if found:
            correct_guess += 1

    return correct_guess

'''
Compares element wise between two matrix
Return True if identical
'''
def check_success(target_array,my_array):
    if my_array is None: # failed to recover any dual vectors
        print("Failed to recover any dual vector!")
        return False
    tx,ty = target_array.shape
    mx,my = my_array.shape

    correct_guess = 0
    for i in range(tx):
        if (my_array == target_array[i]).all(axis = 1).any():
            correct_guess += 1

    if tx > mx: # too small!
        print("Dual vectors missing: ", tx-mx)
        print("correct guess: {} / {}".format(correct_guess,tx))
        return False
    if tx == mx and (target_array == my_array).all():
        # perfectly correct
        return True

    if correct_guess== tx: # found all but too many
        print("Found all {} parity check vectors".format(tx))
        print("Total dual vectors recovered: {}".format(mx))
        return False

    print("Total dual vectors recovered: {}".format(mx))
    print("correct guess: {} / {}".format(correct_guess,tx))
    return False

