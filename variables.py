'''
Some variables to input
'''
import numpy as np

mb = 4#4  # base checks -> related to parity vectors
nb = 16# 9 #16  # base vars -> base rate ~ 0.9
Z = 64# 16#64  # lifting factor (choose 64,128,256,...)
target_rate = 1 - mb / nb
codeword_len = nb * Z
databit_num = codeword_len - mb * Z

density = 0.15  # number of ones in a P matrix
if codeword_len > 500:
    density = 0.08
elif codeword_len >= 2000:
    density = 0.05

LARGE_CODE = True if codeword_len > 50 else False
parity_num = codeword_len-databit_num
noise_level= 10
pooling_factor = 3 #7 # magic number - best
BGCE = True # without BGCE, one may get more dual vectors but they are likely to be erronous
threshold = round(((pooling_factor-1)*codeword_len)*0.325)  # suggested beta coeff on the paper
if not BGCE: # if GCE, higher the threshold
    threshold = round((pooling_factor*codeword_len)*0.325) # codeword_amount/2 * beta_opt (0.65, based on experience)

NOISE_PROB = 0.0001#0.0001# 0.0001

np.random.seed(seed=0) # 6: erronous seed

PRINT_VAR_SETTING = True


# print variable settings
if PRINT_VAR_SETTING:
    print('#' * 5, "variable settings", '#' * 5)
    print("(n, k) = ({}, {})".format(codeword_len, databit_num))
    print("Number of code words: ", pooling_factor*codeword_len)
    print("Threshold: ", threshold)
    print('#' * 30)