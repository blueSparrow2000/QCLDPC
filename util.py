import csv
from variables import * # need variables for filename
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def save_data(filename, data):
    with open('H_saves/%s.txt'%filename, 'w', newline='') as f:
        f.write(data)

def save_matrix(H, filename = ''):
    m,n = H.shape
    data = ""
    for i in range(m):
        for j in range(n):
            data += "%d"%(H[i,j])
        data += "\n"
    if filename =='':
        filename = "n_{}_k_{}".format(codeword_len, databit_num)
    save_data(filename, data)

# convert data into H matrix
def read_matrix(filename):
    H = []
    with open('H_saves/%s.txt'%filename, 'r', newline='') as f:
        lines = f.readlines()
        for line in lines:
            row = []
            line = line.strip()  # delete newline characters
            for bit in line:
                row.append(int(bit))
            H.append(row)

    return np.array(H)

# save img file of H matrix in binary image
def save_image_data(H, filename=None):
    if filename is None:
        filename = "n_{}_k_{}".format(codeword_len, databit_num)
    plt.imsave('H_saves/%s.png'%filename, H, cmap=cm.gray)


# this is to exaggerate the difference
def save_error_image(A, diff, mode = 'stripes'):
    m,n = A.shape
    if mode == 'stripes':
        for i in range(m):
            error = False
            for j in range(n):
                if diff[i,j]:
                    error = True
                    break
            if error:
                for j in range(n):
                    diff[i, j] = 1
        save_image_data(diff, filename="recovery_diff_stripes")

    elif mode=='blob':  # increase dot size => 3x3
        blob_radius = 2
        blob = np.zeros((m,n), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                if diff[i,j]:
                    i_min = max(0,i-blob_radius)
                    i_max = min(i+blob_radius+1, m)
                    j_min = max(0,j-blob_radius)
                    j_max = min(j+blob_radius+1, n)
                    for bi in range(i_min,i_max):
                        for bj in range(j_min, j_max):
                            blob[bi,bj] = 1
        save_image_data(blob, filename="recovery_diff_blob")


def write_csv_row(filename, datarow):
    with open('csv/%s.csv'%filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(datarow)

def append_csv_row(filename, datarow):
    with open('csv/%s.csv'%filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(datarow)


def write_csv_all(filename, data):
    with open('csv/%s.csv'%filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def save_recovery_data_row_csv(datarow):
    global codeword_len, databit_num
    filename = "n_{}_k_{}".format(codeword_len, databit_num) # _pooling_{}_thr_{}
    append_csv_row(filename, datarow)


def init_header(header_info):
    global codeword_len, databit_num
    filename = "n_{}_k_{}".format(codeword_len, databit_num) # _pooling_{}_thr_{}
    write_csv_row(filename, header_info)








