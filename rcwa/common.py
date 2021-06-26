import argparse
import os

import numpy as np
import toml

def matmul(*args):
    '''helper function that accepts an arbitrary number of matrices to multiply'''
    if len(args) == 1 or len(args) == 0:
        raise ValueError('Need at least two args')
    count = 0
    ret = None
    for arg in args:
        if count == 0:
            ret = arg
        else:
            ret = np.matmul(ret, arg)
        count += 1
    return ret

def redheffer_star_prod(sa_mat, sb_mat, unit_mat):
    Nharm = int(sa_mat.shape[0]/4)
    sa_11_mat = sa_mat[0:2*Nharm, 0:2*Nharm]
    sa_12_mat = sa_mat[0:2*Nharm, 2*Nharm:4*Nharm]
    sa_21_mat = sa_mat[2*Nharm:4*Nharm, 0:2*Nharm]
    sa_22_mat = sa_mat[2*Nharm:4*Nharm, 2*Nharm:4*Nharm]

    sb_11_mat = sb_mat[0:2*Nharm, 0:2*Nharm]
    sb_12_mat = sb_mat[0:2*Nharm, 2*Nharm:4*Nharm]
    sb_21_mat = sb_mat[2*Nharm:4*Nharm, 0:2*Nharm]
    sb_22_mat = sb_mat[2*Nharm:4*Nharm, 2*Nharm:4*Nharm]

    d_mat = matmul(sa_12_mat, np.linalg.inv(unit_mat - matmul(sb_11_mat, sa_22_mat)))
    f_mat = matmul(sb_21_mat, np.linalg.inv(unit_mat - matmul(sa_22_mat, sb_11_mat)))

    s_11_mat = sa_11_mat + matmul(d_mat, sb_11_mat, sa_21_mat)
    s_12_mat = matmul(d_mat, sb_12_mat)
    s_21_mat = matmul(f_mat, sa_21_mat)
    s_22_mat = sb_22_mat + matmul(f_mat, sa_22_mat, sb_12_mat)

    s_ret_mat = np.concatenate((np.concatenate((s_11_mat, s_12_mat), axis=1),
                                np.concatenate((s_21_mat, s_22_mat), axis=1)))
    return s_ret_mat

def get_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise FileNotFoundError('{} not a valid input file'.format(args.path))

    input_toml = toml.load(args.path)
    return input_toml
