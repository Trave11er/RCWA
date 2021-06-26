# coding: utf-8
import cmath

import numpy as np

from rcwa.common import matmul, redheffer_star_prod, get_input
from rcwa.structure import HomogeneousStructure
from rcwa.source import Source
from rcwa._constants import UNIT_MAT_2D

def save_outputs(R, T):
    with open('output.toml', 'w') as fid:
        fid.write('[R]\n00 = {:.4f}\n'.format(R))
        fid.write('[T]\n00 = {:.4f}\n'.format(T))
        fid.write('[R_T]\n00 = {:.4f}\n'.format(R + T))

class TMM():
    '''Calculates transmission through a stack of uniform layers'''
    def __prepare(self, structure, source):
        nr1 = np.sqrt(structure.UR1*structure.ER1)
        self.k_inc = source.K0*nr1*\
                     np.array(([np.sin(source.THETA)*np.cos(source.PHI),
                                np.sin(source.THETA)*np.sin(source.PHI), np.cos(source.THETA)]))
        S_global = np.array(([0, 0, 1, 0], [0, 0, 0, 1],
                             [1, 0, 0, 0], [0, 1, 0, 0]))
        return S_global

    def compute(self, structure, source):
        S_global = self.__prepare(structure, source)
        S_global = self.__compute_layers(structure, source, S_global)
        S_global = self.__compute_superstrate(structure, S_global)
        S_global = self.__compute_substrate(structure, S_global)
        R, T = self.__get_R_T(structure, source, S_global)
        return R, T

    def __compute_layers(self, structure, source, S_global):
        kx, ky = self.k_inc[0], self.k_inc[1]
        # take layers into account
        for i in range(0, structure.num_layers):
            ur = structure.ur_vec[i]
            er = structure.er_vec[i]
            l = structure.layer_thicknesses_vec[i]
            s_layer_mat = self.__calc_s_mat(l, ur, er, kx, ky, source.K0)
            S_global = redheffer_star_prod(S_global, s_layer_mat, UNIT_MAT_2D)
        return S_global
    @staticmethod
    def __calc_gap_layer_params(kx, ky):
        ur = 1
        er = 1
        q_mat = np.array(([kx*ky, ur*er+ky*ky], [-(ur*er+kx*kx), -kx*ky]))/ur
        v_mat = -1j*q_mat
        return v_mat

    @staticmethod
    def __calc_layer_params(ur, er, kx, ky):
        q_mat = np.array(([kx*ky, ur*er-kx*kx], [ky*ky-ur*er, -kx*ky]))/ur
        kz = cmath.sqrt(ur*er-kx*kx-ky*ky)
        omega_mat = 1j*kz*np.array(([1, 0], [0, 1]))
        v_mat = np.matmul(q_mat, np.linalg.inv(omega_mat))
        return omega_mat, v_mat

    def __calc_s_mat(self, layer_thickness, ur, er, kx, ky, K0):
        omegai_mat, vi_mat = self.__calc_layer_params(ur, er, kx, ky)
        vg_mat = self.__calc_gap_layer_params(kx, ky)
        ai_mat = UNIT_MAT_2D + np.matmul(np.linalg.inv(vi_mat), vg_mat)
        bi_mat = UNIT_MAT_2D - np.matmul(np.linalg.inv(vi_mat), vg_mat)
        xi_mat = np.diag(np.exp(np.diag(omegai_mat)*K0*layer_thickness))
        ai_inv_mat = np.linalg.inv(ai_mat)
        di_mat = ai_mat - matmul(xi_mat, bi_mat, ai_inv_mat, xi_mat, bi_mat)
        di_inv_mat = np.linalg.inv(di_mat)
        s_11_mat = matmul(di_inv_mat, matmul(
            xi_mat, bi_mat, ai_inv_mat, xi_mat, ai_mat) - bi_mat)
        s_12_mat = matmul(di_inv_mat, xi_mat, ai_mat
                          - matmul(bi_mat, ai_inv_mat, bi_mat))
        # S_12 = S_21, S_11 = S_22
        s_mat = np.concatenate((np.concatenate((s_11_mat, s_12_mat), axis=1),
                                np.concatenate((s_12_mat, s_11_mat), axis=1)))
        return s_mat

    def __compute_superstrate(self, structure, S_global):
        kx, ky = self.k_inc[0], self.k_inc[1]
        # take superstrate into account
        _, v_ref_mat = self.__calc_layer_params(structure.UR1, structure.ER1, kx, ky)
        vg_mat = self.__calc_gap_layer_params(kx, ky)
        a_ref_mat = UNIT_MAT_2D + matmul(np.linalg.inv(vg_mat), v_ref_mat)
        b_ref_mat = UNIT_MAT_2D - matmul(np.linalg.inv(vg_mat), v_ref_mat)
        s_ref_11_mat = -matmul(np.linalg.inv(a_ref_mat), b_ref_mat)
        s_ref_12_mat = 2*np.linalg.inv(a_ref_mat)
        s_ref_21_mat = 0.5*(a_ref_mat
                            - matmul(b_ref_mat, np.linalg.inv(a_ref_mat), b_ref_mat))
        s_ref_22_mat = matmul(b_ref_mat, np.linalg.inv(a_ref_mat))
        s_ref_mat = np.concatenate((
            np.concatenate((s_ref_11_mat, s_ref_12_mat), axis=1),
            np.concatenate((s_ref_21_mat, s_ref_22_mat), axis=1)))
        S_global = redheffer_star_prod(s_ref_mat, S_global, UNIT_MAT_2D)
        return S_global

    def __compute_substrate(self, structure, S_global):
        kx, ky = self.k_inc[0], self.k_inc[1]
        # take substrate into account
        _, v_trn_mat = self.__calc_layer_params(structure.UR2, structure.ER2, kx, ky)
        vg_mat = self.__calc_gap_layer_params(kx, ky)
        a_trn_mat = UNIT_MAT_2D + matmul(np.linalg.inv(vg_mat), v_trn_mat)
        b_trn_mat = UNIT_MAT_2D - matmul(np.linalg.inv(vg_mat), v_trn_mat)
        s_trn_11_mat = matmul(b_trn_mat, np.linalg.inv(a_trn_mat))
        s_trn_12_mat = 0.5*(a_trn_mat - matmul(b_trn_mat,
                                               np.linalg.inv(a_trn_mat), b_trn_mat))
        s_trn_21_mat = 2*np.linalg.inv(a_trn_mat)
        s_trn_22_mat = -matmul(np.linalg.inv(a_trn_mat), b_trn_mat)
        s_trn_mat = np.concatenate((np.concatenate((s_trn_11_mat, s_trn_12_mat), axis=1),
                                    np.concatenate((s_trn_21_mat, s_trn_22_mat), axis=1)))
        S_global = redheffer_star_prod(S_global, s_trn_mat, UNIT_MAT_2D)
        return S_global

    def __get_R_T(self, structure, source, S_global):
        z_unit_vec = np.array(([0, 0, 1]))
        alpha_te = np.cross(z_unit_vec, self.k_inc)
        alpha_te = alpha_te/np.linalg.norm(alpha_te)
        alpha_tm = np.cross(alpha_te, self.k_inc)
        alpha_tm = alpha_tm/np.linalg.norm(alpha_tm)
        E_inc = source.norm_P_TM*alpha_tm + source.norm_P_TE*alpha_te
        c_inc = np.array(([E_inc[0], E_inc[1], 0, 0]))

        c_ret = matmul(S_global, c_inc)
        e_field_ref = np.array(([c_ret[0], c_ret[1], 0]))
        e_field_ref[2] = -(self.k_inc[0]*e_field_ref[0]+self.k_inc[1]*e_field_ref[1])/self.k_inc[2]
        k_trn = np.array(([self.k_inc[0], self.k_inc[1], 0]))
        nr2 = np.sqrt(structure.ER2*structure.UR2)
        k_trn[2] = np.sqrt(source.K0*source.K0*nr2*nr2 - k_trn[0]*k_trn[0] - k_trn[1]*k_trn[1])
        e_field_trn = np.array(([c_ret[2], c_ret[3], 0]))
        e_field_trn[2] = -(k_trn[0]*e_field_trn[0]+k_trn[1]*e_field_trn[1])\
            /(k_trn[2])
        R = round((np.linalg.norm(e_field_ref))**2, 4)
        T = round((np.linalg.norm(e_field_trn))**2*((k_trn[2]/structure.UR2).real)\
                /((self.k_inc[2]/structure.UR1).real), 4)
        return R, T

def tmm_(input_toml):
    source = Source(input_toml)
    structure = HomogeneousStructure(input_toml, source.norm_lambda)
    tmm = TMM()
    R, T = tmm.compute(structure, source)
    save_outputs(R, T)

if __name__ == '__main__':
    input_toml = get_input()
    tmm_(input_toml)
