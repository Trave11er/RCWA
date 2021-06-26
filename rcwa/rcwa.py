# coding: utf-8
import numpy as np

from rcwa.common import matmul, redheffer_star_prod, get_input
from rcwa.source import Source
from rcwa.structure import PeriodicStructure

class Harmonics():
    '''Keeps track of Fourier expansion paramters'''
    def __init__(self, input_toml):
        # number of spatial harmonics along x and y
        self.P_high = input_toml['periodicity']['harmonics_x']
        self.Q_high = input_toml['periodicity']['harmonics_y']
        self.P_low = -self.P_high
        self.Q_low = -self.Q_high
        self.P_range = 2*self.P_high + 1
        self.Q_range = 2*self.Q_high + 1
        self.P_vec = np.arange(self.P_low, self.P_low + self.P_range)
        self.Q_vec = np.arange(self.Q_low, self.Q_low + self.Q_range)
        self.Nharm = self.P_range*self.Q_range

def save_outputs(P_vec, Q_vec, R, T, P_range, Nharm):
    np.set_printoptions(precision=4)
    with open('output.toml', 'w') as fid:
        fid.write('[R]\n')
        for i in range(0, Nharm):
            fid.write('{:.0f}{:.0f} = '.format(P_vec[i%P_range], Q_vec[i//P_range]) + '{:.4f}\n'.format(R[i]))
        fid.write('sum = {:.4f}\n'.format(np.sum(R)))
        fid.write('\n[T]\n')
        for i in range(0, Nharm):
            fid.write('{:.0f}{:.0f} = '.format(P_vec[i%P_range], Q_vec[i//P_range]) + '{:.4f}\n'.format(T[i]))
        fid.write('sum = {:.4f}\n'.format(np.sum(T)))
        fid.write('\n[R_T]\n')
        fid.write('sum = {:.4f}'.format(np.sum(R+T)))

class RCWA():
    '''Calculates diffraction efficiencies of a period structure'''
    def compute(self, structure, source, harmonics):
        S_global = self.__prepare(structure, source, harmonics)
        S_global = self.__compute_layers(structure, source, harmonics, S_global)
        S_global = self.__compute_superstrate(structure, harmonics, S_global)
        S_global = self.__compute_substrate(structure, harmonics, S_global)
        R, T = self.__get_R_T(structure, source, harmonics, S_global)
        return R, T

    def __prepare(self, structure, source, harmonics):
        # initialise parameters for the computation
        Nharm = harmonics.Nharm
        nr1 = np.sqrt(structure.UR2*structure.ER2)
        self.k_inc = source.K0*nr1*np.array(([np.sin(source.THETA)*np.cos(source.PHI), np.sin(source.THETA)*np.sin(source.PHI), np.cos(source.THETA)]))
        self.k_inc /= source.K0
        self.k_ref = np.array(([self.k_inc[0], self.k_inc[1], -self.k_inc[2]]))
        nr2 = np.sqrt(structure.UR1*structure.ER1)
        self.k_trn = np.array(([self.k_ref[0], self.k_ref[1], 0]))
        self.k_trn[2] = np.sqrt(source.K0*source.K0*nr2*nr2 - self.k_trn[0]*self.k_trn[0] - self.k_trn[1]*self.k_trn[1])

        self.kx_mat = np.zeros((Nharm, Nharm), dtype=complex)
        self.ky_mat = np.zeros((Nharm, Nharm), dtype=complex)
        kz_0_mat = np.zeros((Nharm, Nharm), dtype=complex)
        self.kz_ref_mat = np.zeros((Nharm, Nharm), dtype=complex)
        self.kz_trn_mat = np.zeros((Nharm, Nharm), dtype=complex)
        for i, i_harm in enumerate(harmonics.P_vec):
            for j, j_harm in enumerate(harmonics.Q_vec):
                pos = i*harmonics.P_range + j
                self.kx_mat[pos, pos] = self.k_inc[0] - j_harm*2*np.pi/(structure.Lx*source.K0)
                self.ky_mat[pos, pos] = self.k_inc[1] - i_harm*2*np.pi/(structure.Ly*source.K0)
                kz_0_mat[pos, pos] = np.conj(np.lib.scimath.sqrt(1 - (self.kx_mat[pos, pos])**2 - (self.ky_mat[pos, pos])**2))
                self.kz_ref_mat[pos, pos] = -np.conj(np.lib.scimath.sqrt(np.conj(structure.ER2)*np.conj(structure.UR2)-\
                                            self.kx_mat[pos, pos]**2-self.ky_mat[pos, pos]**2))
                self.kz_trn_mat[pos, pos] = np.conj(np.lib.scimath.sqrt(np.conj(structure.ER1)*np.conj(structure.UR1)-\
                                            self.kx_mat[pos, pos]**2-self.ky_mat[pos, pos]**2))

        self.I_mat = np.diag(np.ones(Nharm))
        self.I_big_mat = np.diag(np.ones(2*Nharm))
        self.zeros_mat = np.zeros((Nharm, Nharm))
        zeros_big_mat = np.zeros((2*Nharm, 2*Nharm))
        self.W0_mat = np.diag(np.ones(2*Nharm))
        P0_mat = np.zeros((2*Nharm, 2*Nharm), dtype=complex)
        P0_mat[0:Nharm, 0:Nharm] = matmul(self.kx_mat, self.ky_mat)
        P0_mat[0:Nharm, Nharm:2*Nharm] = self.I_mat - matmul(self.kx_mat, self.kx_mat)
        P0_mat[Nharm:2*Nharm, 0:Nharm] = matmul(self.ky_mat, self.ky_mat) - self.I_mat
        P0_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -matmul(self.ky_mat, self.kx_mat)
        Q0_mat = P0_mat
        lambda0_mat = np.concatenate((np.concatenate((1j*kz_0_mat, self.zeros_mat), axis=1),
                                      np.concatenate((self.zeros_mat, 1j*kz_0_mat), axis=1)))
        self.V0_mat = matmul(Q0_mat, np.linalg.inv(lambda0_mat))

        S_global = np.concatenate((np.concatenate((zeros_big_mat, self.I_big_mat), axis=1),
                                   np.concatenate((self.I_big_mat, zeros_big_mat), axis=1)))
        return S_global

    def __compute_layers(self, structure, source, harmonics, S_global):
        Nharm = harmonics.Nharm
        kx_mat = self.kx_mat
        ky_mat = self.ky_mat
        # iterating through layers
        for i in range(0, structure.num_layers):
            ERC = structure.erc_vec[i]
            URC = structure.urc_vec[i]
            thickness = structure.L[i]

            P_mat = np.zeros((2*Nharm, 2*Nharm), dtype=complex)
            P_mat[0:Nharm, 0:Nharm] = matmul(kx_mat, np.linalg.inv(ERC), ky_mat)
            P_mat[0:Nharm, Nharm:2*Nharm] = URC - matmul(kx_mat, np.linalg.inv(ERC), kx_mat)
            P_mat[Nharm:2*Nharm, 0:Nharm] = matmul(ky_mat, np.linalg.inv(ERC), ky_mat) - URC
            P_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -matmul(ky_mat, np.linalg.inv(ERC), kx_mat)

            Q_mat = np.zeros((2*Nharm, 2*Nharm), dtype=complex)
            Q_mat[0:Nharm, 0:Nharm] = matmul(kx_mat, np.linalg.inv(URC), ky_mat)
            Q_mat[0:Nharm, Nharm:2*Nharm] = ERC - matmul(kx_mat, np.linalg.inv(URC), kx_mat)
            Q_mat[Nharm:2*Nharm, 0:Nharm] = matmul(ky_mat, np.linalg.inv(URC), ky_mat) - ERC
            Q_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -matmul(ky_mat, np.linalg.inv(URC), kx_mat)

            Omega_sq_mat = matmul(P_mat, Q_mat)
            v_vec, W_mat = np.linalg.eig(Omega_sq_mat)
            lambda_mat = np.diag(np.conj(np.sqrt(v_vec)))
            V_mat = matmul(Q_mat, W_mat, np.linalg.inv(lambda_mat))
            A_mat = matmul(np.linalg.inv(W_mat), self.W0_mat) +\
                    matmul(np.linalg.inv(V_mat), self.V0_mat)
            B_mat = matmul(np.linalg.inv(W_mat), self.W0_mat) -\
                    matmul(np.linalg.inv(V_mat), self.V0_mat)
            X_mat = np.diag(np.exp(-np.diag(lambda_mat)*source.K0*thickness))
            S_11 = matmul(np.linalg.inv(A_mat -\
                   matmul(X_mat, B_mat, np.linalg.inv(A_mat), X_mat, B_mat)),
                   (matmul(X_mat, B_mat, np.linalg.inv(A_mat), X_mat, A_mat) - B_mat))
            S_12 = matmul(np.linalg.inv(A_mat -\
                   matmul(X_mat, B_mat, np.linalg.inv(A_mat), X_mat, B_mat)),
                   X_mat, (A_mat - matmul(B_mat, np.linalg.inv(A_mat), B_mat)))
            S = np.concatenate((np.concatenate((S_11, S_12), axis=1),
                                np.concatenate((S_12, S_11), axis=1)))
            S_global = redheffer_star_prod(S_global, S, self.I_big_mat)
        return S_global

    def __compute_superstrate(self, structure, harmonics, S_global):
        Nharm = harmonics.Nharm
        kx_mat = self.kx_mat
        ky_mat = self.ky_mat
        # reflection region
        Q_ref_mat = np.zeros((2*Nharm, 2*Nharm), dtype = complex)
        Q_ref_mat[0:Nharm, 0:Nharm] = matmul(kx_mat, ky_mat)
        Q_ref_mat[0:Nharm, Nharm:2*Nharm] = structure.UR2*structure.ER2*self.I_mat -\
                                            matmul(kx_mat, kx_mat)
        Q_ref_mat[Nharm:2*Nharm, 0:Nharm] = matmul(ky_mat, ky_mat) -\
                                            structure.UR2*structure.ER2*self.I_mat
        Q_ref_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -matmul(ky_mat, kx_mat)
        Q_ref_mat /= structure.UR2

        self.W_ref_mat = np.diag(np.ones(2*Nharm))

        lambda_ref_mat = np.concatenate((np.concatenate((-1j*self.kz_ref_mat, self.zeros_mat), axis=1),
                                np.concatenate((self.zeros_mat, -1j*self.kz_ref_mat), axis=1)))
        V_ref_mat = matmul(Q_ref_mat, np.linalg.inv(lambda_ref_mat))
        A_ref_mat = matmul(np.linalg.inv(self.W0_mat), self.W_ref_mat) +\
                    matmul(np.linalg.inv(self.V0_mat), V_ref_mat)
        B_ref_mat = matmul(np.linalg.inv(self.W0_mat), self.W_ref_mat) -\
                    matmul(np.linalg.inv(self.V0_mat), V_ref_mat)
        S_ref_11 = -matmul(np.linalg.inv(A_ref_mat), B_ref_mat)
        S_ref_12 = 2*np.linalg.inv(A_ref_mat)
        S_ref_21 = 0.5*(A_ref_mat - matmul(B_ref_mat, np.linalg.inv(A_ref_mat), B_ref_mat))
        S_ref_22 = matmul(B_ref_mat, np.linalg.inv(A_ref_mat))
        S_ref = np.concatenate((np.concatenate((S_ref_11, S_ref_12), axis=1),
                                np.concatenate((S_ref_21, S_ref_22), axis=1)))

        S_global = redheffer_star_prod(S_ref, S_global, self.I_big_mat)
        return S_global

    def __compute_substrate(self, structure, harmonics, S_global):
        Nharm = harmonics.Nharm
        kx_mat = self.kx_mat
        ky_mat = self.ky_mat
        # transmission region
        Q_trn_mat = np.zeros((2*Nharm, 2*Nharm), dtype=complex)
        Q_trn_mat[0:Nharm, 0:Nharm] = matmul(kx_mat, ky_mat)
        Q_trn_mat[0:Nharm, Nharm:2*Nharm] = structure.UR1*structure.ER1*self.I_mat -\
                                            matmul(kx_mat, kx_mat)
        Q_trn_mat[Nharm:2*Nharm, 0:Nharm] = matmul(ky_mat, ky_mat) -\
                                            structure.UR1*structure.ER1*self.I_mat
        Q_trn_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -matmul(ky_mat, kx_mat)
        Q_trn_mat /= structure.UR1

        self.W_trn_mat = np.diag(np.ones(2*Nharm))
        lambda_trn_mat = np.concatenate((np.concatenate((1j*self.kz_trn_mat, self.zeros_mat), axis=1),
                                         np.concatenate((self.zeros_mat, 1j*self.kz_trn_mat), axis=1)))
        V_trn_mat = matmul(Q_trn_mat, np.linalg.inv(lambda_trn_mat))
        A_trn_mat = matmul(np.linalg.inv(self.W0_mat), self.W_trn_mat) +\
                    matmul(np.linalg.inv(self.V0_mat), V_trn_mat)
        B_trn_mat = matmul(np.linalg.inv(self.W0_mat), self.W_trn_mat) -\
                    matmul(np.linalg.inv(self.V0_mat), V_trn_mat)

        S_trn_11 = matmul(B_trn_mat, np.linalg.inv(A_trn_mat))
        S_trn_12 = 0.5*(A_trn_mat - matmul(B_trn_mat, np.linalg.inv(A_trn_mat), B_trn_mat))
        S_trn_21 = 2*np.linalg.inv(A_trn_mat)
        S_trn_22 = -matmul(np.linalg.inv(A_trn_mat), B_trn_mat)
        S_trn = np.concatenate((np.concatenate((S_trn_11, S_trn_12), axis=1),
                                np.concatenate((S_trn_21, S_trn_22), axis=1)))
        S_global = redheffer_star_prod(S_global, S_trn, self.I_big_mat)
        return S_global

    def __get_R_T(self, structure, source, harmonics, S_global):
        Nharm = harmonics.Nharm
        # computing input and output fields
        delta_vec = np.zeros(Nharm)
        delta_vec[int(np.floor(Nharm/2))] = 1
        z_unit_vec = np.array(([0, 0, 1]))
        alpha_TE = np.cross(z_unit_vec, self.k_inc)
        alpha_TE = alpha_TE/np.linalg.norm(alpha_TE)
        alpha_TM = np.cross(alpha_TE, self.k_inc)
        alpha_TM = alpha_TM/np.linalg.norm(alpha_TM)
        E_inc = np.zeros(2*Nharm, dtype=complex)
        E_inc[0:Nharm] = (source.P_TM*alpha_TM + source.P_TE*alpha_TE)[0]*delta_vec
        E_inc[Nharm:2*Nharm] = (source.P_TM*alpha_TM + source.P_TE*alpha_TE)[1]*delta_vec
        c_inc = matmul(np.linalg.inv(self.W_ref_mat), E_inc)
        E_ref_xy = matmul(self.W_ref_mat, S_global[0:2*Nharm, 0:2*Nharm], c_inc)
        E_trn_xy = matmul(self.W_trn_mat, S_global[2*Nharm:4*Nharm, 0:2*Nharm], c_inc)
        E_ref_z = -matmul(np.linalg.inv(self.kz_ref_mat),\
                  matmul(self.kx_mat, E_ref_xy[0:Nharm]) +\
                  matmul(self.ky_mat, E_ref_xy[Nharm:2*Nharm]))
        E_trn_z = -matmul(np.linalg.inv(self.kz_trn_mat),\
                  matmul(self.kx_mat, E_trn_xy[0:Nharm]) +\
                  matmul(self.ky_mat, E_trn_xy[Nharm:2*Nharm]))

        r_sq = np.zeros((Nharm), dtype=complex)
        t_sq = np.zeros((Nharm), dtype=complex)
        for i in range(0, Nharm):
            r_sq[i] = E_ref_xy[i]*np.conj(E_ref_xy[i]) +\
                      E_ref_xy[Nharm+i]*np.conj(E_ref_xy[Nharm+i]) +\
                      E_ref_z[i]*np.conj(E_ref_z[i])
            t_sq[i] = E_trn_xy[i]*np.conj(E_trn_xy[i]) +\
                      E_trn_xy[Nharm+i]*np.conj(E_trn_xy[Nharm+i]) +\
                      E_trn_z[i]*np.conj(E_trn_z[i])

        R = -np.real(matmul(self.kz_ref_mat, r_sq)/structure.UR2)
        R /= np.real(self.k_inc[2]/structure.UR2)#

        T = np.real(matmul(self.kz_trn_mat, t_sq)/structure.UR1)
        T /= np.real(self.k_inc[2]/structure.UR1)
        return R, T

def rcwa_(input_toml):
    source = Source(input_toml)
    structure = PeriodicStructure(input_toml, source.norm_lambda)
    harmonics = Harmonics(input_toml)
    structure.set_convmat(harmonics.P_range, harmonics.Q_range)
    rcwa = RCWA()
    R, T = rcwa.compute(structure, source, harmonics)
    save_outputs(harmonics.P_vec, harmonics.Q_vec, R, T,\
                 harmonics.P_range, harmonics.Nharm)

if __name__ == '__main__':
    input_toml = get_input()
    rcwa_(input_toml)
