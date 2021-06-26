from pathlib import Path

import numpy as np

class HomogeneousStructure:
    '''Structure for TMM; has only uniform layers;
    those can have non-trivial epsilon and mu values'''
    def __init__(self, input_toml, norm_lambda):
        self._set_epsilon(input_toml)
        self._set_mu(input_toml)
        self._set_layers(input_toml, norm_lambda)

    def _set_epsilon(self, input_toml):
        # permittivity in the reflection region
        self.ER1 = input_toml['superstrate']['epsilon']
        # permittivity in the transmission region
        self.ER2 = input_toml['substrate']['epsilon']

    def _set_mu(self, input_toml):
        # permeability in the reflection region
        self.UR1 = input_toml['superstrate']['mu']
        # permeability in the transmission region
        self.UR2 = input_toml['substrate']['mu']

    def _set_layers(self, input_toml, norm_lambda):
        # define layers
        self.num_layers = len(input_toml['layer'])
        self.ur_vec = [None]*self.num_layers
        self.er_vec = [None]*self.num_layers
        self.layer_thicknesses_vec = [None]*self.num_layers
        for i in range(0, self.num_layers):
            self.ur_vec[i] = input_toml['layer'][i]['mu']
            self.er_vec[i] = input_toml['layer'][i]['epsilon']
            self.layer_thicknesses_vec[i] = input_toml['layer'][i]['thickness']*norm_lambda

class PeriodicStructure(HomogeneousStructure):
    '''Structure for RCWA; can have uniform and periodic layers;
    requires mu = 1 for all layers'''
    def __init__(self, input_toml, norm_lambda):
        self._set_epsilon(input_toml)
        self._set_mu()
        self._set_layers(input_toml, norm_lambda)

    def _set_mu(self):
        # permeability in the reflection region
        self.UR1 = 1.0
        # permeability in the transmission region
        self.UR2 = 1.0

    def _set_layers(self, input_toml, norm_lambda):
        # period in x
        self.Lx = input_toml['periodicity']['period_x']*norm_lambda
        # period in y
        self.Ly = input_toml['periodicity']['period_y']*norm_lambda
        # BUILD DEVICE LAYERS ON HIGH RESOLUTION GRID
        #number of point along x in real-space grid
        self.Nx = 512
        #number of point along y in real-space grid
        self.Ny = int(np.ceil((self.Nx*self.Ly/self.Lx)))

        self.num_layers = len(input_toml['layer'])
        self.L = [None]*self.num_layers
        self.er_vec = [None]*self.num_layers
        self.ur_vec = [None]*self.num_layers

        for i in range(0, self.num_layers):
            self.L[i] = input_toml['layer'][i]['thickness']*norm_lambda
            self.ur_vec[i] = 1.0*np.ones((self.Nx, self.Ny))
            epsilon = input_toml['layer'][i]['epsilon']
            # TODO complex epsilon
            if type(epsilon) == float or type(epsilon) == int:
                self.er_vec[i] = epsilon*np.ones((self.Nx, self.Ny))
            else:
                path = Path(epsilon)
                if path.exists():
                    self.er_vec[i] = np.loadtxt(path, delimiter=',')
                else:
                    raise ValueError('Invalid epsilon for layer {} - should be a float or a path to .csv file instead got {}'.format(i, epsilon))

    def set_convmat(self, P_range, Q_range):
        self.erc_vec = [None]*self.num_layers
        self.urc_vec = [None]*self.num_layers
        for i in range(0, self.num_layers):
            self.erc_vec[i] = self.convmat(self.er_vec[i], P_range, Q_range)
            self.urc_vec[i] = self.convmat(self.ur_vec[i], P_range, Q_range)

    @staticmethod
    def convmat(A, P, Q=1):
        '''Calculate centered FFT of a complex matrix A truncated to P (Q) harmonics in x (y)'''
        Nx = A.shape[0]
        Ny = A.shape[1]
        assert(P <= Nx and Q <= Ny), 'Cannot have more Fourier pts than real-space pts'

        # comp. indices of spatial harmonics
        Nharmonics = P*Q
        p = range(-int(np.floor(P/2)), int(np.floor(P/2))+1)
        q = range(-int(np.floor(Q/2)), int(np.floor(Q/2))+1)

        # do fft
        A = np.fft.fft2(A)/(Nx*Ny)
        A = np.fft.fftshift(A)

        # locate zeroth harmonics
        p0 = int(np.floor(Nx/2))
        q0 = int(np.floor(Ny/2))

        # calc the convolutoiin matrix
        ret = np.zeros((Nharmonics, Nharmonics), dtype=complex)
        for qrow in range(0, Q):
            for prow in range(0, P):
                row = (qrow)*P + prow
                for qcol in range(0, Q):
                    for pcol in range(0, P):
                        col = (qcol)*P + pcol
                        pfft = p[prow] - p[pcol]
                        qfft = q[qrow] - q[qcol]
                        ret[row, col] = A[p0+pfft, q0+qfft]
        return ret
