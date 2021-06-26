import numpy as np

from rcwa._constants import DEGREES_TO_RAD

class Source:
    def __init__(self, input_toml):
        self.norm_lambda = 2*np.pi/input_toml['source']['wavelength']
        self.K0 = 1
        self.THETA = input_toml['source']['theta'] * DEGREES_TO_RAD
        self.PHI = input_toml['source']['phi'] * DEGREES_TO_RAD
        self.P_TE = input_toml['source']['te_amplitude'][0] + \
                input_toml['source']['te_amplitude'][1]*1j  # amplitude of TE polarization

        self.P_TM = input_toml['source']['tm_amplitude'][0] + \
                input_toml['source']['tm_amplitude'][1]*1j  # amplitude of TM polarization
        # normalise polarisation
        norm_pol = np.sqrt((np.real(self.P_TE))**2 + (np.imag(self.P_TE))**2 + \
        (np.real(self.P_TM))**2 + (np.imag(self.P_TM))**2)
        self.norm_P_TM = self.P_TM/norm_pol
        self.norm_P_TE = self.P_TE/norm_pol
