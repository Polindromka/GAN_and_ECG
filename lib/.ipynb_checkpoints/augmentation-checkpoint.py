import numpy as np
import numba

@numba.njit
def drop_random_channel(ecg, p):
    for i in range(ecg.shape[0]):
        if np.random.random() < p:
            ecg[i] = 0
    return ecg

@numba.njit
def add_random_noise(ecg, noise_type, p, a):
    if np.random.random() < p:

        if noise_type == 'uniform':
            noise = (np.random.random(ecg.shape) * 2 - 1)

        elif noise_type == 'normal':
            noise = np.random.randn(*ecg.shape)

        else:
            raise ValueError('Unknown NOISE_TYPE')

        ecg += noise * a
    return ecg