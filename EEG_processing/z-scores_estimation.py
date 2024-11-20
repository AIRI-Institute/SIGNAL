#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import mne
import numpy as np
from scipy.signal import butter
from scipy.signal import lfilter
import tqdm

NCOND = 4


def get_zscores(epochs_reconst, cond1, cond2):
    epochs1 = epochs_reconst[cond1].get_data()
    epochs2 = epochs_reconst[cond2].get_data()

    epochs1_mean = epochs1.mean(axis=0)
    epochs1_std = epochs1.std(axis=0)

    epochs2_mean = epochs2.mean(axis=0)
    epochs2_std = epochs2.std(axis=0)

    N_1 = epochs1.shape[0]
    N_2 = epochs2.shape[0]

    z_matrix = np.zeros(epochs1_mean.shape)
    
    n_channels = epochs1_mean.shape[0]
    n_times = epochs1_mean.shape[1]

    for channel in range(n_channels):
        for timestamp in range(n_times):
            part1 = epochs1_mean[channel, timestamp
                    ] - epochs2_mean[channel, timestamp]
            part2 = np.sqrt(1 / N_1 * (np.square(
                epochs1_std[channel, timestamp])) +
                1 / N_2 * (np.square(epochs2_std[channel, timestamp]))) / 2
            z_matrix[channel, timestamp] = part1 / part2
    return z_matrix


def get_zmatrix(data, event_names):
    [b,a] = butter(3,0.01)
    x = []

    for cond in range(NCOND):
        cur_cond_data = data[event_names[cond]].get_data()
        x_cur = np.zeros((cur_cond_data.shape[0], 
                          cur_cond_data.shape[1], 
                          301))
        for i in range(cur_cond_data.shape[0]):
            tmp = lfilter(b, a, np.squeeze(cur_cond_data[i]))
            x_cur[i] = tmp[:, 0:tmp.shape[1]:4]
        x.append(x_cur)

    Nc = len(x)
    mx = []
    sx = []
    Nt = []

    for cond in range(NCOND):
        mx.append(np.squeeze(np.mean(x[cond], axis=0)))
        sx.append(np.squeeze(np.std(x[cond], axis=0)))
        Nt.append(cur_cond_data.shape[0])

    z = np.zeros((Nc, Nc, x[0].shape[1], x[0].shape[2]))

    for cond1 in list(range(Nc)):
        for cond2 in list(range(cond1+1, Nc)):
            tmp = (mx[cond1] - mx[cond2]) / np.sqrt((1 / Nt[
                cond1] * np.square(sx[cond1]) +  1 / Nt[cond2] * np.square(
                sx[cond2])) / 2)
            z[cond1, cond2] = tmp
            z[cond2, cond1] = -tmp
            
    return z, x, Nt, Nc


def permutation_test(x, z, Nt, Nc):
    random_seed = 0
    p = np.zeros((Nc, Nc, z.shape[2], z.shape[3]))

    for cond1 in tqdm.tqdm(range(Nc)):
        for cond2 in range(cond1+1, Nc):
            data12 = np.concatenate((x[cond1], x[cond2]))
            N = Nt[cond1] + Nt[cond2]
            N1 = Nt[cond1]
            N2 = Nt[cond2]
            zs = np.zeros((100, x[cond1].shape[1], x[cond1].shape[2]))
            for bs in range(100):
                try:
                    np.random.seed(random_seed)                
                    i1 = np.fix(np.random.rand(1, N1) * (N-1))[0].astype(int)
                    random_seed += 1
                    np.random.seed(random_seed)               
                    i2 = np.fix(np.random.rand(1, N2) * (N-1))[0].astype(int)
                    i1 = np.where(i1 >= data12.shape[0],
                                  np.random.choice(np.arange(
                                      data12.shape[0]), 1), i1)
                    i2 = np.where(i2 >= data12.shape[0],
                                  np.random.choice(
                                      np.arange(data12.shape[0]), 1), i2)
                    random_seed += 1
                    mx1s = np.squeeze(np.mean(data12[i1, :, :], axis=0))
                    mx2s = np.squeeze(np.mean(data12[i2, :, :], axis=0))
                    sx1s = np.squeeze(np.std(data12[i1, :, :], axis=0))
                    sx2s = np.squeeze(np.std(data12[i2, :, :], axis=0))
                    zs[bs, :, :] = (mx2s - mx1s) / np.sqrt(
                    (1 / N1 * np.square(sx1s) + 1 / N2 * np.square(sx2s)) / 2 )
                except Exception as e:
                    print(e)

            for i in range(zs.shape[1]):
                for j in range(zs.shape[2]):
                    try:
                        p[cond1, cond2, i, j] = np.sum(zs[:, i, j] > z[
                            cond1, cond2, i, j]) / zs[:, i, j].shape[0]
                        p[cond2, cond1, i, j] = np.sum(zs[:, i, j] < z[
                            cond1, cond2, i, j]) / zs[:, i, j].shape[0]
                    except:
                        print('here')
    return p


if __name__ == '__main__':
    for p_ID in range(21):
        perm_dir = 'permutation_results'
        if not os.path.exists(perm_dir):
            os.mkdir(perm_dir)
        filename = os.path.join('epoched_data',
                                f'p{p_ID}_epochs.set')
        epochs = mne.read_epochs_eeglab(filename, verbose=True)
        # Here we group events into 3 blocks based on congruency type
        event_names = [['Stimulus/S  1_1', 'Stimulus/S  1_2', 'Stimulus/S  1_3'],  # normal
                       ['Stimulus/S  2_1', 'Stimulus/S  2_2', 'Stimulus/S  2_3'],  # semantic error
                       ['Stimulus/S  3_1', 'Stimulus/S  3_2', 'Stimulus/S  3_3'],  # grammatical error
                       ['Stimulus/S  4_1', 'Stimulus/S  4_2', 'Stimulus/S  4_3']]  # semantic-grammatical error
        z, x, Nt, Nc = get_zmatrix(epochs, event_names)
        print('Z matrix saved')
        p = permutation_test(x, z, Nt, Nc)
        print('Permutations done')
        results = {'z': z, 'x': x, 'Nt': Nt, 'p': p}
        with open(os.path.join(perm_dir,
                               f'p{p_ID}_results_permutations.npy'), 'wb') as f:
            np.save(f, results)
        print('Permutations saved')

        print(epochs['Stimulus/S  1_1'].get_data()[0])
        epochs['Stimulus/S  1_1'].average().plot_joint()
