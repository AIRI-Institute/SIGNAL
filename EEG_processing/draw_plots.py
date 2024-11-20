#!/usr/bin/env python
# coding: utf-8

import numpy as np
import mne
import tqdm
import os
from scipy.stats import combine_pvalues
from matplotlib import pyplot as plt
import matplotlib.colors as colors

NCOND = 4
WORKDIR =  os.getcwd()

def plot_permutation_test(p=None, Nc=12, show = False, folder = '.', 
                          alpha=0.05, postfix=None, cond_dict = None,
                          data_eeg=None):
    fig, axes= plt.subplots(Nc, Nc, figsize=(30, 30))
    for c1 in range(Nc):
        for c2 in range(Nc):
            if c1 == c2:
                continue
            pvals = axes[c1, c2].imshow(p[c1, c2] < alpha, 
                                          aspect='auto')
            fig.colorbar(pvals, ax=axes[c1, c2])
            axes[c1, c2].set_title( f'{cond_dict[c1]} VS {cond_dict[c2]}',
                                    fontsize=35)
            axes[c1, c2].set_xlabel('Time, ms')
            axes[c1, c2].set_ylabel('Channels')
            axes[c1, c2].set_xticks(np.arange(0, 301, 25),
                                    labels=np.arange(-400, 802, 100))
            axes[c1, c2].set_yticks(np.arange(0, 63),
                                    labels=data_eeg.ch_names)
    fig.tight_layout()
    if show:
        fig.show()
    if postfix is None:
        fig.savefig(os.path.join(folder, f'mean_pvals.pdf'))
    else:
        fig.savefig(os.path.join(folder, f'mean_pvals_{postfix}.pdf'))


def plot_zmatrix(z=None, Nc=12, show=False, folder = None,
                 postfix=None, cond_dict=None, data_eeg=None):
    fig, axes= plt.subplots(Nc, Nc, figsize=(30, 30))
    for c1 in range(Nc):
        for c2 in range(Nc):
            if c1 == c2:
                continue
            bounds = np.arange(-1, 1, 0.2)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            zscores = axes[c1, c2].imshow(z[c1, c2],
                                          extent=[0, z[c1, c2].shape[1],
                                                  z[c1, c2].shape[0], 0],
                                          aspect='auto', norm=norm)
            fig.colorbar(zscores, ax=axes[c1, c2])
            axes[c1, c2].set_title( f'{cond_dict[c1]} VS {cond_dict[c2]}',
                                    fontsize=35)
            axes[c1, c2].set_ylabel('Channels')
            axes[c1, c2].set_xticks(np.arange(0, 301, 25),
                                    labels=np.arange(-400, 802, 100))
            axes[c1, c2].set_yticks(np.arange(0, 63), labels=data_eeg.ch_names)
            axes[c1, c2].set_xlabel('Time, ms')
    fig.tight_layout()
    if show:
        fig.show()
    if folder is not None:
        if postfix is None:
            fig.savefig(os.path.join(folder, f'mean_z-scores.pdf'))
        else: fig.savefig(os.path.join(folder, f'mean_z-scores_{postfix}.pdf'))

def topo_plot_zmatrix(matrix, data_eeg, cond_dict, show=False, folder = None):
    fig = plt.figure(layout='constrained', figsize=(30, 30))
    subfigs = fig.subfigures(6, 1, wspace=0.02,)
    condition_pairs = []
    counter = 0
    for c1 in range(NCOND):
        for c2 in range(c1, NCOND):
            if c1 == c2:
                continue
            axs = subfigs[counter].subplots(1, 10, gridspec_kw=dict(
                width_ratios=[2, 2, 2, 2, 2, 2, 2, 2, 2, 0.5]))
            title = f'{cond_dict[c1]} VS {cond_dict[c2]}'
            condition_pairs.append(title)
            f_evoked = mne.EvokedArray(np.repeat(matrix[c1, c2][:,100:],
                                                 4, axis=1),
                                       data_eeg.info, tmin=-0.)
            f_evoked.plot_topomap(times=np.arange(-0., 0.801, 0.1),
                                  axes = axs, show=False, contours=0)
            subfigs[counter].suptitle(title, fontsize=45,
                                      fontname='Times New Roman')
            show_times =  [f"{int(time)} ms" for time in range(0,802,100)]
            for ax in axs[1::2]:
                ax.set_title('')
            for ax, title_ in list(zip(axs,show_times))[::2]:
                if '-0.0' in ax.get_title():
                    ax.set_title('0 ms', fontsize=30,
                                 fontname='Times New Roman')
                else:
                    ax.set_title(title_, fontsize=30,
                                 fontname='Times New Roman')
            counter += 1
    if show:
        plt.show()
    fig.savefig(os.path.join(folder, 'topoplot.pdf'))


if __name__ == '__main__':

    example_epochs = mne.io.read_epochs_eeglab(
        os.path.join(WORKDIR, 'epoched_data/p0_epochs.set'))

    # data_all = np.load(os.path.join(WORKDIR, 'average/RDM_neuro.npy'), allow_pickle=True).item()
    # np.save('RDM_neuro.npy', {"p_values_mean": p_values_mean, 'dataset_z':dataset_z})

    dataset_p = []
    dataset_z = []
    for p_ID in range(21):
        permutation_result = np.load(
            os.path.join(WORKDIR, 'permutation_data',
                         f'p{p_ID}_results_permutations.npy'),
            allow_pickle=True).item()
        dataset_p.append(permutation_result['p'])
        dataset_z.append(permutation_result['z'])


    dataset_p = np.array(dataset_p)
    dataset_z = np.array(dataset_z)

    NChannels = dataset_p.shape[3]
    NTimes = dataset_p.shape[4]

    p_values_mean = np.zeros((NCOND, NCOND, NChannels, NTimes))

    for cond1 in tqdm.tqdm(range(NCOND)):
        for cond2 in range(NCOND):
            for channel in range(NChannels):
                for timestamp in range(NTimes):
                    p_values_mean[cond1, cond2, channel,
                    timestamp] = combine_pvalues(
                        dataset_p[:, cond1, cond2, channel, timestamp])[1]

    p_values_mean_flat = []

    for cond1 in tqdm.tqdm(range(NCOND)):
        for cond2 in range(cond1 + 1, NCOND):
            p_values_mean_flat.append(p_values_mean[cond1, cond2, :, :])

    dataset_z_mean = dataset_z.mean(axis=0)

    a = np.array(p_values_mean_flat).flatten()
    a.sort()
    b = (np.arange(a.shape[0]) / a.shape[0]) * 0.1

    corrected_p = 0.05
    for i in range(a.shape[0]):
        if a[i] > b[i]:
            corrected_p = a[i]
            break

    data_z_masked = np.where(p_values_mean <= corrected_p, dataset_z_mean, 0)

    fig, axes = plt.subplots(1, 1)

    axes.plot(a)
    axes.plot(b)

    print(corrected_p)

    cond_dict = ['Normal',
                'Sem Incongruent',
                'Gram Incongruent',
                'S&G Incongruent']

    result_dir = os.path.join(WORKDIR, 'result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    plot_permutation_test(p_values_mean, folder=result_dir, 
                        Nc = 4, alpha=corrected_p, postfix='fdr_corrected',
                        cond_dict = cond_dict, data_eeg=example_epochs)

    plot_permutation_test(p_values_mean, folder=result_dir, 
                        Nc = 4, alpha=0.05,
                        cond_dict = cond_dict, data_eeg=example_epochs)

    plot_zmatrix(dataset_z_mean, folder=result_dir, Nc=4,
                 cond_dict = cond_dict, data_eeg=example_epochs)

    plot_zmatrix(data_z_masked, Nc=4, folder=result_dir,
                 postfix='masked',cond_dict = cond_dict,
                 data_eeg=example_epochs)

    topo_plot_zmatrix(data_z_masked, example_epochs, cond_dict,
                      show=False, folder = result_dir)

    example_epochs.plot_sensors(show_names=True)
