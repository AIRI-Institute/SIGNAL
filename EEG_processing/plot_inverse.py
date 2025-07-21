#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import os
import pathlib
import mne
import json
import matplotlib.pyplot as plt
import numpy as np
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, make_inverse_operator, apply_inverse_epochs
from mne.datasets import eegbci, fetch_fsaverage

WORKDIR = pathlib.Path(__file__).parent.parent.resolve()


def read_epochs(pID=None, filepath=None, workdir=None):
    if filepath is None:
        filepath = os.path.join(workdir, f'participants/p{pID}/p{pID}_raw.fif')
    raw = mne.io.read_raw_fif(filepath)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)  # needed for inverse modeling
    events=mne.events_from_annotations(raw);
    epochs = mne.Epochs(raw, events[0], 
                        events[1], detrend=0, baseline=(None, 0),
                        tmin=-0.4, tmax=0.8, )
    return epochs
subject = "fsaverage"
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)
src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
trans = os.path.join(WORKDIR, 'p0-trans.fif')
LABELS = mne.read_labels_from_annot(subject, parc='HCPMMP1')
label_names = [(idx, lab.name) for idx, lab in enumerate(LABELS)]

def make_inverse(epochs, trans, src, bem, plot=False):
    noise_cov_adhoc = mne.make_ad_hoc_cov(epochs.info)
    noise_cov_adhoc =mne.cov.prepare_noise_cov(noise_cov_adhoc, epochs.info, rank='full', verbose=False)
    if plot:
        noise_cov_adhoc.plot(epochs.info, verbose=False)
    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None, verbose=False
    );
    inverse_operator = make_inverse_operator(
            epochs.info, fwd, noise_cov_adhoc, loose=0.2, depth=0.8, verbose=False
        );
    return inverse_operator

def make_stc(epochs, inverse, trans, 
             src, bem, save=False,
             filename=None,
             apply_to_epochs=False):
    method = "dSPM"
    snr = 3
    lambda2 = 1 / snr**2
    if apply_to_epochs:
        stc = apply_inverse_epochs(
            epochs,
            inverse,
            lambda2,
            method=method,
            pick_ori=None,
            verbose=False,
            );
    else:
        stc, residual = apply_inverse(
            epochs.average(),
            inverse,
            lambda2,
            method=method,
            return_residual=True,
            verbose=False,
            );
    if save:
        np.save(os.path.join(WORKDIR, f'p{pID}', 'stc', f'stc_{filename}'), stc)
    return stc

epochs_all = []
for p_ID in range(21):
    p_filename = os.path.join(WORKDIR, f'participants/p{p_ID}/data_processed/p{p_ID}_epochs.fif')
    epochs = mne.read_epochs(p_filename)
    epochs.set_eeg_reference(projection=True)
    epochs_all.append(epochs)

conds = list(epochs_all[0].event_id.keys())
Ntimes = epochs_all[0].get_data().shape[2]
Nconds = len(epochs_all[0].event_id)
inverse = make_inverse(epochs_all[0], trans, src, bem, plot=False)

my_events = []
epoch_data = []

for ep in epochs_all:
    dat = ep.get_data()
    my_events.append(ep.events)
    epoch_data.append(dat)
    print(ep.events.shape, dat.shape)
    
my_events = np.concatenate(my_events)
my_events[:, 0] = np.arange(my_events.shape[0])
my_event_id = epochs_all[0].event_id
epoch_data = np.concatenate(epoch_data)
all_epochs = mne.EpochsArray(data = epoch_data,
                               events=my_events, 
                               event_id=my_event_id, 
                               info=epochs_all[0].info, baseline=(None, 0),
                               tmin=-0.4 )    

cond_norm = ['Stimulus/S  1_1', 'Stimulus/S  1_2', 'Stimulus/S  1_3'] 
cond_sem = ['Stimulus/S  2_1', 'Stimulus/S  2_2', 'Stimulus/S  2_3']
cond_gram = ['Stimulus/S  3_1', 'Stimulus/S  3_2', 'Stimulus/S  3_3'] 
cond_sg = ['Stimulus/S  4_1', 'Stimulus/S  4_2', 'Stimulus/S  4_3']
evoked_norm = all_epochs[cond_norm].average().pick("eeg")
evoked_sem = all_epochs[cond_sem].average().pick("eeg")
evoked_gram = all_epochs[cond_gram].average().pick("eeg")
evoked_sg = all_epochs[cond_sg].average().pick("eeg")

get_ipython().run_line_magic('matplotlib', 'inline')
all_epochs[cond_norm].average().plot_joint();
all_epochs[cond_sem].average().plot_joint();
all_epochs[cond_gram].average().plot_joint();
all_epochs[cond_sg].average().plot_joint();

def plot_inverse(cond1, cond2, evoked_1, evoked_2, inverse,
                snr = 3, lambda2_x = 1, contrast=1):
    method = "dSPM"
    snr = snr
    lambda2 = lambda2_x / snr**2
    stc_1, residual_1 = apply_inverse(
            evoked_1,
            inverse,
            lambda2,
            method=method,
            pick_ori=None,
            return_residual=True,
            verbose=False,
        );
    if cond2 is not None:
        stc_2, residual_2 = apply_inverse(
                evoked_2,
                inverse,
                lambda2,
                method=method,
                pick_ori=None,
                return_residual=True,
                verbose=False,
            );

    if contrast == 1:
        stc = stc_1 - stc_2
    elif contrast == -1:
        stc = stc_2 - stc_1
    elif contrast == 0:
        stc = stc_1

    fig, ax = plt.subplots()
    ax.plot(1e3 * stc.times, stc.data[::100, :].T)
    ax.set(xlabel="time (ms)", ylabel="%s value" % method)
    fig, axes = plt.subplots()
    evoked_1.plot(axes=axes)
    if evoked_2 is not None:
        evoked_2.plot(axes=axes)
    for text in list(axes.texts):
        text.remove()
    for line in axes.lines:
        line.set_color("#98df81")
    vertno_max, time_max = stc.get_peak(hemi="lh")

    surfer_kwargs = dict(
            hemi="split",
            subjects_dir=subjects_dir,
            surface = 'pial',
            clim=dict(kind="value", lims=[8, 12, 15]),
            views="lateral",
            initial_time=time_max,
            time_unit="s",
            size=(800, 800),
            smoothing_steps=10,
        )
    brain = stc.plot(**surfer_kwargs)
    if contrast != 0:
        brain.add_text(1, 9, f"dSPM {cond1} VS {cond2}", "title", font_size=24)
    else:
        brain.add_text(1, 9, f"dSPM {cond1}", "title", font_size=24)


plot_inverse('SemGram', cond2='Sem', evoked_1=evoked_sg, evoked_2=evoked_sem, inverse=inverse,
                snr = 3, lambda2_x = 1, contrast=1)
