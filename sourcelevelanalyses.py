'''
===============================================
Source-Level Analysis Code Summary

Project: "Electrophysiological Correlates of Lucid Dreaming: Sensor- and Source-Level Signatures"
Designed and coded by: Çağatay Demirel
===============================================

This script provides a streamlined summary of the source-level analyses 
conducted for this study, focusing on core feature extraction. It encompasses the following key procedures:  

1. Source-Level Power Estimation:
   - dSPM and eLORETA inverse solutions applied to estimate cortical power across conditions 
     (Early REM, Later REM, Lucid Dreaming, Wake).

2. Functional Connectivity Analysis:
   - Spectral connectivity metrics (e.g., PLI, wPLI-debiased) computed across cortical regions 
     for each condition.

3. Lucid Dreaming Extent-Based Analyses:
   - Computation of sensor-level GFP across frequency bands to assess signal strength and variability over time.
   - Source-level power and functional connectivity evaluated around lucidity onset, contrasting
     pre- and post-eye signaling periods.

This code is designed for efficient EEG feature extraction, facilitating further downstream analyses.
'''

import numpy as np
import mne
import gc

from mne.datasets import fetch_fsaverage
import os.path as op

from mne.minimum_norm import apply_inverse_epochs
from mne_connectivity import spectral_connectivity_epochs

from mne.minimum_norm import make_inverse_operator, compute_source_psd_epochs
from scipy.signal import hilbert, savgol_filter
#%% ==================== Functional connectivity between conditions ======================
def source_level_funcconv13(epochs, fmin, fmax, tmin_noisecov=None, tmax_noisecov=None, raw_for_noisecov=None,
                            snr=3.0, noise_cov=True, multi_inverse=False, inverse_method='dSPM', if_avg_ref_info=True):
    
    # ======== Check if its already average re-referenced =========
    avg_proj = any(proj['desc'].lower().startswith('average eeg reference') for proj in epochs.info['projs'])
    if(if_avg_ref_info == True):
        if avg_proj:
            print("An average EEG reference projector is present.")
        else:
            print("No average EEG reference projector found.")
    else:
        print('There will be absolutely no average EEG referencing')
    # ======== Check if its already average re-referenced =========
    
    # ======== Check if noise cov data already average re-referenced =========
    if(noise_cov == True):
        
        avg_proj = any(proj['desc'].lower().startswith('average eeg reference') for proj in raw_for_noisecov.info['projs'])
        if(if_avg_ref_info == True):
            if avg_proj:
                print("An average EEG reference projector is present in baseline for noise cov.")
            else:
                print("No average EEG reference projector found in baseline for noise cov.")
        else:
            print('There will be absolutely no average EEG referencing')
    # ======== Check if noise cov data already average re-referenced =========
    
    montage = mne.channels.make_standard_montage('standard_1005')
    epochs.set_montage(montage)
    
    # =========== Prehandling epochs to set avg rereferencing with further proj normalization ============
    epochs.set_eeg_reference('average', projection=True) #average re-referencing
    epochs.info.normalize_proj()
    epochs.apply_proj()
    # =========== Prehandling epochs to set avg rereferencing with further proj normalization ============

    # =========== Prehandling baseline noise cov raw to set avg rereferencing with further proj normalization ============
    if(noise_cov == True):
        raw_for_noisecov.set_montage(montage)
        raw_for_noisecov.set_eeg_reference('average', projection=True) #average re-referencing
        raw_for_noisecov.info.normalize_proj()
        raw_for_noisecov.apply_proj()
    # =========== Prehandling baseline noise cov raw to set avg rereferencing with further proj normalization ============
    
    # ======= Ready template =========
    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)
    
    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    # ======= Ready template =========
    
    # ======== Forward solution =========
    fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
    print(fwd)
    # ======== Forward solution =========
    
    # ==== Compute regularized noise covariance =====
    if(noise_cov == True):
        noise_cov = mne.compute_raw_covariance(raw_for_noisecov, tmin=tmin_noisecov, tmax=tmax_noisecov, method='empirical', rank=None, verbose=True, n_jobs=-1)
    else:
        noise_cov = mne.make_ad_hoc_cov(epochs.info)
    # ==== Compute regularized noise covariance =====
    
    # ======= Inverse operator ========
    inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, 
                                             depth=0.8)
    del fwd
    # ======= Inverse operator ========
    
    # ========================== Compute inverse solution of con =================================
    # snr = 3.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    
    con_res_list = list()
    
    if(multi_inverse == True):
        
        for selected_inverse_method in inverse_method:
    
            method = selected_inverse_method  # use dSPM method (could also be MNE or sLORETA)
            stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                        pick_ori="normal", return_generator=True)
            # ====== Compute inverse solution ======
            
            # ========== Get labels =========
            # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
            labels = mne.read_labels_from_annot(subject, parc='aparc',
                                                subjects_dir=subjects_dir)
            labels = labels[0:68]
            label_colors = [label.color for label in labels]
            label_colors = label_colors[0:68]
            # ========== Get labels =========
            
            # ==== Average the source estimates within each label using sign-flips to reduce
            # signal cancellations, also here we return a generator =====
            
            src = inverse_operator['src']
            adjacency = mne.spatial_src_adjacency(src)
            label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip', 
                                                     return_generator=True)
            
            sfreq = epochs.info['sfreq']  # the sampling frequency
            # connection_methods = ['coh', 'pli', 'wpli2_debiased', 'ciplv']
            connection_methods = ['coh', 'pli', 'wpli2_debiased']
            con = spectral_connectivity_epochs(label_ts, method=connection_methods, mode='multitaper', 
                                               sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, 
                                               mt_adaptive=True, n_jobs=-1)
            
            # con is a 3D array, get the connectivity for the first (and only) freq. band
            # for each method
            con_res = dict()
            for method, c in zip(connection_methods, con):
                con_res[method] = c.get_data(output='dense')[:, :, 0]
            # con_res[connection_methods[0]] = con.get_data(output='dense')[:, :, 0]
            
            con_res_list.append(con_res)
            
    else:
        
        method = inverse_method  # use dSPM method (could also be MNE or sLORETA)
        stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                    pick_ori="normal", return_generator=True)
        # ====== Compute inverse solution ======
        
        # ========== Get labels =========
        # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
        labels = mne.read_labels_from_annot(subject, parc='aparc',
                                            subjects_dir=subjects_dir)
        labels = labels[0:68]
        label_colors = [label.color for label in labels]
        label_colors = label_colors[0:68]
        # ========== Get labels =========
        
        # ==== Average the source estimates within each label using sign-flips to reduce
        # signal cancellations, also here we return a generator =====
        
        src = inverse_operator['src']
        adjacency = mne.spatial_src_adjacency(src)
        label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip', 
                                                 return_generator=True)
        
        sfreq = epochs.info['sfreq']  # the sampling frequency
        # connection_methods = ['coh', 'pli', 'wpli2_debiased', 'ciplv']
        connection_methods = ['coh', 'pli', 'wpli2_debiased']
        con = spectral_connectivity_epochs(label_ts, method=connection_methods, mode='multitaper', 
                                           sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, 
                                           mt_adaptive=True, n_jobs=-1)
        
        # con is a 3D array, get the connectivity for the first (and only) freq. band
        # for each method
        con_res = dict()
        for method, c in zip(connection_methods, con):
            con_res[method] = c.get_data(output='dense')[:, :, 0]
        # con_res[connection_methods[0]] = con.get_data(output='dense')[:, :, 0]
        
        con_res_list.append(con_res)
    # ========================== Compute inverse solution of con =================================
        
    return con_res_list, labels, label_colors, con, adjacency

#%% ======= Source level functional connectivity calculation v1.3 =======
import time 

all_epochs = 'load'
epochs_list = 'load'

all_coh_res_list_dSPM = list()
all_coh_res_list_eLORETA = list()

fmin = (2, 4, 8, 12, 30, 36) #hz
fmax = (4, 8, 12, 30, 36, 45) #hz
normalize_proj = [True for d in range(7)] + [True for d in range(2)] + [True for d in range(5)] + [True for d in range(5)]

for i in range(4):
    temp_coh_res_list_allfreqbands_dSPM = list()
    temp_coh_res_list_allfreqbands_eLORETA = list()
    for j in range(6):
        temp_coh_res_freqband_dSPM = list()
        temp_coh_res_freqband_eLORETA = list()
        for k in range(19):
            
            begin = time.time()
            
            temp_coh_res_list, _, _, _, _ = source_level_funcconv13(epochs = all_epochs[i][k].copy(), fmin=fmin[j], fmax=fmax[j], snr=3, noise_cov=False, multi_inverse=True, 
                                                                    inverse_method=['dSPM', 'eLORETA'], 
                                                                    raw_for_noisecov=None, 
                                                                    tmin_noisecov=None, 
                                                                    tmax_noisecov=None, if_avg_ref_info=True)
            
            temp_coh_res_freqband_dSPM.append(temp_coh_res_list[0])
            temp_coh_res_freqband_eLORETA.append(temp_coh_res_list[1])
            
            end = time.time()
            print(end - begin)
            print(str(i) + "_" + str(j) + "_" + str(k))
            
        temp_coh_res_list_allfreqbands_dSPM.append(temp_coh_res_freqband_dSPM)
        temp_coh_res_list_allfreqbands_eLORETA.append(temp_coh_res_freqband_eLORETA)
    
    all_coh_res_list_dSPM.append(temp_coh_res_list_allfreqbands_dSPM)
    all_coh_res_list_eLORETA.append(temp_coh_res_list_allfreqbands_eLORETA)
#%% ==================== Source-level power between conditions ======================
'''
data load & preparation
'''

def source_level_surface_mne_v4(epochs, fmin, fmax, raw_for_noisecov, tmin_noisecov, tmax_noisecov, normalize_proj=False, 
                              snr=3.0, noise_cov=True, inverse_method='dSPM', multi_inverse=False):
    
    if_avg_ref_info = True
    # ======== Check if its already average re-referenced =========
    avg_proj = any(proj['desc'].lower().startswith('average eeg reference') for proj in epochs.info['projs'])
    if(if_avg_ref_info == True):
        if avg_proj:
            print("An average EEG reference projector is present.")
        else:
            print("No average EEG reference projector found.")
    else:
        print('There will be absolutely no average EEG referencing')
    # ======== Check if its already average re-referenced =========
    
    # ======== Check if noise cov data already average re-referenced =========
    avg_proj = any(proj['desc'].lower().startswith('average eeg reference') for proj in raw_for_noisecov.info['projs'])
    if(if_avg_ref_info == True):
        if avg_proj:
            print("An average EEG reference projector is present in baseline for noise cov.")
        else:
            print("No average EEG reference projector found in baseline for noise cov.")
    else:
        print('There will be absolutely no average EEG referencing')
    # ======== Check if noise cov data already average re-referenced =========
    
    montage = mne.channels.make_standard_montage('standard_1005')
    epochs.set_montage(montage)
    raw_for_noisecov.set_montage(montage)
    
    # =========== Prehandling epochs to set avg rereferencing with further proj normalization ============
    epochs.set_eeg_reference('average', projection=True) #average re-referencing
    epochs.info.normalize_proj()
    epochs.apply_proj()
    # =========== Prehandling epochs to set avg rereferencing with further proj normalization ============

    # =========== Prehandling baseline noise cov raw to set avg rereferencing with further proj normalization ============
    raw_for_noisecov.set_eeg_reference('average', projection=True) #average re-referencing
    raw_for_noisecov.info.normalize_proj()
    raw_for_noisecov.apply_proj()
    # =========== Prehandling baseline noise cov raw to set avg rereferencing with further proj normalization ============
    
    # ======= Ready template =========
    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)
    
    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    # ======= Ready template =========
    
    # ======== Forward solution =========
    fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
    print(fwd)
    # ======== Forward solution =========
    
    # ==== Compute regularized noise covariance =====
    if noise_cov:
        noise_cov = mne.compute_raw_covariance(raw_for_noisecov, tmin=tmin_noisecov, tmax=tmax_noisecov, method='empirical', rank=None, verbose=True, n_jobs=-1)
    else:
        noise_cov = mne.make_ad_hoc_cov(epochs.info)
    # ==== Compute regularized noise covariance =====
    
    # ======= Inverse operator ========
    inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov, loose='auto', depth=0.8)  # loose parameter adjusted for surface model
    del fwd
    # ======= Inverse operator ========
    
    # =============  Apply the inverse operator to each condition ===============
    lambda2 = 1.0 / snr ** 2
    
    stcs_list = list()
    if(multi_inverse == True):
        
        for selected_inverse_method in inverse_method:
            
            # =============  Apply the inverse operator to each condition ===============
            # snr = 3.0  # use lower SNR for single epochs
            lambda2 = 1.0 / snr ** 2
            
            stcs_list_temp = list()
            for condition in epochs.event_id.keys():
                    
                stcs = compute_source_psd_epochs(
                epochs[condition],
                inverse_operator,
                lambda2=lambda2,
                method=selected_inverse_method,
                fmin=fmin,
                fmax=fmax,
                bandwidth=4,
                label=None,
                return_generator=True,
                verbose=True,
                n_jobs=-1
                )
                
                # compute average PSD over the first 10 epochs
                psd_avg = np.zeros(20484)
                for i, stc in enumerate(stcs):
                    psd_avg += stc.mean().data[:,0]
                psd_avg /= len(epochs[condition])
                stcs_list_temp.append(psd_avg)  # overwrite the last epoch's data with the average
            # =============  Apply the inverse operator to each condition ===============
            
            stcs_list.append(stcs_list_temp)
            
    else:
        
        # =============  Apply the inverse operator to each condition ===============
        for condition in epochs.event_id.keys():
                
            stcs = compute_source_psd_epochs(
            epochs[condition],
            inverse_operator,
            lambda2=lambda2,
            method=inverse_method,
            fmin=fmin,
            fmax=fmax,
            bandwidth=4,
            label=None,
            return_generator=True,
            verbose=True,
            n_jobs=-1
            )
            
            # compute average PSD over the first 10 epochs
            psd_avg = np.zeros(20484)
            for i, stc in enumerate(stcs):
                psd_avg += stc.mean().data[:,0]
            psd_avg /= len(epochs[condition])
            stcs_list.append(psd_avg)  # overwrite the last epoch's data with the average
        # =============  Apply the inverse operator to each condition ===============

    return stcs_list

#%% ========================= Multiple MNE variations (dSPM + eLORETA) no noise covariance=======================
fmin = (2, 4, 8, 12, 30, 36) #hz
fmax = (4, 8, 12, 30, 36, 45) #hz

all_stc_avgs_dSPM = np.zeros((4, 6, 19, 20484))
all_stc_avgs_eLORETA = np.zeros((4, 6, 19, 20484))

normalize_proj = [True for d in range(7)] + [True for d in range(2)] + [True for d in range(5)] + [True for d in range(5)]

longraw = 'load'
chosen_all_relativitimeintervals = 'load'

conditions = ['Early REM', 'Later REM', 'Lucid Dreaming', 'Wake']
tot_count = 6 * 19
for b in range(6):
    for i in range(19):
        print('======== New estimation =======')
        begin = time.time()
        
        stcs = source_level_surface_mne_v4(epochs = epochs_list[i].copy(), fmin=fmin[b], fmax=fmax[b], normalize_proj=normalize_proj[i], 
                                         snr=3, noise_cov=False, multi_inverse=True, inverse_method=['eLORETA', 'dSPM'], raw_for_noisecov=longraw[i].copy(), 
                                         tmin_noisecov=chosen_all_relativitimeintervals[i,0], tmax_noisecov=chosen_all_relativitimeintervals[i,1])  
        
        for k in range(4):
            all_stc_avgs_eLORETA[k,b,i] = stcs[0][k]
        for k in range(4):
            all_stc_avgs_dSPM[k,b,i] = stcs[1][k]
        
        end = time.time()
        print(end - begin)
        print('======== New estimation =======')
        print('Progress, percentage -> %0.2f' % (((b*19 + i) / tot_count) * 100))
#%% ==================== LD extent GFP analysis ======================

# =========== Preparatory functions =============
def envelopeCreator(timeSignal, degree=3, intervalLength=51, hilbert_transform=False):
    if(hilbert_transform == True):
        timeSignal = np.abs(hilbert(timeSignal))
        
    amplitude_envelopeFiltered = savgol_filter(timeSignal, intervalLength, degree)
    return amplitude_envelopeFiltered  

def gfp_from_raw_v2(raw, crop_onset, duration, Fs, baseline=(0,10), mode='mean', intervalLength=51):
    
    edge_length = intervalLength / Fs / 2
    edge_crop = int(Fs * edge_length)

    raw.crop(crop_onset - edge_length, crop_onset + duration + edge_length) #initial crop with edge in both sides
    raw_data = envelopeCreator(raw.copy()._data, degree=3, intervalLength=intervalLength, hilbert_transform=True)
    
    raw.crop(edge_length, edge_length + duration) #further crop by removing edges
    raw_data = raw_data[:, edge_crop : -1 * edge_crop]

    times = raw.times
    gfp = raw_data**2
    gfp_baselinecorrected = mne.baseline.rescale(gfp, times, baseline=baseline, mode=mode) #baseline with first 5 seconds
    
    return gfp_baselinecorrected
#%% ================ GFP extraction ================
duration, Fs = 30, 100 #Hz
freq_bands = [(2, 4), (4, 8), (8, 12), (12,30), (30, 36), (36, 45), (2, 48)]

gfp_avg_allfreqbands = np.zeros((7, 64, duration * Fs + 1))
gfp_allsubjects_allfreqbands = np.zeros((7, 19, 64, duration * Fs + 1))
baseline = (0,5)
mode = 'ratio'
intervalLength = 305

longepochchunk_d1_1, longepochchunk_d1_2, longepochchunk_d2, longepochchunk_d3, ch_names1, ch_names2, ch_names3 = 'load'

longepochchunk_d1_1_int = [None, ['T8','FC4'],None,None,None]
longepochchunk_d1_2_int = [['CPz', 'PO4', 'Cz'], None, None, ['F5'], None, None]
longepochchunk_d2_int = [['FTT8h'], None]
longepochchunk_d3_int = [None, None, ['FFT8h', 'FC4', 'FC6'], None, None, None, None]

d11_on, d12_on, d2_on, d3_on = 'load'

for j in range(len(freq_bands)):
    
    ''' =========== Extract GFP from multiple rec d1-1 ============ '''

    # ===== Check initial crop information of each rec =====
    for i in range(len(longepochchunk_d1_1)):
        print(longepochchunk_d1_1[i][2])
    # ===== Check initial crop information of each rec =====
    
    # ===== Extract time initially =======
    temp_raw = longepochchunk_d1_1[0][0].copy()
    temp_raw.crop(0, duration) # 1 + 1 = 2 minutes
    times = temp_raw.times
    # ===== Extract time initially =======
    
    gfps_BCI = np.zeros((5, 64, duration * Fs + 1))
    chosen_datalabels = [0,2,4,6,8]
    for i in range(5):
        temp_raw = longepochchunk_d1_1[chosen_datalabels[i]][0].copy()
        temp_raw.pick_types(eeg=True)
        
        if(longepochchunk_d1_1_int[i] is not None):
            temp_raw.info['bads'] = longepochchunk_d1_1_int[i]
            temp_raw.interpolate_bads()
            
        ''' ======= Add artificial channels ========== '''
        new_ch_names = list(set(ch_names3) - set(ch_names2))
        
        fake_raw = temp_raw.copy().pick_channels(['Fp1', 'Fz', 'F3', 'F7', 'FC5'])
        fake_raw.rename_channels(mapping={'Fp1': new_ch_names[0], 'Fz': new_ch_names[1], 
                                             'F3': new_ch_names[2], 'F7': new_ch_names[3],
                                             'FC5': new_ch_names[4]})
        temp_raw.add_channels([fake_raw])
        
        # ======= Remontage ========
        montage = mne.channels.make_standard_montage('standard_1005')
        temp_raw.set_montage(montage)
        # ======= Remontage ========
        
        # ====== Interpolate ======
        temp_raw.info['bads'] = new_ch_names
        temp_raw.interpolate_bads()
        # ====== Interpolate ======
        
        ''' ======= Add artificial channels ========== '''
        
        temp_raw.resample(Fs)
        temp_raw.filter(l_freq=freq_bands[j][0], h_freq=freq_bands[j][1], method='iir')
        
        temp_gfp = gfp_from_raw_v2(raw=temp_raw.copy(), crop_onset=d11_on[i] - int(duration / 2), Fs=Fs, duration=duration, baseline=baseline, mode=mode, intervalLength=intervalLength)
        gfps_BCI[i] = temp_gfp
    ''' =========== Extract GFP from multiple rec d1-1 ============  '''
    #%
    ''' =========== Extract GFP from multiple rec d1-2 ============ '''
    
    # ===== Check initial crop information of each rec =====
    for i in range(len(longepochchunk_d1_2)):
        print(longepochchunk_d1_2[i][2])
    # ===== Check initial crop information of each rec =====
    
    # ===== Extract time initially =======
    temp_raw = longepochchunk_d1_2[0][0].copy()
    temp_raw.crop(0, duration) # 1 + 1 = 2 minutes
    times = temp_raw.times
    # ===== Extract time initially =======
    
    gfps_BCImeditation = np.zeros((5, 64, duration * Fs + 1))
    for i in range(5):
        temp_raw = longepochchunk_d1_2[i][0].copy()
        temp_raw.pick_types(eeg=True)
        
        if(longepochchunk_d1_2_int[i] is not None):
            temp_raw.info['bads'] = longepochchunk_d1_2_int[i]
            temp_raw.interpolate_bads()
        
        temp_raw.resample(Fs)
        temp_raw.filter(l_freq=freq_bands[j][0], h_freq=freq_bands[j][1], method='iir')
        # temp_raw.pick_channels(ch_names=common_channels)
            
        temp_gfp = gfp_from_raw_v2(raw=temp_raw.copy(), crop_onset=d12_on[i] - int(duration / 2), Fs=Fs, duration=duration, baseline=baseline, mode=mode, intervalLength=intervalLength)
        gfps_BCImeditation[i] = temp_gfp   
    ''' =========== Extract GFP from multiple rec d1-2 ============ '''
    #%
    ''' =========== Extract GFP from multiple rec d2 ============ '''
    
    # ===== Check initial crop information of each rec =====
    for i in range(len(longepochchunk_d2)):
        print(longepochchunk_d2[i][2])
    # ===== Check initial crop information of each rec =====
    
    # ===== Extract time initially =======
    temp_raw = longepochchunk_d2[0][0].copy()
    temp_raw.crop(0, duration) # 1 + 1 = 2 minutes
    times = temp_raw.times
    # ===== Extract time initially =======
    
    gfps_LDCueing = np.zeros((2, 64, duration * Fs + 1))
    for i in range(2):
        temp_raw = longepochchunk_d2[i][0].copy()
        temp_raw.pick_types(eeg=True)
        
        if(longepochchunk_d2_int[i] is not None):
            temp_raw.info['bads'] = longepochchunk_d2_int[i]
            temp_raw.interpolate_bads()
        
        ''' ======= Add artificial channels ========== '''
        new_ch_names = list(set(ch_names3) - set(ch_names1))
        
        fake_raw = temp_raw.copy().pick_channels(['Fp1', 'Fz'])
        fake_raw.rename_channels(mapping={'Fp1': new_ch_names[0], 'Fz': new_ch_names[1]})
        temp_raw.add_channels([fake_raw])
        # ====== Add artificial channels =======
        
        # ======= Remontage ========
        montage = mne.channels.make_standard_montage('standard_1005')
        temp_raw.set_montage(montage)
        # ======= Remontage ========
        
        # ====== Interpolate ======
        temp_raw.info['bads'] = new_ch_names
        temp_raw.interpolate_bads()
        # ====== Interpolate ======
        
        new_ch_names2 = list(set(ch_names3) - set(temp_raw.ch_names)) #trial
        ''' ======= Add artificial channels ========== '''
        
        temp_raw.resample(Fs)
        temp_raw.filter(l_freq=freq_bands[j][0], h_freq=freq_bands[j][1], method='iir')
        # temp_raw.pick_channels(ch_names=ch_names3)
        try:
            temp_raw.pick_channels(ch_names=ch_names3)
        except IndexError as e:
            temp_raw.pick_channels(ch_names=ch_names3)
            print('error :' + str(e))
        
        temp_gfp = gfp_from_raw_v2(raw=temp_raw.copy(), crop_onset=d2_on[i] - int(duration / 2), Fs=Fs, duration=duration, baseline=baseline, mode=mode, intervalLength=intervalLength)
        gfps_LDCueing[i] = temp_gfp
    ''' =========== Extract GFP from multiple rec d2 ============ '''
    #%   
    ''' =========== Extract GFP from multiple rec d3 ============ '''
    
    # ===== Check initial crop information of each rec =====
    for i in range(len(longepochchunk_d3)):
        print(longepochchunk_d3[i][2])
    # ===== Check initial crop information of each rec =====
    
    # ===== Extract time initially =======
    temp_raw = longepochchunk_d3[0][0].copy()
    temp_raw.crop(0, duration) # 1 + 1 = 2 minutes
    times = temp_raw.times
    # ===== Extract time initially =======
    
    gfps_Lucireta = np.zeros((7, 64, duration * Fs + 1))
    for i in range(7):
        temp_raw = longepochchunk_d3[i][0].copy()
        temp_raw.pick_types(eeg=True)
        
        if(longepochchunk_d3_int[i] is not None):
            temp_raw.info['bads'] = longepochchunk_d3_int[i]
            temp_raw.interpolate_bads()
        
        temp_raw.resample(Fs)
        temp_raw.filter(l_freq=freq_bands[j][0], h_freq=freq_bands[j][1], method='iir')
        temp_raw.pick_channels(ch_names=ch_names3)
        
        temp_gfp = gfp_from_raw_v2(raw=temp_raw.copy(), crop_onset=d3_on[i] - int(duration / 2), Fs=Fs, duration=duration, baseline=baseline, mode=mode, intervalLength=intervalLength)
        gfps_Lucireta[i] = temp_gfp
    ''' =========== Extract GFP from multiple rec d3 ============ '''
    
    ''' ========== Merging individual gfps for unity ============ '''
    gfps_all = np.row_stack((gfps_BCI, gfps_BCImeditation, gfps_LDCueing, gfps_Lucireta))
    
    # ====== Min - max scaling (standardization) =======
    # for i in range(19):
    #     gfps_all[i] /= np.max(gfps_all[i])
    # ====== Min - max scaling (standardization) =======
    
    gfps_avg = np.mean(gfps_all, axis=0)
    ''' ========== Merging individual gfps for unity ============ '''
    
    gfp_avg_allfreqbands[j] = gfps_avg
    gfp_allsubjects_allfreqbands[j] = gfps_all #all subject data
    
    print('***** Freq band %d has finished *****' % j)
    
#%% ==================== LD extent source-level power analyses ======================
def source_level_surface_mne_temporal_v5(epochs, fmin, fmax, baseline=(0,5), experiment=(0,None), normalize_proj=False,
                              snr=3.0, noise_cov=True, inverse_method='dSPM', multi_inverse=False, if_avg_ref_info=True):
    
    # ======== Check if its already average re-referenced =========
    avg_proj = any(proj['desc'].lower().startswith('average eeg reference') for proj in epochs.info['projs'])
    if(if_avg_ref_info == True):
        if avg_proj:
            print("An average EEG reference projector is present.")
        else:
            print("No average EEG reference projector found.")
    else:
        print('There will be absolutely no average EEG referencing, no no no.')
    # ======== Check if its already average re-referenced =========
    
    montage = mne.channels.make_standard_montage('standard_1005')
    epochs.set_montage(montage)
    
    # =========== Prehandling epochs to set avg rereferencing with further proj normalization ============
    epochs.set_eeg_reference('average', projection=True) #average re-referencing
    epochs.info.normalize_proj()
    epochs.apply_proj()
    # =========== Prehandling epochs to set avg rereferencing with further proj normalization ============
    
    # =========== Separate baseline and experimental single epochs via cropping ============
    epochs_experimental = epochs.copy().crop(experiment[0], experiment[1])
    epochs_baseline = epochs.copy().crop(baseline[0], baseline[1])
    # =========== Separate baseline and experimental single epochs via cropping ============
    
    # ======= Ready template =========
    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)
    
    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    # ======= Ready template =========
    
    # ======== Forward solution =========
    fwd = mne.make_forward_solution(epochs_experimental.info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
    print(fwd)
    # ======== Forward solution =========
    
    # ==== Compute regularized noise covariance =====
    if noise_cov:
        noise_cov = mne.compute_covariance(epochs, tmin=baseline[0], tmax=baseline[1], method='empirical', rank=None, verbose=True, n_jobs=-1)
    else:
        noise_cov = mne.make_ad_hoc_cov(epochs.info)
    # ==== Compute regularized noise covariance =====
    
    # ======= Inverse operator ========
    inverse_operator = make_inverse_operator(epochs_experimental.info, fwd, noise_cov, loose='auto', depth=0.8)  # loose parameter adjusted for surface model
    del fwd
    # ======= Inverse operator ========
    
    # =============  Apply the inverse operator to each condition ===============
    lambda2 = 1.0 / snr ** 2
    
    stcs_normalized_list = list()
    if(multi_inverse == True):
        
        for selected_inverse_method in inverse_method:
    
            stc_experimental = compute_source_psd_epochs(
            epochs_experimental,
            inverse_operator,
            lambda2=lambda2,
            method=selected_inverse_method,
            fmin=fmin,
            fmax=fmax,
            bandwidth=4,
            label=None,
            return_generator=False,
            verbose=True,
            n_jobs=-1
            )[0].mean()
            
            stc_baseline = compute_source_psd_epochs(
            epochs_baseline,
            inverse_operator,
            lambda2=lambda2,
            method=selected_inverse_method,
            fmin=fmin,
            fmax=fmax,
            bandwidth=4,
            label=None,
            return_generator=False,
            verbose=True,
            n_jobs=-1
            )[0].mean()
            
            stc_baseline = stc_baseline._data[:,0]
            stc_experimental = stc_experimental._data[:,0]
            
            # ========== Baseline Normalization ==========
            stc_contrasted = (stc_experimental - stc_baseline)
            stc_normed = stc_contrasted / np.max(np.abs(stc_contrasted))
            
            stcs_normalized_list.append(stc_normed)
            
            print('normalization applied')
            # ========== Baseline Normalization ==========
    
    else:
            
            stc_experimental = compute_source_psd_epochs(
            epochs_experimental,
            inverse_operator,
            lambda2=lambda2,
            method=inverse_method,
            fmin=fmin,
            fmax=fmax,
            bandwidth=4,
            label=None,
            return_generator=False,
            verbose=True,
            n_jobs=-1
            )[0].mean()
            
            stc_baseline = compute_source_psd_epochs(
            epochs_baseline,
            inverse_operator,
            lambda2=lambda2,
            method=inverse_method,
            fmin=fmin,
            fmax=fmax,
            bandwidth=4,
            label=None,
            return_generator=False,
            verbose=True,
            n_jobs=-1
            )[0].mean()
            
            stc_baseline = stc_baseline._data[:,0]
            stc_experimental = stc_experimental._data[:,0]
            
            # ========== Baseline Normalization ==========
            # Example: Subtracting baseline STC from experimental STC
        
            stc_contrasted = (stc_experimental - stc_baseline)
            stc_normed = stc_contrasted / np.max(np.abs(stc_contrasted))
            
            stcs_normalized_list.append(stc_normed)
            
            print('normalization applied')
            # ========== Baseline Normalization ==========

    return stcs_normalized_list
#%% ======= Source-level activation analysis ==========
epochs_temporal = 'load'

# ======================== Mode subtract / baseline =======================
bands = ['30-36Hz', '36-45Hz']
fminmax = [(30,36), (36,45)]
experiment_tminmax = [(15.25, 19.75), (14,20)]

all_stcs_dSPM_maxdivision = dict()
all_stcs_eLORETA_maxdivision = dict()

for i in range(4):
    
    temp_stcs_dSPM_maxdivision = list()
    temp_stcs_eLORETA_maxdivision = list()
    
    for j in range(19):
        
        temp_epochs = epochs_temporal[j].copy()
        
        temp_epochs.drop_channels(['TP10', 'Iz', 'FT10', 'FT9', 'TP9'])
        
        temp_stc = source_level_surface_mne_temporal_v5(temp_epochs, fmin=fminmax[i][0], fmax=fminmax[i][1], baseline=(0,5), experiment=(experiment_tminmax[i][0], experiment_tminmax[i][1]), 
                                             normalize_proj=False, snr=3.0, noise_cov=True, inverse_method=['dSPM', 'eLORETA'], multi_inverse=True)
        
        temp_stcs_dSPM_maxdivision.append(temp_stc[0]['maxdivision_normed'])
        temp_stcs_eLORETA_maxdivision.append(temp_stc[1]['maxdivision_normed'])

        del temp_stc
        del temp_epochs
        gc.collect()
        
        print('Surface source-localization of LD extend, recording ' + str(j) + ' has finished')
        
    all_stcs_dSPM_maxdivision[bands[i]] = temp_stcs_dSPM_maxdivision
    all_stcs_eLORETA_maxdivision[bands[i]] = temp_stcs_eLORETA_maxdivision
# ======================== Mode subtract / baseline ======================
#%% ==================== LD extent source-level functional connectivity analyses ======================
def source_level_funccon_v4_temporal(epochs, fmin, fmax, baseline=(0, 5), experiment=(0, None),
                                     snr=3.0, noise_cov=True, multi_inverse=False, inverse_method='dSPM', if_avg_ref_info=True):

    # ======== Check if it's already average re-referenced =========
    avg_proj = any(proj['desc'].lower().startswith('average eeg reference') for proj in epochs.info['projs'])
    if if_avg_ref_info:
        if avg_proj:
            print("An average EEG reference projector is present.")
        else:
            print("No average EEG reference projector found.")
    else:
        print("There will be absolutely no average EEG referencing! (no!)")

    montage = mne.channels.make_standard_montage('standard_1005')
    epochs.set_montage(montage)

    # =========== Prehandling epochs to set avg rereferencing with further proj normalization ============
    epochs.set_eeg_reference('average', projection=True)  # average re-referencing
    epochs.info.normalize_proj()
    epochs.apply_proj()
    # =========== Prehandling epochs to set avg rereferencing with further proj normalization ============

    # =========== Separate baseline and experimental single epochs via cropping ============
    epochs_experimental = epochs.copy().crop(experiment[0], experiment[1])
    epochs_baseline = epochs.copy().crop(baseline[0], baseline[1])
    # =========== Separate baseline and experimental single epochs via cropping ============

    # ======= Ready template =========
    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    # ======= Ready template =========

    # ======== Forward solution =========
    fwd = mne.make_forward_solution(epochs_experimental.info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
    print(fwd)
    # ======== Forward solution =========

    # ==== Compute regularized noise covariance =====
    if noise_cov:
        noise_cov = mne.compute_covariance(epochs, tmin=baseline[0], tmax=baseline[1], method='empirical', rank=None, verbose=True, n_jobs=-1)
    else:
        noise_cov = mne.make_ad_hoc_cov(epochs.info)
    # ==== Compute regularized noise covariance =====

    # ======= Inverse operator ========
    inverse_operator = make_inverse_operator(epochs_experimental.info, fwd, noise_cov, loose='auto', depth=0.8)  # loose parameter adjusted for surface model
    del fwd
    # ======= Inverse operator ========

    # ========================== Compute inverse solution of con =================================
    lambda2 = 1.0 / snr ** 2
    con_res_list = list()

    inverse_methods = inverse_method if multi_inverse else [inverse_method]

    for method in inverse_methods:
        # Apply inverse solution
        stcs_experimental = apply_inverse_epochs(epochs_experimental, inverse_operator, lambda2, method,
                                                 pick_ori="normal", return_generator=True)

        stcs_baseline = apply_inverse_epochs(epochs_baseline, inverse_operator, lambda2, method,
                                             pick_ori="normal", return_generator=True)

        # ========== Get labels =========
        labels = mne.read_labels_from_annot(subject, parc='aparc',
                                            subjects_dir=subjects_dir)
        labels = labels[:68]  # Extract 68 cortical labels
        label_colors = [label.color for label in labels]

        # ==== Extract time courses from each label ====
        src = inverse_operator['src']
        adjacency = mne.spatial_src_adjacency(src)
        label_ts_experimental = mne.extract_label_time_course(stcs_experimental, labels, src, mode='mean_flip',
                                                              return_generator=True)
        label_ts_baseline = mne.extract_label_time_course(stcs_baseline, labels, src, mode='mean_flip',
                                                          return_generator=True)

        sfreq = epochs.info['sfreq']
        # connection_methods = ['coh', 'pli', 'wpli2_debiased']
        connection_methods = ['dpli', 'ciplv', 'ppc']

        # Compute connectivity for experimental
        con_exp = spectral_connectivity_epochs(label_ts_experimental, method=connection_methods, mode='multitaper',
                                               sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True,
                                               mt_adaptive=True, n_jobs=-1)

        # Compute connectivity for baseline
        con_base = spectral_connectivity_epochs(label_ts_baseline, method=connection_methods, mode='multitaper',
                                                sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True,
                                                mt_adaptive=True, n_jobs=-1)

        # Extract connectivity matrices
        con_res_exp = {method: c.get_data(output='dense')[:, :, 0] for method, c in zip(connection_methods, con_exp)}
        con_res_base = {method: c.get_data(output='dense')[:, :, 0] for method, c in zip(connection_methods, con_base)}

        # ================= Apply Baseline Normalization =================
        con_res_normed = {}

        for method in connection_methods:

            con_res_normed[method] = (con_res_exp[method] - con_res_base[method]) / con_res_base[method]
            print(f'Percent normalization applied for {method}')

        # ================================================================

        con_res_list.append(con_res_normed)

    return con_res_list, labels, label_colors, adjacency

#%% ======= Source-level functional connectivity activation analysis ==========
'''
data load & preparation
'''

# ======================== Mode subtract / baseline =======================
bands = ['30-36Hz', '36-45Hz']
fminmax = [(30, 36), (36, 45)]
experiment_tminmax = [(15.25, 19.75), (14, 20)]

# Initialize storage dictionaries
all_stcs_dSPM_percentnorm_fc = dict()
all_stcs_eLORETA_percentnorm_fc = dict()
all_stcs_dSPM_maxdivision_fc = dict()
all_stcs_eLORETA_maxdivision_fc = dict()

for i in range(2):
    
    temp_stcs_dSPM_percentnorm = {'dpli': []}
    temp_stcs_eLORETA_percentnorm = {'dpli': []}
    temp_stcs_dSPM_maxdivision = {'dpli': []}
    temp_stcs_eLORETA_maxdivision = {'dpli': []}
    
    for j in range(19):  # Loop over recordings
        
        temp_epochs = epochs_temporal[j].copy()
        temp_epochs.drop_channels(['TP10', 'Iz', 'FT10', 'FT9', 'TP9'])

        # Compute functional connectivity for source-level
        temp_stc, _, _, _ = source_level_funccon_v4_temporal(
            temp_epochs, fmin=fminmax[i][0], fmax=fminmax[i][1],
            baseline=(0, 5), experiment=(experiment_tminmax[i][0], experiment_tminmax[i][1]),
            snr=3.0, noise_cov=True, inverse_method=['dSPM', 'eLORETA'], multi_inverse=True)

        # ===== Store results for dSPM =====
        for method in ['dpli']:
            temp_stcs_dSPM_percentnorm[method].append(temp_stc[0][method]['percent_normed'])

        # ===== Store results for eLORETA =====
        for method in ['dpli']:
            temp_stcs_eLORETA_percentnorm[method].append(temp_stc[1][method]['percent_normed'])

        # Free memory
        del temp_stc
        del temp_epochs
        gc.collect()
        
        print(f'Surface source-localization of LD extend, recording {j} has finished')
    
    # =============================== Convert list to numpy ===================================
    dpli_dspm_percent_numpy = np.zeros((19, 68, 68))
    
    for k in range(19):
        dpli_dspm_percent_numpy[k] = temp_stcs_dSPM_percentnorm['dpli'][k]
        
    temp_stcs_dSPM_percentnorm['dpli'] = dpli_dspm_percent_numpy

    ''' ================================================ '''
    
    dpli_eLORETA_percent_numpy = np.zeros((19, 68, 68))

    for k in range(19):
        dpli_eLORETA_percent_numpy[k] = temp_stcs_eLORETA_percentnorm['dpli'][k]

    temp_stcs_eLORETA_percentnorm['dpli'] = dpli_eLORETA_percent_numpy
    # =============================== Convert list to numpy ===================================

    # Store in main dictionary
    all_stcs_dSPM_percentnorm_fc[bands[i]] = temp_stcs_dSPM_percentnorm
    all_stcs_eLORETA_percentnorm_fc[bands[i]] = temp_stcs_eLORETA_percentnorm
# ======================== Mode subtract / baseline ======================