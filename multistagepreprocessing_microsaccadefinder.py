import mne
import numpy as np
import os
from mne.preprocessing import (
    ICA, create_eog_epochs, create_ecg_epochs, corrmap)
import matplotlib.pyplot as plt
import cv2
import pickle

from pyprep.prep_pipeline import PrepPipeline
import mne
import yasa
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
from scipy.signal import hilbert

from mne.io import concatenate_raws, read_raw_edf
import time
from scipy.signal import find_peaks

import extremeEEGSignalAnalyzer as chetto_EEG
chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()
#%% ============= Multi-stage preprocessing pipeline ================
'''
Use case scenario:
    
raw_preprocessed, raw_robustzscored, raw_preeyesignal, raw_robustzscored_preeyesignal, temp_lucidity_period = whole_preprocessing(raw=raw, temp_lucidity_period=lucidity_period_fild_new.copy()[j], bad_segments=None, j=j, 
                                                                                                                                  ransac=False, line_noise=50, initial_l_freq=None, ifpyprep=False, avg_reference=False)
'''

def whole_preprocessing(raw, temp_lucidity_period=None, bad_segments=None, j=0, ransac=False, line_noise=50, initial_l_freq=None, ifpyprep=True, avg_reference=False,
                        tail=True):

    if(temp_lucidity_period is not None):
        
        # ===== Initial crop ======
        if(tail == True):
            absolute_begin, absolute_end = np.min(temp_lucidity_period) - 10, np.max(temp_lucidity_period) + 10
        else:
            absolute_begin, absolute_end = np.min(temp_lucidity_period), np.max(temp_lucidity_period)
        raw.crop(absolute_begin, absolute_end)
        temp_lucidity_period = temp_lucidity_period - absolute_begin
        # ===== Initial crop ======
        
        # ===== Bad segment remover ====== 
        if(bad_segments is not None):
            raw_1st = raw.copy().crop(0, bad_segments[0])
            raw_2nd = raw.copy().crop(bad_segments[1])
            raw = concatenate_raws([raw_1st, raw_2nd])
            bad_segment_size = bad_segments[1] - bad_segments[0]
            temp_lucidity_period[temp_lucidity_period > bad_segments[1]] -= bad_segment_size
        # ===== Bad segment remover ====== 
    
    # ====== Select only EEG =======
    raw_eogemgecg = raw.copy().filter(l_freq=1, h_freq=49, method='iir')
    # raw_eogemgecg.set_eeg_reference('average', projection=False, ch_type='eeg')
    
    selected_eogecgemgs = raw_eogemgecg.copy().pick_types(eog=True, emg=True, ecg=True).ch_names
    selected_eogecgemgs.append('Fp1')
    raw_eogemgecg.pick(selected_eogecgemgs)
    raw_eogemgecg.rename_channels({'Fp1': 'Fpp1'})
    
    # raw_eogemgecg.crop(absolute_begin, absolute_begin)
    raw.pick_types(eeg=True)
    # ====== Select only EEG =======
    
    # ======== Add EOG & EMG & ECG again ==========
    info = raw_eogemgecg.info
    eogemgecg_raw = mne.io.RawArray(raw_eogemgecg._data, info)
    # ======== Add EOG & EMG & ECG again ==========
    
    Fs = raw.info['sfreq']
    # ransac, line_noise = False, 50
    
    eog_indice = 0
    # ================ Initial preparation ========================
    
    # =============== PyPREP ====================
    if(ifpyprep == True):
        
        raw.filter(l_freq=initial_l_freq, h_freq=49, method='iir')
        
        montage = mne.channels.make_standard_montage("standard_1005") #this is chosen
    
        if(line_noise is None):
            line_noise = []
        else:
            line_noise = np.arange(line_noise, Fs / 2, line_noise)
        
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": line_noise,
            # "line_freqs": [],
            "max_iterations": 8
        }
        prep = PrepPipeline(raw, prep_params, montage, ransac=ransac, random_state=31)
        prep.fit()
        raw._data = prep.raw.get_data()
    # =============== PyPREP ====================
    
    # ===== High-pass filtering ======
    if(ifpyprep == True):
        if(initial_l_freq is None):
            raw.filter(l_freq=1, h_freq=None, method='iir')
        else:
            raw.filter(l_freq=None, h_freq=None, method='iir')
    else:
        raw.filter(l_freq=1, h_freq=49, method='iir')
    # ===== High-pass filtering ======
        
    # ========= Average re-reference =========
    if(avg_reference == True):
        raw.set_eeg_reference('average', projection=False, ch_type='eeg')
    # ========= Average re-reference =========
    
    # ================================ ASR ==========================================
    # ======= Get the calibration time-interval ==========
    window_length, step_length = int(Fs * 300), int(Fs * 300)
    time_interval_amount = int((raw._data.shape[1] - window_length) / step_length + 1)
    
    # =========== Calculate time ===========
    print('Total time: %0.2f seconds' % raw.times[-1])
    print('Estimated total time: %0.2f seconds' % (raw.times[-1] / (2 * 2.5)))
    print('Estimated total time: %0.2f minutes' % (raw.times[-1] / (2 * 2.5 * 60)))
    # =========== Calculate time ===========
    
    number_of_stds = np.zeros(time_interval_amount)
    for i in range(time_interval_amount):
        temp_std = 0
        for ii in range(raw._data.shape[0]): #num_of_channels
            temp_std += np.std(raw._data[ii][i * step_length: i * step_length + window_length])
        number_of_stds[i] = np.mean(temp_std)
    
    # selected_index = np.argmin(number_of_stds)
    selected_indices = np.argsort(number_of_stds)[:4]
    
    calibration_time_seconds_array = list()
    num_of_ch = len(raw.ch_names)
    train_X = np.empty(shape=[num_of_ch,0])
    for i in range(4):
        calibration_time_indx = np.arange(selected_indices[i] * step_length, selected_indices[i] * step_length + window_length)
        # calibration_time_seconds_array.append(calibration_time_indx / Fs)
        train_X = np.append(train_X, raw._data[:, calibration_time_indx], axis=1)
    # ======= Get the calibration time-interval ==========
    
    # ========== Train on a clean portion of data with rASR ========
    win_len = 2
    asr = ASR(method='euclid', estimator='scm', win_len=win_len, win_overlap=0.66, cutoff=5, sfreq=Fs) #2 seconds is roughly the length of the artifact
    asr.fit(train_X)
    # ========== Train on a clean portion of data with rASR ========
    
    # ========= Transform data windows =========
    # ===== 1st way =====
    length = win_len #seconds
    window, step = int(Fs*length), int(Fs*length)
    X = sliding_window(raw._data, window=window, step=step)
    Y = np.zeros_like(X)
    
    # ======= Time estimator =======
    start = time.time()
    asr.transform(X[:, 0, :])
    end = time.time()
    print('Estimated total run time for rASR is %.2f hours' % ((end-start) * X.shape[1] / 3600))
    
    for i in range(X.shape[1]):
        start = time.time()
        Y[:, i, :] = asr.transform(X[:, i, :])
        print(str(j) + '----' + str(i))
        end = time.time()
        print('%.2f seconds' % (end-start))
    
    if(len(raw._data[0]) % step > 0):
        last_tail_raw_pyprep_ssp = raw._data[:,-1 * (len(raw._data[0]) % step):]
        raw._data = np.append(Y.reshape(len(raw._data), -1), last_tail_raw_pyprep_ssp, axis=1)
    else:
        raw._data = Y.reshape(len(raw._data), -1)
    # ===== 1st way =====yprep_rasr._data), -1)
    # ========= Transform data windows =========
    
    # ====== Add channels =======
    # Create the new RawArray if necessary (with adjusted info)
    raw.add_channels([eogemgecg_raw])
    raw.drop_channels(['Fpp1'])
    # ====== Add channels =======
    # ================================ ASR ==========================================
    
    # raw_pyprep_ssp = chetto_EEG.SSP_artifact_removal(raw_pyprep)
    raw = chetto_EEG.SSP_artifact_removal(raw)
    raw_robustzscored = chetto_EEG.robustZScore(raw)
    
    if(temp_lucidity_period is not None):
        # ========== Catch the pre-eye signaling ===========
        raw_preeyesignal = raw.copy().crop(temp_lucidity_period[4,0], temp_lucidity_period[4,1])
        raw_robustzscored_preeyesignal = raw_robustzscored.copy().crop(temp_lucidity_period[4,0], temp_lucidity_period[4,1])
        # ========== Catch the pre-eye signaling ===========
        
        return raw, raw_robustzscored, raw_preeyesignal, raw_robustzscored_preeyesignal, temp_lucidity_period
    
    else:
        return raw, raw_robustzscored 
#%% =========== Micro-saccade finder algorithm ==============
'''
Usecase:
    
events, event_times = microsaccade_finder(raw=raw, eog_indice_l=61, eog_indice_r=62, i_o=4500,
                                          threshold_percentile=99.5, heog=None, method='slope')
'''

def microsaccade_finder(raw, i_o, eog_indice_l=None, eog_indice_r=None, heog=None, threshold_percentile=99.9, method='2ndorderdiff', hilbertt=True):
    
    raw_inspection = raw.copy()
    
    if(heog is not None):
        h_eog = raw_inspection._data[heog]
    else:
        eog_left = raw_inspection._data[eog_indice_l]
        eog_right = raw_inspection._data[eog_indice_r]
        h_eog = eog_left - eog_right
    
    Fs = raw_inspection.info['sfreq']
    
    # ===== Highcut adjustment based on Nyquist frequency =====
    if(Fs >= 200):
        highcut = 99
    elif(Fs == 100):
        highcut = 49
    else:
        highcut = 100
    # ===== Highcut adjustment based on Nyquist frequency =====
    
    h_eog_filtered = chetto_EEG.butter_bandpass_filter(h_eog, lowcut=30, highcut=highcut, fs=Fs,
                                                          order=6, filter_type='iir')
    h_eog_filtered2 = np.abs(hilbert(h_eog_filtered))
    
    # ======= Slope generation ========
    slope_array = np.zeros(len(h_eog))
    win_size = int(100 * (Fs / 500))
    step_amount = len(slope_array) - win_size + 1
    
    for i in range(step_amount):
        slope_array[i], _ = np.polyfit(x=np.arange(win_size),y=h_eog[i:i + win_size], deg=1)
        
    slope_array = np.abs(slope_array)
    slope_array /= max(slope_array)
    # ======= Slope generation ======== 
   
    # ======== 2nd order derivative =========
    h_eog_2ndorderdiff = np.abs(np.diff(a=h_eog, n=2))
    h_eog_2ndorderdiff /= max(h_eog_2ndorderdiff)
    # ======== 2nd derivative =========
    
    # ====== Avg of 2nd order der and slope ======
    avg = (slope_array[:-2] + h_eog_2ndorderdiff) / 2
    # ====== Avg of 2nd order der and slope ======
    
    if(method == '2ndorderdiff'):
         h_eog_filtered3 = h_eog_filtered2[:-2] * h_eog_2ndorderdiff
    elif(method == 'slope'):
         h_eog_filtered3 = h_eog_filtered2 * slope_array
    elif(method == 'avg'):
         h_eog_filtered3 = h_eog_filtered2[:-2] * avg   

    # ======== Event generation =======
    q3 = np.percentile(h_eog_filtered3, threshold_percentile)
    peaks_detected = find_peaks(h_eog_filtered3, height=q3, distance=120)[0] 
    # ======== Event generation =======

    event_times = peaks_detected / Fs
    events = np.zeros((len(peaks_detected), 3))
    events[:,0], events[:,2] = peaks_detected + i_o * Fs, np.ones(len(peaks_detected)) * 1
    events = events.astype(int)
    
    return events, event_times
