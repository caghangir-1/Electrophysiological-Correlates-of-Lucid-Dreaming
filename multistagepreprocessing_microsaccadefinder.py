'''
Custom EEG Preprocessing Pipeline for the EEG & MEG Recordings 
Project: "Electrophysiological Correlates of Lucid Dreaming: Sensor and Source-Level Signatures"
Designed and coded by: Çağatay Demirel
'''

import mne
import numpy as np
from pyprep.prep_pipeline import PrepPipeline
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
from scipy.signal import hilbert
from mne.io import concatenate_raws
import time
from scipy.signal import find_peaks, butter, iirfilter, filtfilt
#%% =============== Custom helper functions created for the actual functions ===============
def SSP_artifact_removal(raw):
    raw_copy = raw.copy()
    
    eog_indexes = mne.pick_types(raw_copy.info, eog=True)
    ecg_indexes = mne.pick_types(raw_copy.info, ecg=True)
    
    if(len(eog_indexes) > 0):
       eog_projs, _ = mne.preprocessing.compute_proj_eog(raw_copy, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True, verbose=False)
       raw_copy.add_proj(eog_projs)
    print('EOG Projs Created')
    if(len(ecg_indexes) > 0):
       ecg_projs, _ = mne.preprocessing.compute_proj_ecg(raw_copy, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True, verbose=False)
       raw_copy.add_proj(ecg_projs)
    print('ECG Projs Created')
        
    if(len(eog_indexes) > 0 or len(ecg_indexes) > 0):
        raw_copy.apply_proj()
    
    return raw_copy

def robustZScore(raw, ifnumpy=False):
    
    raw_robustzscore = raw.copy()
    
    if(ifnumpy == False):
        channel_amount = len(raw._data)
        for i in range(channel_amount):
            MAD = np.median(np.abs(raw._data[i] - np.median(raw._data[i])))
            raw_robustzscore._data[i] = 0.6745 * (raw._data[i] - np.median(raw._data[i])) / MAD
    else:
        channel_amount = len(raw)
        raw_robustzscore = raw.copy()
        for i in range(channel_amount):
            MAD = np.median(np.abs(raw[i] - np.median(raw[i])))
            raw_robustzscore[i] = 0.6745 * (raw[i] - np.median(raw[i])) / MAD
        
    return raw_robustzscore

def butter_bandpass(lowcut, highcut, fs, filter_type, order=3): # 3 ten sonra lfilter NaN degerler vermeye basliyor
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if(filter_type == 'butter'):
        b, a = butter(order, [low, high], btype='band', analog=False)
    elif(filter_type == 'iir'):
        b, a = iirfilter(order, [low, high], btype='bandpass', analog=False, ftype='butter')
    return b, a
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, filter_type='iir'):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, filter_type=filter_type)
#    y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y
#%% ============= Multi-stage preprocessing pipeline ================
'''

* This preprocessing pipeline combines already established methods to create an optimized workflow for artifact removal in EEG data :
1) PREP (early stage preprocessing, auto-detection & interpolation bad channels) : https://pyprep.readthedocs.io/en/latest/auto_examples/run_full_prep.html#sphx-glr-auto-examples-run-full-prep-py
2) ASR (Artifact Subspace Reconstruction) : https://nbara.github.io/python-meegkit/modules/meegkit.asr.html
3) SSP (Signal Space Projection) : https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html

* Manual for how to use and suggested parameter tunings:
1) Parameters:
- raw: MNE Raw object representing the EEG data.
- condition_segment_timings (optional): Condition segment timings that might be given as input in case you do not want to preprocess the whole recording but crop the relevant part includes your condition(s)
 * tail: parameter is a potential safeguard against edge artifacts that arise when filtering EEG data. EEG signals are typically processed using time-domain convolution filters (e.g., FIR, IIR), 
   which require surrounding data points for accurate estimation. Without a buffer zone around the segment of interest, filtering near the boundaries becomes unreliable, leading to signal distortions.
 * heavily suggested to make tail "true" in case you want to crop your long EEG recording.
 * entering segment timings also useful in case a computer's system specifications are limited and shorten the duration of the processing.
- bad_segments (optional): Time segments with corrupted data (flat signal, sweating artifacts e.g.) to exclude. 
- j: index to track which recording you are going through
- ransac: Boolean flag to enable/disable the RANSAC step in PyPREP. This parameter determines whether the Random Sample Consensus (RANSAC) algorithm is applied for automatic detection 
  and interpolation of bad EEG channels. This step enhances data quality by identifying channels with irregular or non-physiological activity. This parameter can only be used for EEG layout having
  higher than 19-channels (cannot set True for low-density EEGs e.g. 6-channel PSGs!)
- line_noise: Frequency of powerline noise to remove (e.g., 50 Hz for Europe, Asia, Africa, and Australia -- 60 Hz Standard in the United States, Canada).
- initial_l_freq: Initial high-pass filter cutoff frequency.
- ifpyprep: Boolean flag to enable/disable the initial PyPREP step. If the raw EEG has significant corruption, suggestion is to set it false because it can detect all the channels as "bad", 
            and cannot interpolate due to lack of clean reference channels.
- avg_reference: Boolean flag to apply average referencing. Suggested for mid-high density EEGs

2) Example suggested use-case:
    
    '
     * load raw as MNE object
     * set & organize channel layout of raw object
    '
    
 a1) For low-density EEG (in case having a heavy noise profile):
    
    raw_preprocessed, raw_robustzscored, raw_preeyesignal, raw_robustzscored_preeyesignal, temp_lucidity_period = whole_preprocessing(raw=raw, condition_segment_timings=None, 
                                                                                                                  bad_segments=None, j=j, ransac=False, line_noise=50, initial_l_freq=None, ifpyprep=False, 
                                                                                                                  avg_reference=False)
 a2) For low-density EEG (in case not having a heavy noise profile):
   
   raw_preprocessed, raw_robustzscored, raw_preeyesignal, raw_robustzscored_preeyesignal, temp_lucidity_period = whole_preprocessing(raw=raw, condition_segment_timings=None, 
                                                                                                                 bad_segments=None, j=j, ransac=False, line_noise=50, initial_l_freq=None, ifpyprep=True, 
                                                                                                                 avg_reference=False)
    
 b) For mid-low-density EEG

    raw_preprocessed, raw_robustzscored, raw_preeyesignal, raw_robustzscored_preeyesignal, temp_lucidity_period = whole_preprocessing(raw=raw, condition_segment_timings=None, 
                                                                                                                  bad_segments=None, j=j, ransac=False, line_noise=50, initial_l_freq=None, ifpyprep=True, 
                                                                                                                  avg_reference=True)
    
 c) For high-density EEG & MEG (ransac parameter is highly effective on very-high density layouts)

   raw_preprocessed, raw_robustzscored, raw_preeyesignal, raw_robustzscored_preeyesignal, temp_lucidity_period = whole_preprocessing(raw=raw, condition_segment_timings=None, 
                                                                                                                 bad_segments=None, j=j, ransac=True, line_noise=50, initial_l_freq=None, ifpyprep=True, 
                                                                                                                 avg_reference=True)
    
3) Warning:
* The recording data should be long enough (approx >1 hour) especially for ASR step given that the our algorithm also finds clean segments as a training data of Euclidean ASR model.
* (Very) short mid-high density recordings e.g. up to 30 minutes might perform better with independent component analysis (ICA) compared to ASR. 
* SSP requires at least one EOG or ECG channel, and the final processing step will fail if neither is available.   
* Riemann ASR (rASR) was also tested but introduced an artificial noise profile instead of achieving effective attenuation, leading us to adopt Euclidean ASR instead.
'''

def whole_preprocessing(raw, condition_segment_timings=None, bad_segments=None, j=0, ransac=False, line_noise=50, initial_l_freq=None, ifpyprep=True, avg_reference=False,
                        tail=True):

    if(condition_segment_timings is not None):
        
        # ===== Initial crop ======
        if(tail == True):
            absolute_begin, absolute_end = np.min(condition_segment_timings) - 10, np.max(condition_segment_timings) + 10
        else:
            absolute_begin, absolute_end = np.min(condition_segment_timings), np.max(condition_segment_timings)
        raw.crop(absolute_begin, absolute_end)
        condition_segment_timings = condition_segment_timings - absolute_begin
        # ===== Initial crop ======
        
        # ===== Bad segment remover ====== 
        if(bad_segments is not None):
            raw_1st = raw.copy().crop(0, bad_segments[0])
            raw_2nd = raw.copy().crop(bad_segments[1])
            raw = concatenate_raws([raw_1st, raw_2nd])
            bad_segment_size = bad_segments[1] - bad_segments[0]
            condition_segment_timings[condition_segment_timings > bad_segments[1]] -= bad_segment_size
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
    
    ''' =============== PyPREP ==================== '''
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
    ''' =============== PyPREP ==================== '''
    
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
    
    ''' ================================ ASR ========================================== '''
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
    asr = ASR(method='euclid', estimator='scm', win_len=win_len, win_overlap=0.66, cutoff=5, sfreq=Fs) #2 seconds found be to an approximately optimal length
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
    ''' ================================ ASR ========================================== '''
    
    ''' ================================ SSP ========================================== '''
    raw = SSP_artifact_removal(raw)
    ''' ================================ SSP ========================================== '''

    raw_robustzscored = robustZScore(raw)
    
    return raw, raw_robustzscored 
#%% =========== Micro-saccade finder algorithm ==============
'''
This is micro & miniature saccade finder algorithm developed to guide the ones aiming to detect potential saccades.

* Steps of the algorithm:
1) As a preparation step, make sure that at least one potential EOG channel exist or in the worst case scenario without having EOG channel, use one of the fronto-lateral EEG channels (for instance: F7, F8, Fp1, Fp2 etc.)
2) Extraction of re-referenced EOGs --> HEOG
3) Band-pass filtering between 30 Hz till close to Nyquist range.
4) Hilbert transform to the filtered horizontal EOG (hEOG) signal, which converts it into an analytic signal. By taking the absolute value of the analytic signal, the envelope of the EOG signal is extracted. 
   This envelope represents the instantaneous amplitude of the eye movement activity over time.
5) Next step is processing the Hilbert envelope with one of 3 different scenarios (slope generation, taking the 2nd order derivative or combination of slope and 2nd order derivative)
6) Dynamic threshold based on the given percentile (e.g. 95%, 99%, 99.9%) of the processed EOG Hilbert envelope by calculating the percentile of the signal amplitude, and apply peak finder algorithm from Scipy
   to detect potential saccadic activities reflecting as "peak" the processed envelope.
7) Refine the absolute timing of the determined saccades by adding the initial buffer time (helps for event, epoch, evoked generation for further analyses)

* Manual for how to use and suggested parameter tunings: The intended function heavily relies on the input HEOG specifications and parameters are quite non-trivial that could be tuned for specific EOG series.
1) Parameters:
- raw: The raw EEG/EOG recording from MNE, including at least potential EOG channel(s)
- i_o: Initial absolute onset time (in seconds) from which the raw data is analyzed.
- eog_indice_l (optional): The index of the left EOG channel within the raw data array.
- eog_indice_r (optional): The index of the right EOG channel within the raw data array.
- heog (optional): The index of the horizontal EOG (HEOG) channel in the raw data. If you already have horizontally referenced EOG in your raw data, the algorithm will ignore the individual EOGs
- threshold_percentile: Percentile threshold to detect significant peaks in the filtered EOG envelope. Higher values (e.g., 99.9) result in fewer but more prominent detections.

- method: Used to further process Hilbert envelopes to refine & de-noise for preparation for peak detection
 * '2ndorderdiff': Based on the second derivative of the signal.
 * 'slope' (default): Uses the linear slope of the EOG signal ---> works better compared to others
 * 'avg': Averages both slope and second derivative.

- distance: Numerical value represent the smallest possible points between 2 potential saccades (very exploratory parameter depending on the intensity of saccades and sampling rate of the EOG signal)

2) Example suggested use-case:

'
 * load raw as MNE object
 * set & organize channel layout of raw object
'

a) For having already horizontally referenced EOGs:
    
   events, event_times = microsaccade_finder(raw=raw, heog=64, i_o=4500, threshold_percentile=99.5, method='slope')

b) For having invidiual left and right EOGs and re-referencing step hasn't been taken:
    
   events, event_times = microsaccade_finder(raw=raw, eog_indice_l=64, eog_indice_r=65, i_o=4500, threshold_percentile=99.5, method='slope')
 
3) Warning: 
* The algorithm may detect a few false-positive saccades and mostly useful for large sequences to detect potential saccades. False detection is heavily relies on EOG placement and eye movement intensity.
This is why we further made a visual inspection to remove wrongly labeled saccades in the project. 
* Suggested range of threshold_percentile is in between 95 - 99.9% and distance=120 for sampling rate of 100 Hz (based on trials). However, parameter tuning is suggested.
'''

def microsaccade_finder(raw, i_o, eog_indice_l=None, eog_indice_r=None, heog=None, threshold_percentile=99.9, method='slope', distance=120):
    
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
    
    h_eog_filtered = butter_bandpass_filter(h_eog, lowcut=30, highcut=highcut, fs=Fs,
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
    peaks_detected = find_peaks(h_eog_filtered3, height=q3, distance=distance)[0] 
    # ======== Event generation =======

    event_times = peaks_detected / Fs
    events = np.zeros((len(peaks_detected), 3))
    events[:,0], events[:,2] = peaks_detected + i_o * Fs, np.ones(len(peaks_detected)) * 1
    events = events.astype(int)
    
    return events, event_times
