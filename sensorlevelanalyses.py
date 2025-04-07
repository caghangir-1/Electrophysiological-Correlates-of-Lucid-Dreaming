'''
===============================================
Sensor-Level Analysis Code Summary

Project: "Electrophysiological Correlates of Lucid Dreaming: Sensor and Source-Level Signatures"
Designed and coded by: Çağatay Demirel
===============================================

This script provides a streamlined summary of the sensor-level analyses 
conducted for this study, focusing on the core processing. It encompasses the following key procedures:

1. Power Spectral Density (PSD) Calculation:
   - Multitaper PSD estimation across conditions (Early REM, Later REM, Lucid Dreaming, Wake).
   - Log-transformation of PSD values for further statistical analyses.

2. Complexity and Entropy Metrics:
   - Computation of Lempel-Ziv Complexity (LZC), Permutation Entropy, Approximate Entropy, 
     Sample Entropy, and Higuchi Fractal Dimension for each condition.
   - Standardized parameters applied across all conditions for consistency.

3. Topographical Analysis:
   - Grand average current source density (CSD) transformation for enhanced spatial resolution.
   - Application of CSD filters to individual epochs using a consistent stiffness parameter.

4. Normalization and Within-Subject Comparisons:
   - Within-subject normalization of power across conditions using decibel (dB) scaling.
   - Comparisons include Later REM vs. Early REM, Lucid Dreaming vs. REM, and Lucid Dreaming vs. Wake.

'''

import numpy as np

import mne
import math
import entropy as ent
from mne.channels import find_ch_adjacency
from mne.preprocessing import compute_current_source_density
import os
from scipy.signal import hilbert

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#%% =========== Some preparatory functions ==============
def lempel_ziv_complexity(raw, approach='hilbert'):
    """Lempel-Ziv complexity for a binary sequence """
    
    binary_sequence = np.zeros(len(raw))
    if(approach == 'median'):
        raw_median = np.median(raw)
        binary_sequence[raw > raw_median] = 1
    elif(approach == 'hilbert'):
        raw_h = np.abs(hilbert(raw))
        threshold = np.mean(raw_h)
        binary_sequence[raw_h > threshold] = 1
    elif(approach == 'onethreestd'):
        threshold = np.mean(raw) + np.std(raw) * 1.3
        binary_sequence[raw > threshold] = 1
    
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
                
    complexity = complexity * math.log(complexity,3) / len(binary_sequence)
    
    # complexity = LZC(binary_sequence)
    # complexity = complexity * math.log(complexity,3) / len(binary_sequence)

    return complexity

def freqband_power_split_and_conditioncontrast(power_normed_list, normalization='maxdivision', chan_size=59):
    ''' the reason of naming as power normed list is because the given power data has multiple 
    trials for each recording '''
    
    freq_arr = np.array([[2,4],[4,8],[8,12],[12,30],[30,36],[36,45]])
    whole_data_for_all_freqbands = list()
    subject_size = len(power_normed_list)
    for j in range(len(freq_arr)):
        
        whole_data = np.empty(shape=[0,chan_size])
        whole_data_subject_avg = np.zeros((subject_size,chan_size))
        for i in range(subject_size):
            freqs_min, freqs_max = np.min(np.argwhere(freqs >= freq_arr[j,0])), np.max(np.argwhere(freqs < freq_arr[j,1]))
        
            temp_data = power_normed_list[i]
            temp_data = np.mean(temp_data[:,freqs_min:freqs_max], axis=1)
                        
            if(normalization == 'zscore'):
                temp_data = (temp_data - np.mean(temp_data)) / np.std(temp_data)
            elif(normalization == 'maxdivision'):
                temp_data = temp_data / np.max(np.abs(temp_data))
            else:
                print('no normalization')
    
            whole_data_subject_avg[i] = temp_data
               
        whole_data_for_all_freqbands.append(whole_data_subject_avg)
        
    return whole_data_for_all_freqbands
#%% ========= Multitaper PSD list for confidence band ============
epochs = 'load'

psds_earlyrem = np.empty(shape=[185])
psds_laterrem = np.empty(shape=[185])
psds_lucid = np.empty(shape=[185])
psds_wake = np.empty(shape=[185])
fmin, fmax = 2, 48

for i in range(44):
    spectrum = epochs[i]['Early REM'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    temp_psds_earlyrem, freqs = spectrum.get_data(return_freqs=True)
    temp_psds_earlyrem = np.mean(temp_psds_earlyrem, axis=(0,1))
    psds_earlyrem = np.row_stack((psds_earlyrem, temp_psds_earlyrem))
    
    spectrum = epochs[i]['Later REM'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    temp_psds_laterrem, freqs = spectrum.get_data(return_freqs=True)
    temp_psds_laterrem = np.mean(temp_psds_laterrem, axis=(0,1))
    psds_laterrem = np.row_stack((psds_laterrem, temp_psds_laterrem))
    
    spectrum = epochs[i]['Lucid Dreaming'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    temp_psds_lucid, freqs = spectrum.get_data(return_freqs=True)
    temp_psds_lucid = np.mean(temp_psds_lucid, axis=(0,1))
    psds_lucid = np.row_stack((psds_lucid, temp_psds_lucid))
    
    spectrum = epochs[i]['Wake'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    temp_psds_wake, freqs = spectrum.get_data(return_freqs=True)
    temp_psds_wake = np.mean(temp_psds_wake, axis=(0,1))
    psds_wake = np.row_stack((psds_wake, temp_psds_wake))

psds_earlyrem_nologtransform = psds_earlyrem.copy()
psds_laterrem_nologtransform = psds_laterrem.copy()
psds_lucid_nologtransform = psds_lucid.copy()
psds_wake_nologtransform = psds_wake.copy()

psds_earlyrem = 10. * np.log10(psds_earlyrem[1:])
psds_laterrem = 10. * np.log10(psds_laterrem[1:])
psds_lucid = 10. * np.log10(psds_lucid[1:])
psds_wake = 10. * np.log10(psds_wake[1:])
#%% ======================= Entropy calculation ==========================
# import pyeeg
import neurokit2 as nk

# ===== Pre-defined values ======
fmin, fmax = 2, 48
# ===== Pre-defined values ======

# ====== Initialize parameters =======
perm_ent_order = 6
apEn_order = 6
SampEn_order = 2
R = 0
for i in range(len(epochs)):
    R += np.std(epochs[i]._data)
R /= len(epochs)
R *= 0.2
# ====== Initialize parameters =======

# ====== Early REM ========
avg_entropy_all_early_rem = np.zeros(len(epochs))
avg_spectentropy_all_early_rem = np.zeros(len(epochs))
avg_appentropy_all_early_rem = np.zeros(len(epochs))
avg_sampleentropy_all_early_rem = np.zeros(len(epochs))
avg_hhtenropy_all_early_rem = np.zeros(len(epochs))
avg_fooof_all_early_rem = np.zeros(len(epochs))
avg_fooof_r2_all_early_rem = np.zeros(len(epochs))

avg_LZCs_onethreshold_early_rem = np.zeros(len(epochs))
avg_LZCs_hilbert_early_rem = np.zeros(len(epochs))
avg_LZCs_median_early_rem = np.zeros(len(epochs))

avg_higuchifractal_early_rem = np.zeros(len(epochs))

for k in range(len(epochs)):
    
    print(k)
    
    temp_epochs = epochs[k]
    avg_entropy_per_subject = np.zeros(len(temp_epochs['Early REM']))
    avg_spectentropy_per_subject = np.zeros(len(temp_epochs['Early REM']))
    avg_appentropy_per_subject = np.zeros(len(temp_epochs['Early REM']))
    avg_sampleentropy_per_subject = np.zeros(len(temp_epochs['Early REM']))
    avg_hhtentropy_per_subject = np.zeros(len(temp_epochs['Early REM']))
    avg_fooof_per_subject = np.zeros(len(temp_epochs['Early REM']))
    avg_fooof_r2_per_subject = np.zeros(len(temp_epochs['Early REM']))
    
    avg_higuchifractal_per_subject = np.zeros(len(temp_epochs['Early REM']))
    
    avg_LZCs_onethreshold_per_subject = np.zeros(len(temp_epochs['Early REM']))
    avg_LZCs_hilbert_per_subject = np.zeros(len(temp_epochs['Early REM']))
    avg_LZCs_median_per_subject = np.zeros(len(temp_epochs['Early REM']))
    
    spectrum = temp_epochs['Early REM'].compute_psd(method="multitaper", fmin=2, fmax=48)
    temp_psds, freqs = spectrum.get_data(return_freqs=True)
    
    for i in range(len(temp_epochs['Early REM'])): #epochs per-subject
        temp_psds_epoch = temp_psds[i]
        temp_data = temp_epochs['Early REM']._data[i]
        temp_entropies = np.zeros(6)
        temp_spectentropies = np.zeros(6)
        temp_appentropies = np.zeros(6)
        temp_sampleentropies = np.zeros(6)
        temp_hht_entropies = np.zeros(6)
        
        temp_higuchifractal = np.zeros(6) #continue from here!
        
        temp_LZCs_onethreshold = np.zeros(6)
        temp_LZCs_hilbert = np.zeros(6)
        temp_LZCs_median = np.zeros(6)
        
        for j in range(6): #6 channels
            # ======= Entropy Analysis =======
            temp_entropies[j] = ent.perm_entropy(temp_data[j], order=perm_ent_order, delay=2, normalize=True)
            # temp_spectentropies[j] = ent.spectral_entropy(temp_data[j], sf=100, method='welch', normalize=True)
            
            temp_appentropies[j] = ent.app_entropy(temp_data[j], order=apEn_order, metric='euclidean')
            temp_sampleentropies[j] = ent.sample_entropy(temp_data[j], order=2)
            
            # hht_psd = hilbert_huang_transform(temp_data[j], sample_rate=100, fmin=fmin, fmax=fmax)
            # temp_hht_entropies[j] = ent.perm_entropy(hht_psd, order=6, delay=2, normalize=True)
            # ======= Entropy Analysis =======
            
            # ======= LZCs ========
            temp_LZCs_onethreshold[j] = lempel_ziv_complexity(temp_data[j], approach='onethreestd')
            temp_higuchifractal[j], _ = nk.fractal_higuchi(temp_data[j], k_max='default', show=False)
            
            print('channel no %d' % j)
            # temp_LZCs_hilbert[j] = chetto_EEG.lempel_ziv_complexity(temp_data[j], approach='hilbert')
            # temp_LZCs_median[j] = chetto_EEG.lempel_ziv_complexity(temp_data[j], approach='median')
            # ======= LZCs ========
        
        avg_entropy_per_subject[i] = np.mean(temp_entropies)
        avg_spectentropy_per_subject[i] = np.mean(temp_spectentropies)
        avg_appentropy_per_subject[i] = np.mean(temp_appentropies)
        avg_sampleentropy_per_subject[i] = np.mean(temp_sampleentropies)
        avg_hhtentropy_per_subject[i] = np.mean(temp_hht_entropies)
        
        avg_LZCs_onethreshold_per_subject[i] = np.mean(temp_LZCs_onethreshold)
        avg_LZCs_hilbert_per_subject[i] = np.mean(temp_LZCs_hilbert)
        avg_LZCs_median_per_subject[i] = np.mean(temp_LZCs_median)
        
        avg_higuchifractal_per_subject[i] = np.mean(temp_higuchifractal)
        
    avg_entropy_all_early_rem[k] = np.mean(avg_entropy_per_subject) 
    avg_spectentropy_all_early_rem[k] = np.mean(avg_spectentropy_per_subject) 
    avg_appentropy_all_early_rem[k] = np.mean(avg_appentropy_per_subject)
    avg_sampleentropy_all_early_rem[k] = np.mean(avg_sampleentropy_per_subject)
    avg_hhtenropy_all_early_rem[k] = np.mean(avg_hhtentropy_per_subject)
    avg_fooof_all_early_rem[k], avg_fooof_r2_all_early_rem = np.mean(avg_fooof_per_subject), np.mean(avg_fooof_r2_per_subject)
    
    avg_LZCs_onethreshold_early_rem[k] = np.mean(avg_LZCs_onethreshold_per_subject)
    avg_LZCs_hilbert_early_rem[k] = np.mean(avg_LZCs_hilbert_per_subject)
    avg_LZCs_median_early_rem[k] = np.mean(avg_LZCs_median_per_subject)
    
    avg_higuchifractal_early_rem[k] = np.mean(avg_higuchifractal_per_subject)
    
# ====== Early REM ========

# ====== Later REM ========
avg_entropy_all_later_rem = np.zeros(len(epochs))
avg_spectentropy_all_later_rem = np.zeros(len(epochs))
avg_appentropy_all_later_rem = np.zeros(len(epochs))
avg_sampleentropy_all_later_rem = np.zeros(len(epochs))
avg_hhtenropy_all_later_rem = np.zeros(len(epochs))
avg_fooof_all_later_rem = np.zeros(len(epochs))
avg_fooof_r2_all_later_rem = np.zeros(len(epochs))

avg_LZCs_onethreshold_later_rem = np.zeros(len(epochs))
avg_LZCs_hilbert_later_rem = np.zeros(len(epochs))
avg_LZCs_median_later_rem = np.zeros(len(epochs))

avg_higuchifractal_later_rem = np.zeros(len(epochs))

for k in range(len(epochs)):
    
    print(k)
    
    temp_epochs = epochs[k]
    avg_entropy_per_subject = np.zeros(len(temp_epochs['Later REM']))
    avg_spectentropy_per_subject = np.zeros(len(temp_epochs['Later REM']))
    avg_appentropy_per_subject = np.zeros(len(temp_epochs['Later REM']))
    avg_sampleentropy_per_subject = np.zeros(len(temp_epochs['Later REM']))
    avg_hhtentropy_per_subject = np.zeros(len(temp_epochs['Later REM']))
    avg_fooof_per_subject = np.zeros(len(temp_epochs['Later REM']))
    avg_fooof_r2_per_subject = np.zeros(len(temp_epochs['Later REM']))
    
    avg_higuchifractal_per_subject = np.zeros(len(temp_epochs['Later REM']))
    
    avg_LZCs_onethreshold_per_subject = np.zeros(len(temp_epochs['Later REM']))
    avg_LZCs_hilbert_per_subject = np.zeros(len(temp_epochs['Later REM']))
    avg_LZCs_median_per_subject = np.zeros(len(temp_epochs['Later REM']))
    
    spectrum = temp_epochs['Later REM'].compute_psd(method="multitaper", fmin=2, fmax=48)
    temp_psds, freqs = spectrum.get_data(return_freqs=True)
    
    for i in range(len(temp_epochs['Later REM'])): #epochs per-subject
        temp_psds_epoch = temp_psds[i]
        temp_data = temp_epochs['Later REM']._data[i]
        temp_entropies = np.zeros(6)
        temp_spectentropies = np.zeros(6)
        temp_appentropies = np.zeros(6)
        temp_sampleentropies = np.zeros(6)
        temp_hht_entropies = np.zeros(6)
        
        temp_LZCs_onethreshold = np.zeros(6)
        temp_LZCs_hilbert = np.zeros(6)
        temp_LZCs_median = np.zeros(6)
        
        for j in range(6): #6 channels
            # ======= Entropy Analysis =======
            temp_entropies[j] = ent.perm_entropy(temp_data[j], order=perm_ent_order, delay=2, normalize=True)
            # temp_spectentropies[j] = ent.spectral_entropy(temp_data[j], sf=100, method='welch', normalize=True)
            
            temp_appentropies[j] = ent.app_entropy(temp_data[j], order=apEn_order, metric='euclidean')
            temp_sampleentropies[j] = ent.sample_entropy(temp_data[j], order=2)

            # hht_psd = hilbert_huang_transform(temp_data[j], sample_rate=100, fmin=fmin, fmax=fmax)
            # temp_hht_entropies[j] = ent.perm_entropy(hht_psd, order=6, delay=2, normalize=True)
            # ======= Entropy Analysis =======
            
            # ======= LZCs ========
            temp_LZCs_onethreshold[j] = lempel_ziv_complexity(temp_data[j], approach='onethreestd')
            # temp_LZCs_hilbert[j] = lempel_ziv_complexity(temp_data[j], approach='hilbert')
            # temp_LZCs_median[j] = lempel_ziv_complexity(temp_data[j], approach='median')
            temp_higuchifractal[j], _ = nk.fractal_higuchi(temp_data[j], k_max='default', show=False)
            
            print('channel no %d' % j)
            # ======= LZCs ========
        
        avg_entropy_per_subject[i] = np.mean(temp_entropies)
        avg_spectentropy_per_subject[i] = np.mean(temp_spectentropies)
        avg_appentropy_per_subject[i] = np.mean(temp_appentropies)
        avg_sampleentropy_per_subject[i] = np.mean(temp_sampleentropies)
        avg_hhtentropy_per_subject[i] = np.mean(temp_hht_entropies)
        
        avg_LZCs_onethreshold_per_subject[i] = np.mean(temp_LZCs_onethreshold)
        avg_LZCs_hilbert_per_subject[i] = np.mean(temp_LZCs_hilbert)
        avg_LZCs_median_per_subject[i] = np.mean(temp_LZCs_median)
        
        avg_higuchifractal_per_subject[i] = np.mean(temp_higuchifractal)
        
    avg_entropy_all_later_rem[k] = np.mean(avg_entropy_per_subject) 
    avg_spectentropy_all_later_rem[k] = np.mean(avg_spectentropy_per_subject) 
    avg_appentropy_all_later_rem[k] = np.mean(avg_appentropy_per_subject)
    avg_sampleentropy_all_later_rem[k] = np.mean(avg_sampleentropy_per_subject)
    avg_hhtenropy_all_later_rem[k] = np.mean(avg_hhtentropy_per_subject)
    avg_fooof_all_later_rem[k] = np.mean(avg_fooof_per_subject)
    avg_fooof_r2_all_later_rem[k] = np.mean(avg_fooof_r2_per_subject)
    
    avg_LZCs_onethreshold_later_rem[k] = np.mean(avg_LZCs_onethreshold_per_subject)
    avg_LZCs_hilbert_later_rem[k] = np.mean(avg_LZCs_hilbert_per_subject)
    avg_LZCs_median_later_rem[k] = np.mean(avg_LZCs_median_per_subject)
    
    avg_higuchifractal_later_rem[k] = np.mean(avg_higuchifractal_per_subject)
# ====== Later REM ========

# ====== Lucid ========
avg_entropy_all_lucid = np.zeros(len(epochs))
avg_spectentropy_all_lucid = np.zeros(len(epochs))
avg_appentropy_all_lucid = np.zeros(len(epochs))
avg_sampleentropy_all_lucid = np.zeros(len(epochs))
avg_hhtenropy_all_lucid = np.zeros(len(epochs))
avg_fooof_all_lucid = np.zeros(len(epochs))
avg_fooof_r2_all_lucid = np.zeros(len(epochs))

avg_LZCs_onethreshold_lucid = np.zeros(len(epochs))
avg_LZCs_hilbert_lucid = np.zeros(len(epochs))
avg_LZCs_median_lucid = np.zeros(len(epochs))

avg_higuchifractal_lucid = np.zeros(len(epochs))

for k in range(len(epochs)):
    
    print(k)
    
    temp_epochs = epochs[k]
    avg_entropy_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    avg_spectentropy_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    avg_appentropy_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    avg_sampleentropy_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    avg_hhtentropy_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    avg_fooof_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    avg_fooof_r2_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    
    avg_LZCs_onethreshold_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    avg_LZCs_hilbert_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    avg_LZCs_median_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    
    avg_higuchifractal_per_subject = np.zeros(len(temp_epochs['Lucid Dreaming']))
    
    spectrum = temp_epochs['Lucid Dreaming'].compute_psd(method="multitaper", fmin=2, fmax=48)
    temp_psds, freqs = spectrum.get_data(return_freqs=True)
    
    for i in range(len(temp_epochs['Lucid Dreaming'])): #epochs per-subject
        temp_psds_epoch = temp_psds[i]
        temp_data = temp_epochs['Lucid Dreaming']._data[i]
        temp_entropies = np.zeros(6)
        temp_spectentropies = np.zeros(6)
        temp_appentropies = np.zeros(6)
        temp_sampleentropies = np.zeros(6)
        temp_hht_entropies = np.zeros(6)
        
        temp_LZCs_onethreshold = np.zeros(6)
        temp_LZCs_hilbert = np.zeros(6)
        temp_LZCs_median = np.zeros(6)
        
        for j in range(6): #6 channels
            # ======= Entropy Analysis =======
            temp_entropies[j] = ent.perm_entropy(temp_data[j], order=perm_ent_order, delay=2, normalize=True)
            # temp_spectentropies[j] = ent.spectral_entropy(temp_data[j], sf=100, method='welch', normalize=True)
            
            temp_appentropies[j] = ent.app_entropy(temp_data[j], order=apEn_order, metric='euclidean')
            temp_sampleentropies[j] = ent.sample_entropy(temp_data[j], order=2)

            
            # hht_psd = hilbert_huang_transform(temp_data[j], sample_rate=100, fmin=fmin, fmax=fmax)
            # temp_hht_entropies[j] = ent.perm_entropy(hht_psd, order=6, delay=2, normalize=True)
            # ======= Entropy Analysis =======
            
            # ======= LZCs ========
            temp_LZCs_onethreshold[j] = lempel_ziv_complexity(temp_data[j], approach='onethreestd')
            # temp_LZCs_hilbert[j] = lempel_ziv_complexity(temp_data[j], approach='hilbert')
            # temp_LZCs_median[j] = lempel_ziv_complexity(temp_data[j], approach='median')
            temp_higuchifractal[j], _ = nk.fractal_higuchi(temp_data[j], k_max='default', show=False)
            
            print('channel no %d' % j)
            # ======= LZCs ========
        
        avg_entropy_per_subject[i] = np.mean(temp_entropies)
        avg_spectentropy_per_subject[i] = np.mean(temp_spectentropies)
        avg_appentropy_per_subject[i] = np.mean(temp_appentropies)
        avg_sampleentropy_per_subject[i] = np.mean(temp_sampleentropies)
        avg_hhtentropy_per_subject[i] = np.mean(temp_hht_entropies)
        
        avg_LZCs_onethreshold_per_subject[i] = np.mean(temp_LZCs_onethreshold)
        avg_LZCs_hilbert_per_subject[i] = np.mean(temp_LZCs_hilbert)
        avg_LZCs_median_per_subject[i] = np.mean(temp_LZCs_median)
        
        avg_higuchifractal_per_subject[i] = np.mean(temp_higuchifractal)
        
    avg_entropy_all_lucid[k] = np.mean(avg_entropy_per_subject) 
    avg_spectentropy_all_lucid[k] = np.mean(avg_spectentropy_per_subject) 
    avg_appentropy_all_lucid[k] = np.mean(avg_appentropy_per_subject)
    avg_sampleentropy_all_lucid[k] = np.mean(avg_sampleentropy_per_subject)
    avg_hhtenropy_all_lucid[k] = np.mean(avg_hhtentropy_per_subject)
    avg_fooof_all_lucid[k] = np.mean(avg_fooof_per_subject)
    avg_fooof_r2_all_lucid[k] = np.mean(avg_fooof_r2_per_subject)
    
    avg_LZCs_onethreshold_lucid[k] = np.mean(avg_LZCs_onethreshold_per_subject)
    avg_LZCs_hilbert_lucid[k] = np.mean(avg_LZCs_hilbert_per_subject)
    avg_LZCs_median_lucid[k] = np.mean(avg_LZCs_median_per_subject)
    
    avg_higuchifractal_lucid[k] = np.mean(avg_higuchifractal_per_subject)
# ====== Lucid ========

# ====== Wake ========
avg_entropy_all_wake = np.zeros(len(epochs))
avg_spectentropy_all_wake = np.zeros(len(epochs))
avg_appentropy_all_wake = np.zeros(len(epochs))
avg_sampleentropy_all_wake = np.zeros(len(epochs))
avg_hhtenropy_all_wake = np.zeros(len(epochs))
avg_fooof_all_wake = np.zeros(len(epochs))
avg_fooof_r2_all_wake = np.zeros(len(epochs))

avg_LZCs_onethreshold_wake = np.zeros(len(epochs))
avg_LZCs_hilbert_wake = np.zeros(len(epochs))
avg_LZCs_median_wake = np.zeros(len(epochs))

avg_higuchifractal_wake = np.zeros(len(epochs))

for k in range(len(epochs)):
    
    print(k)
    
    temp_epochs = epochs[k]
    avg_entropy_per_subject = np.zeros(len(temp_epochs['Wake']))
    avg_spectentropy_per_subject = np.zeros(len(temp_epochs['Wake']))
    avg_appentropy_per_subject = np.zeros(len(temp_epochs['Wake']))
    avg_sampleentropy_per_subject = np.zeros(len(temp_epochs['Wake']))
    avg_hhtentropy_per_subject = np.zeros(len(temp_epochs['Wake']))
    avg_fooof_per_subject = np.zeros(len(temp_epochs['Wake']))
    avg_fooof_r2_per_subject = np.zeros(len(temp_epochs['Wake']))
    
    avg_LZCs_onethreshold_per_subject = np.zeros(len(temp_epochs['Wake']))
    avg_LZCs_hilbert_per_subject = np.zeros(len(temp_epochs['Wake']))
    avg_LZCs_median_per_subject = np.zeros(len(temp_epochs['Wake']))
    
    avg_higuchifractal_per_subject = np.zeros(len(temp_epochs['Wake']))
    
    spectrum = temp_epochs['Wake'].compute_psd(method="multitaper", fmin=2, fmax=48)
    temp_psds, freqs = spectrum.get_data(return_freqs=True)
    
    for i in range(len(temp_epochs['Wake'])): #epochs per-subject
        temp_psds_epoch = temp_psds[i]
        temp_data = temp_epochs['Wake']._data[i]
        temp_entropies = np.zeros(6)
        temp_spectentropies = np.zeros(6)
        temp_appentropies = np.zeros(6)
        temp_sampleentropies = np.zeros(6)
        temp_hht_entropies = np.zeros(6)
        
        temp_LZCs_onethreshold = np.zeros(6)
        temp_LZCs_hilbert = np.zeros(6)
        temp_LZCs_median = np.zeros(6)
        
        for j in range(6): #6 channels
            # ======= Entropy Analysis =======
            temp_entropies[j] = ent.perm_entropy(temp_data[j], order=perm_ent_order, delay=2, normalize=True)
            # temp_spectentropies[j] = ent.spectral_entropy(temp_data[j], sf=100, method='welch', normalize=True)
            
            temp_appentropies[j] = ent.app_entropy(temp_data[j], order=apEn_order, metric='euclidean')
            temp_sampleentropies[j] = ent.sample_entropy(temp_data[j], order=2)
            
            # hht_psd = hilbert_huang_transform(temp_data[j], sample_rate=100, fmin=fmin, fmax=fmax)
            # temp_hht_entropies[j] = ent.perm_entropy(hht_psd, order=6, delay=2, normalize=True)
            # ======= Entropy Analysis =======
            
            # ======= LZCs ========
            temp_LZCs_onethreshold[j] = lempel_ziv_complexity(temp_data[j], approach='onethreestd')
            # temp_LZCs_hilbert[j] = lempel_ziv_complexity(temp_data[j], approach='hilbert')
            # temp_LZCs_median[j] = lempel_ziv_complexity(temp_data[j], approach='median')
            temp_higuchifractal[j], _ = nk.fractal_higuchi(temp_data[j], k_max='default', show=False)
            
            print('channel no %d' % j)
            # ======= LZCs ========
        
        avg_entropy_per_subject[i] = np.mean(temp_entropies)
        avg_spectentropy_per_subject[i] = np.mean(temp_spectentropies)
        avg_appentropy_per_subject[i] = np.mean(temp_appentropies)
        avg_sampleentropy_per_subject[i] = np.mean(temp_sampleentropies)
        avg_hhtentropy_per_subject[i] = np.mean(temp_hht_entropies)
        
        avg_LZCs_onethreshold_per_subject[i] = np.mean(temp_LZCs_onethreshold)
        avg_LZCs_hilbert_per_subject[i] = np.mean(temp_LZCs_hilbert)
        avg_LZCs_median_per_subject[i] = np.mean(temp_LZCs_median)
        
        avg_higuchifractal_per_subject[i] = np.mean(temp_higuchifractal)
        
    avg_entropy_all_wake[k] = np.mean(avg_entropy_per_subject) 
    avg_spectentropy_all_wake[k] = np.mean(avg_spectentropy_per_subject) 
    avg_appentropy_all_wake[k] = np.mean(avg_appentropy_per_subject)
    avg_sampleentropy_all_wake[k] = np.mean(avg_sampleentropy_per_subject)
    avg_hhtenropy_all_wake[k] = np.mean(avg_hhtentropy_per_subject)
    avg_fooof_all_wake[k] = np.mean(avg_fooof_per_subject)
    avg_fooof_r2_all_wake[k] = np.mean(avg_fooof_r2_per_subject)
    
    avg_LZCs_onethreshold_wake[k] = np.mean(avg_LZCs_onethreshold_per_subject)
    avg_LZCs_hilbert_wake[k] = np.mean(avg_LZCs_hilbert_per_subject)
    avg_LZCs_median_wake[k] = np.mean(avg_LZCs_median_per_subject)
    
    avg_higuchifractal_wake[k] = np.mean(avg_higuchifractal_per_subject)
# ====== Wake ========
#%% ============================== Topographical analysis ===========================================

'''
data load & preparation
'''

adjacency, ch_names = find_ch_adjacency(epochs[0].info, ch_type='eeg')

# ======== Grand averaged CSD post processing v1.1 ==========
# Define function to apply CSD filter to individual epochs
def apply_grand_csd(epoch, grand_csd):
    # Apply CSD filter to epoch data
    csd_data = epoch._data * grand_csd[np.newaxis, :, :]

    # Create new Epochs object with CSD data
    epoch._data = csd_data

    return epoch

states = ['Early REM', 'Later REM', 'Lucid Dreaming', 'Wake']

all_temp_epochs = list()
for i in range(19):
        
    # ======== Grand avg CSD =======
    # stiffness = 1 / (2 * 2)**2 #roughly between 0.05 - 0.1
    
    temp_epochs = epochs[i].copy()
    csd_epochs = compute_current_source_density(temp_epochs, stiffness=0.081)
    grand_csd = np.mean(csd_epochs._data, axis=0)
    # temp_epochs = [apply_grand_csd(epoch, grand_csd) for epoch in temp_epochs]
    temp_epochs2 = list()
    for j in range(len(temp_epochs)):
        temp_epochs2.append(apply_grand_csd(temp_epochs[j], grand_csd))
        
    temp_epochs = mne.concatenate_epochs(temp_epochs2)
    # ======== Grand avg CSD =======
    
    epochs[i] = temp_epochs
#%% ================= Functions ================
def power_normalization_psd(power, baseline, dB=True):
    
    if(dB == True):
        baseline = 10 * np.log10(baseline)
        power = 10 * np.log10(power)
    
    avg_baseline = np.mean(baseline, axis=0)
    for i in range(len(power)):
        power[i] = power[i] - avg_baseline
    
    power = np.mean(power, axis=0)
    
    return power
#%% ========= Within-subject normalization of R2 - R1 (Morlet) =========
fmin, fmax = 2, 48

power_normed_list_R2R1 = list()
for i in range(len(epochs)):
    
    # ======== Normed power calculation ========    
    spectrum = epochs[i]['Early REM'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    psds_earlyrem, freqs = spectrum.get_data(return_freqs=True)
    spectrum = epochs[i]['Later REM'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    psds_laterrem, freqs = spectrum.get_data(return_freqs=True)
    
    power_normed = power_normalization_psd(psds_laterrem, psds_earlyrem, dB=True)
    # ======== Normed power calculation ========
    
    power_normed_list_R2R1.append(power_normed)
    
# ========= Within-subject normalization of LD - R2 (Morlet) =========
fmin, fmax = 2, 48

power_normed_list_LR2 = list()
for i in range(len(epochs)):
    
    # ======== Normed power calculation ========
    spectrum = epochs[i]['Later REM'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    psds_laterrem, freqs = spectrum.get_data(return_freqs=True)
    spectrum = epochs[i]['Lucid Dreaming'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    psds_luciddreaming, freqs = spectrum.get_data(return_freqs=True)

    power_normed = power_normalization_psd(psds_luciddreaming, psds_laterrem, dB=True)
    # ======== Normed power calculation ========
    
    power_normed_list_LR2.append(power_normed)
    
# ========= Within-subject normalization of LD - Wake (Morlet) =========
fmin, fmax = 2, 48

power_normed_list_LW = list()
for i in range(len(epochs)):
    
    # ======== Normed power calculation ========
    spectrum = epochs[i]['Wake'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    psds_wake, freqs = spectrum.get_data(return_freqs=True)
    spectrum = epochs[i]['Lucid Dreaming'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    psds_luciddreaming, freqs = spectrum.get_data(return_freqs=True)

    power_normed = power_normalization_psd(psds_luciddreaming, psds_wake, dB=True)
    # ======== Normed power calculation ========
    
    power_normed_list_LW.append(power_normed)
    
# ========= Within-subject normalization of LD - R1 (Morlet) =========
fmin, fmax = 2, 48

power_normed_list_LR1 = list()
for i in range(len(epochs)):
    
    # ======== Normed power calculation ========
    spectrum = epochs[i]['Early REM'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    psds_earlyrem, freqs = spectrum.get_data(return_freqs=True)
    spectrum = epochs[i]['Lucid Dreaming'].compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    psds_luciddreaming, freqs = spectrum.get_data(return_freqs=True)
    
    power_normed = power_normalization_psd(psds_luciddreaming, psds_earlyrem, dB=True)
    # ======== Normed power calculation ========

    power_normed_list_LR1.append(power_normed)
#%% ======== Frequency band wise power split and contrast between LD and others and R2 vs. R1 ========
whole_data_for_all_freqbands_LR2 = freqband_power_split_and_conditioncontrast(power_normed_list_LR2)
whole_data_for_all_freqbands_LR1 = freqband_power_split_and_conditioncontrast(power_normed_list_LR1)
whole_data_for_all_freqbands_LW = freqband_power_split_and_conditioncontrast(power_normed_list_LW)
whole_data_for_all_freqbands_R2R1 = freqband_power_split_and_conditioncontrast(power_normed_list_R2R1)
