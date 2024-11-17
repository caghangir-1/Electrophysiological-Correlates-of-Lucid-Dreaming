# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:06:35 2020

@author: Cagatay Demirel
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter, butter, filtfilt, resample, welch, argrelextrema
from scipy.signal import lfilter, savgol_filter, hilbert, fftconvolve, butter, iirnotch, freqz, firwin, iirfilter
from struct import unpack
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model, svm, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from scipy.stats import kurtosis, skew
import random
import itertools
import pickle
# import librosa
from scipy.fftpack import fft
import pywt
import mne
import csv
from scipy.stats import zscore
import scipy as sp
from scipy.signal import welch
from scipy.integrate import simps
import time
import mne
from lspopt.lsp import spectrogram_lspopt
import seaborn as sns
from mne.time_frequency import psd_array_multitaper
from mne.io import concatenate_raws, read_raw_edf
from scipy.signal import welch, periodogram
from mne.preprocessing import  (ICA, create_eog_epochs, create_ecg_epochs, corrmap, Xdawn)
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.channels import make_standard_montage
from mne import pick_types, events_from_annotations, compute_raw_covariance
from mne.baseline import rescale
from mne.preprocessing import ICA
from mne.stats import bootstrap_confidence_interval
import gc
import librosa
from mne.time_frequency import tfr_multitaper
# from mne.connectivity import seed_target_indices, spectral_connectivity
from mne_connectivity import spectral_connectivity_epochs
# from entropy.entropy import spectral_entropy
# import pyeeg
from matplotlib.legend_handler import HandlerLine2D
# import yasa
import extremeAudioFeatureExtraction as chettoAudio
import math
from fooof import FOOOFGroup

# import antropy as ent
from pyentrp import entropy as ent2
# import emd
# from lempel_ziv_complexity import lempel_ziv_complexity as LZC

class extremeEEGSignalAnalyzer():
    
    def __init__(self):
        
      self.eps = 0.00000001
    
#%%============== Pre-processing =================    
    # def notchFilter(self, data, Fs, f0, Q):
        # w0 = f0/(Fs/2)
        # b, a = iirnotch(w0, Q)
        # y = filtfilt(b, a, data)
        # return y
    
    def notchFilter(self, data, Fs, f0, Q, order=2):
        
        """
        Applies a notch filter to the data.
    
        Parameters:
        data: array-like
            The input signal.
        Fs: float
            The sampling frequency of the data.
        f0: float
            The frequency to be notched out.
        Q: float
            Quality factor - lower Q for broader and shallower notch.
        order: int, optional
            The order of the filter (default is 2).
        
        Returns:
        y: array-like
            The filtered signal.
        """
        w0 = f0 / (Fs / 2)  # Normalize the frequency
        b, a = iirnotch(w0, Q)
        
        # If a broader effect is needed, consider applying the filter multiple times
        y = filtfilt(b, a, data)
        
        # To increase the width and reduce depth, consider running the filter multiple times
        for _ in range(order - 1):
            y = filtfilt(b, a, y)
        
        return y
    
    def notchFilter_2D(self, data, Fs, f0, Q, order=2):
        
        ch_size = data.shape[0]
        for i in range(ch_size):
            data[i] = self.notchFilter(data[i], Fs, f0, Q, order)
            
        return data
    
    def envelopeCreator(self, timeSignal, degree=3, intervalLength=51, hilbert_transform=False):
        if(hilbert_transform == True):
            timeSignal = np.abs(hilbert(timeSignal))
            
        amplitude_envelopeFiltered = savgol_filter(timeSignal, intervalLength, degree)
        return amplitude_envelopeFiltered  
    
    def butter_bandpass(self, lowcut, highcut, fs, filter_type, order=3): # 3 ten sonra lfilter NaN degerler vermeye basliyor
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if(filter_type == 'butter'):
            b, a = butter(order, [low, high], btype='band', analog=False)
        elif(filter_type == 'iir'):
            b, a = iirfilter(order, [low, high], btype='bandpass', analog=False, ftype='butter')
        return b, a
        
    def butter_lowpass(self, cutoff, fs, order=3):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    # band-pass filter between two frequency     
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, filter_type='iir'):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order, filter_type=filter_type)
    #    y = lfilter(b, a, data)
        y = filtfilt(b, a, data)
        return y
    
    def butter_lowpass_filter(self, data, cutoff, fs, order):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    def FFT(self, signal, Fs):
        nFFT = len(signal) / 2
        nFFT = int(nFFT)
        #Hamming Window
        w = np.hamming(len(signal))
        #FFT
        X = abs(fft(signal * w))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        
        fIndexes = (Fs / (2*nFFT)) * np.r_[0:nFFT] # [1,9] 9peet üretti
        
        return X, fIndexes
        
    # def hpssFilter(self, data):
    #     data = librosa.effects.hpss(data.astype("float64"), margin=(1.0,5.0))    
    #     return data
    
    def binPower(self, signal, Band, Fs):
        nFFT = len(signal) / 2
        nFFT = int(nFFT)
        #Hamming Window
        w = np.hamming(len(signal))
        #FFT
        X = abs(fft(signal * w))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        
        power = np.zeros(len(Band) - 1)
        for freq_index in range(0, len(Band) - 1):
            freq = Band[freq_index]
            nextFreq = Band[freq_index + 1]
            beginInd = int(np.floor(freq * len(signal) / Fs))
            endInd = int(np.floor(nextFreq * len(signal) / Fs))
            power[freq_index] = sum(X[beginInd:endInd])
        power_ratio = power / sum(power)
        return power, power_ratio
    
    # Helper function for plotting spread
    def stat_fun(self, x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    
    #%% ============== EEG Feature Extraction ============
    def pfd(self, X, D=None):
        """Compute Petrosian Fractal Dimension of a time series from either two
        cases below:
            1. X, the time series of type list (default)
            2. D, the first order differential sequence of X (if D is provided,
               recommended to speed up)
        In case 1, D is computed using Numpy's difference function.
        To speed up, it is recommended to compute D before calling this function
        because D may also be used by other functions whereas computing it here
        again will slow down.
        """
        if D is None:
            D = np.diff(X)
            D = D.tolist()
        N_delta = 0  # number of sign changes in derivative of the signal
        for i in range(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(X)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))
        
    def hfd(self, X, Kmax):
        """ Compute Hjorth Fractal Dimension of a time series X, kmax
         is an HFD parameter
        """
        L = []
        x = []
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(float(1) / k), 1])
    
        (p, r1, r2, s) = np.linalg.lstsq(x, L)
        return p[0]
    
    def hjorth(self, X, D=None):
        """ Compute Hjorth mobility and complexity of a time series from either two
        cases below:
            1. X, the time series of type list (default)
            2. D, a first order differential sequence of X (if D is provided,
               recommended to speed up)
        In case 1, D is computed using Numpy's Difference function.
        Notes
        -----
        To speed up, it is recommended to compute D before calling this function
        because D may also be used by other functions whereas computing it here
        again will slow down.
        Parameters
        ----------
        X
            list
            a time series
        D
            list
            first order differential sequence of a time series
        Returns
        -------
        As indicated in return line
        Hjorth mobility and complexity
        """
    
        if D is None:
            D = np.diff(X)
            D = D.tolist()
    
        D.insert(0, X[0])  # pad the first difference
        D = np.array(D)
    
        n = len(X)
    
        M2 = float(sum(D ** 2)) / n
        TP = sum(np.array(X) ** 2)
        M4 = 0
        for i in range(1, len(D)):
            M4 += (D[i] - D[i - 1]) ** 2
        M4 = M4 / n
    
        return np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)  # Hjorth Mobility and Complexity
    
    def hurst(self, X):
        """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
        of the time-series is similar to random walk. If H<0.5, the time-series
        cover less "distance" than a random walk, vice verse.
        Parameters
        ----------
        X
            list
            a time series
        Returns
        -------
        H
            float
            Hurst exponent
        Notes
        --------
        Author of this function is Xin Liu
        Examples
        --------
        >>> import pyeeg
        >>> from numpy.random import randn
        >>> a = randn(4096)
        >>> pyeeg.hurst(a)
        0.5057444
        """
        X = np.array(X)
        N = X.size
        T = np.arange(1, N + 1)
        Y = np.cumsum(X)
        Ave_T = Y / T
    
        S_T = np.zeros(N)
        R_T = np.zeros(N)
    
        for i in range(N):
            S_T[i] = np.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = np.ptp(X_T[:i + 1])
    
        R_S = R_T / S_T
        R_S = np.log(R_S)[1:]
        n = np.log(T)[1:]
        A = np.column_stack((n, np.ones(n.size)))
        [m, c] = np.linalg.lstsq(A, R_S)[0]
        H = m
        return H
        
    def dfa(self, X, Ave=None, L=None):
        """Compute Detrended Fluctuation Analysis from a time series X and length of
        boxes L.
        The first step to compute DFA is to integrate the signal. Let original
        series be X= [x(1), x(2), ..., x(N)].
        The integrated signal Y = [y(1), y(2), ..., y(N)] is obtained as follows
        y(k) = \sum_{i=1}^{k}{x(i)-Ave} where Ave is the mean of X.
        The second step is to partition/slice/segment the integrated sequence Y
        into boxes. At least two boxes are needed for computing DFA. Box sizes are
        specified by the L argument of this function. By default, it is from 1/5 of
        signal length to one (x-5)-th of the signal length, where x is the nearest
        power of 2 from the length of the signal, i.e., 1/16, 1/32, 1/64, 1/128,
        ...
        In each box, a linear least square fitting is employed on data in the box.
        Denote the series on fitted line as Yn. Its k-th elements, yn(k),
        corresponds to y(k).
        For fitting in each box, there is a residue, the sum of squares of all
        offsets, difference between actual points and points on fitted line.
        F(n) denotes the square root of average total residue in all boxes when box
        length is n, thus
        Total_Residue = \sum_{k=1}^{N}{(y(k)-yn(k))}
        F(n) = \sqrt(Total_Residue/N)
        The computing to F(n) is carried out for every box length n. Therefore, a
        relationship between n and F(n) can be obtained. In general, F(n) increases
        when n increases.
        Finally, the relationship between F(n) and n is analyzed. A least square
        fitting is performed between log(F(n)) and log(n). The slope of the fitting
        line is the DFA value, denoted as Alpha. To white noise, Alpha should be
        0.5. Higher level of signal complexity is related to higher Alpha.
        Parameters
        ----------
        X:
            1-D Python list or numpy array
            a time series
        Ave:
            integer, optional
            The average value of the time series
        L:
            1-D Python list of integers
            A list of box size, integers in ascending order
        Returns
        -------
        Alpha:
            integer
            the result of DFA analysis, thus the slope of fitting line of log(F(n))
            vs. log(n). where n is the
        Examples
        --------
        >>> import pyeeg
        >>> from numpy.random import randn
        >>> print(pyeeg.dfa(randn(4096)))
        0.490035110345
        Reference
        ---------
        Peng C-K, Havlin S, Stanley HE, Goldberger AL. Quantification of scaling
        exponents and crossover phenomena in nonstationary heartbeat time series.
        _Chaos_ 1995;5:82-87
        Notes
        -----
        This value depends on the box sizes very much. When the input is a white
        noise, this value should be 0.5. But, some choices on box sizes can lead to
        the value lower or higher than 0.5, e.g. 0.38 or 0.58.
        Based on many test, I set the box sizes from 1/5 of    signal length to one
        (x-5)-th of the signal length, where x is the nearest power of 2 from the
        length of the signal, i.e., 1/16, 1/32, 1/64, 1/128, ...
        You may generate a list of box sizes and pass in such a list as a
        parameter.
        """
    
        X = np.array(X)
    
        if Ave is None:
            Ave = np.mean(X)
    
        Y = np.cumsum(X)
        Y -= Ave
    
        if L is None:
            L = np.floor(len(X) * 1 / (2 ** np.array(list(range(4, int(np.log2(len(X))) - 4)))))
    
        F = np.zeros(len(L))  # F(n) of different given box length n
    
        for i in range(0, len(L)):
            n = int(L[i])                        # for each box length L[i]
            if n == 0:
                print("time series is too short while the box length is too big")
                print("abort")
                exit()
            for j in range(0, len(X), n):  # for each box
                if j + n < len(X):
                    c = list(range(j, j + n))
                    # coordinates of time in the box
                    c = np.vstack([c, np.ones(n)]).T
                    # the value of data in the box
                    y = Y[j:j + n]
                    # add residue in this box
                    F[i] += np.linalg.lstsq(c, y)[1]
            F[i] /= ((len(X) / n) * n)
        F = np.sqrt(F)
    
        Alpha = np.linalg.lstsq(np.vstack([np.log(L), np.ones(len(L))]).T, np.log(F))[0][0]
    
        return Alpha
    
    def lempel_ziv_complexity(self, raw, approach='hilbert'):
        """Lempel-Ziv complexity for a binary sequence, customized by Cagatay."""
        
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
                    
        complexity = complexity * math.log(complexity,3) / len(binary_sequence) #we add two because other metric gives 2 more than this (fuck it dont add with 2)
        
        # complexity = LZC(binary_sequence)
        # complexity = complexity * math.log(complexity,3) / len(binary_sequence)

        return complexity


#%% ============== Other Feature Extraction Functions ================

    def stEnergy(self, frame):
        """Computes signal energy of frame"""
        return np.sum(frame ** 2) / np.float64(len(frame))

    def stEnergyEntropy(self, frame, numOfShortBlocks=10):
        
        """Computes entropy of energy"""
        Eol = np.sum(frame ** 2)    # total frame energy
        L = len(frame)
        subWinLength = int(np.floor(L / numOfShortBlocks)) # short block uzunlugu
        if L != subWinLength * numOfShortBlocks:
                frame = frame[0:subWinLength * numOfShortBlocks]
        # subWindows is of size [numOfShortBlocks x L]
        subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()
    
        # Compute normalized sub-frame energies:
        s = np.sum(subWindows ** 2, axis=0) / (Eol + self.eps)
    
        # Compute entropy of the normalized sub-frame energies:
        Entropy = -np.sum(s * np.log2(s + self.eps))
        return Entropy

    def stSpectralEntropy(self, X, numOfShortBlocks=10):
        """Computes the spectral entropy"""
        L = len(X)                         # number of frame samples
        Eol = np.sum(X ** 2)            # total spectral energy
    
        subWinLength = int(np.floor(L / numOfShortBlocks))   # length of sub-frame
        if L != subWinLength * numOfShortBlocks:
            X = X[0:subWinLength * numOfShortBlocks]
    
        subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
        s = np.sum(subWindows ** 2, axis=0) / (Eol + self.eps)                      # compute spectral sub-energies
        En = -np.sum(s*np.log2(s + self.eps))                                    # compute spectral entropy
    
        return En

    def rmsValue(self, timeSignal):
        rms = np.sqrt(np.mean(timeSignal**2))
        return rms

    def lpc(self, signal, Fs):
        """Compute the Linear Prediction Coefficients.
    
        Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
        find the k+1 coefficients of a k order linear filter:
    
          xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]
    
        Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.
    
        Parameters
        ----------
        signal: array_like
            input signal
        order : int
            LPC order (the output will have order + 1 items)"""
    
        order = int(2 + Fs / 1000)
    
        if signal.ndim > 1:
            raise ValueError("Array of rank > 1 not supported yet")
        if order > signal.size:
            raise ValueError("Input signal must have a lenght >= lpc order")
    
        if order > 0:
            p = order + 1
            r = np.zeros(p, signal.dtype)
            # Number of non zero values in autocorrelation one needs for p LPC
            # coefficients
            nx = np.min([p, signal.size])
            x = np.correlate(signal, signal, 'full')
            r[:nx] = x[signal.size-1:signal.size+order]
            phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
            # return np.concatenate(([1.], phi)), None, None
            return phi
        else:
            return np.ones(1, dtype = signal.dtype), None, None

#%% ======================== EEG Spectral / Temporal Feature Fusion ==================================
    def EEG_feature_extraction(self, data, Fs, NFFT=2**15, window='hann'):
        numofMFCC = 10
        numofFeature = 58
        sampleAmount = len(data)
        
        # Defining EEG bands:
        eeg_bands = {'Delta' : (0.5, 3),
                     'Theta' : (3  , 8),
                     'LowAlpha' : (8  , 10),
                     'HighAlpha' : (10, 12),
                     'LowBeta' : (12 , 16),
                     'Beta' : (16 , 20),
                     'HighBeta' : (20, 30),
                     'Sigma_slow': (12 , 14),
                     'Sigma_fast': (14 , 16),
                     'LowGamma' : (30 ,40),
                     'HighGamma' : (40, 48)}
        # Defining freq. resoultion
        fm, _ = periodogram(x = data[0,:], fs = Fs, nfft = NFFT , window = window)  
        freq_ix = dict()
        # Finding the index of different freq bands with respect to "fm" #
        for band in eeg_bands:
            freq_ix[band] = np.where((fm >= eeg_bands[band][0]) &   
                               (fm <= eeg_bands[band][1]))[0]    
        
        featureSet = np.zeros((sampleAmount, numofFeature))
        for i in range(sampleAmount):
            sample = data[i].flatten() #scalar sample
                    
            start = time.time()

            tempFeature = np.zeros((numofFeature))
            tempFeature[0:10] = np.mean(librosa.feature.mfcc(y=sample, sr=Fs, n_mfcc = numofMFCC), axis=1)
            tempFeature[10:11] = np.mean(librosa.feature.spectral_centroid(y=sample, sr=Fs, n_fft=256))
            tempFeature[11:12] = np.mean(librosa.feature.spectral_bandwidth(y=sample, sr=Fs, n_fft=256))
            S = np.abs(librosa.stft(sample))
            sampleFFT = self.FFT(sample, Fs)[0] 
            tempFeature[12:14] = np.mean(librosa.feature.poly_features(S=S, order=1), axis=1)
            tempFeature[14:15] = np.mean(librosa.feature.spectral_rolloff(y=sample, n_fft=256))
            #tempFeature[56:61], tempFeature[61:66] = chettoEEG.binPower(sample, freqBins, Fs) #power/power-ratio
            tempFeature[15:16] = np.mean(librosa.feature.zero_crossing_rate(y=sample, frame_length=2048, hop_length=512))
            try:
                tempFeature[16:17] = self.pfd(sample)
            except:
                tempFeature[16:17] = None
            try:
                tempFeature[17:19] = self.hjorth(sample)
            except:
                tempFeature[17:19] = None
            try:
                tempFeature[19:20] = self.hurst(sample)
            except:
                tempFeature[19:20] = None
            tempFeature[20:21] = self.stEnergy(sample)
            tempFeature[21:22] = self.stEnergyEntropy(sample)
            tempFeature[22:23] = self.stSpectralEntropy(sampleFFT)
#            tempFeature[63:64] = chettoAudio.F0UsingAutocorrelation(sample, self.Fs)
            tempFeature[23:24] = self.rmsValue(sample)
            # tempFeature[24:35] = self.brainwaveFinder(eegSignal=sample, Fs=self.Fs, density=True)
            tempFeature[24:35] = self.welchMaxPowerofBrainwaves(eegSignal=sample, Fs=Fs, \
                                                                     freqBandsIndexes=freq_ix)
            try:
                tempFeature[35:38] = self.lpc(signal=sample.flatten(), Fs=200)[0]
            except:
                tempFeature[35:38] = None
            tempFeature[38:44] = self.statisticalFeatures(signal=sample) #6
            tempFeature[44:45] = np.mean(librosa.feature.rms(y=sample, frame_length=2048, hop_length=512, center=True, \
                                                     pad_mode='reflect')[0])
                
            tempFeature[45:49] = librosa.feature.spectral_flatness(y=sample, S=None, n_fft=2048, hop_length=512, win_length=None,\
                                                  window='hann', center=True)
            tempFeature[49:56] = self.waveletDecomposition(signal=sample)
            tempFeature[56:58] = self.variance_and_meanof_vertex_and_vertex_slope(signal=sample)
            
            featureSet[i] = tempFeature
            
            end = time.time()
            print(end - start)
            
        #==== Change NaN Values ====
        aa, bb = np.where(np.isnan(featureSet))
        for j in np.arange(int(len(aa))):
            featureSet[aa[j],bb[j]] = np.nanmean(featureSet[:,bb[j]])
        #==== Change NaN Values ====
            
        # featureSet = zscore(featureSet, axis=0)
        
        return featureSet    
    
    def EEG_feature_extraction_merge(self, all_data, num_of_channels, Fs): #3D input
        
        sample_size = len(all_data)
        all_features = np.zeros((sample_size, 58*2)) #mean and std of features from multiple EEG channels
        for i in range(sample_size):
            temp_sample_all_channels = self.EEG_feature_extraction(data = all_data[i], Fs=Fs)
            all_features[i] = np.append(np.mean(temp_sample_all_channels, axis=0), np.std(temp_sample_all_channels, axis=0)) #mean, std of features
            
        return all_features
    
    # def hilbert_huang_transform(self, data, sample_rate=100, fmin=2, fmax=48):
        
    #     # ==== Setting values =====
    #     window_size = int(len(data) / 2)
    #     # ==== Setting values =====
        
    #     # === Run a mask sift =====
    #     imf = emd.sift.mask_sift(data, max_imfs=5)
    #     # === Run a mask sift =====
        
    #     # ==== Compute frequency statistics ====
    #     IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
    #     # ==== Compute frequency statistics ====
        
    #     freq_edges, freq_centres = emd.spectra.define_hist_bins(fmin, fmax, window_size, 'linear')
        
    #     # ====== Calculate HHT ========
    #     # Amplitude weighted HHT per IMF
    #     f, spec_weighted = emd.spectra.hilberthuang(IF, IA, freq_edges, sum_imfs=True)
        
    #     return spec_weighted
#%% ========== EOG feature extraction ===============
    def EOG_feature_extraction(self, epochs_data, Fs, window='hann'):
        
        # ====== Preparation ======
        numofFeature = 36
        n_fft = int(2 ** (self.nextpow2(Fs) + 2) / 8)
        S = np.abs(librosa.stft(epochs_data.copy(), n_fft=n_fft))
        sampleFFT = self.FFT(epochs_data.copy(), Fs)[0]
        tempFeature = np.zeros(numofFeature)
        # ====== Preparation ======
        
        # ======  Spectral features ======
        tempFeature[0] = self.stSpectralEntropy(sampleFFT)
        tempFeature[1] = np.mean(librosa.feature.spectral_centroid(y=epochs_data.copy(), sr=Fs, n_fft=n_fft))
        tempFeature[2] = np.mean(librosa.feature.spectral_bandwidth(y=epochs_data.copy(), sr=Fs, n_fft=n_fft))
        tempFeature[3] = np.mean(librosa.feature.spectral_rolloff(y=epochs_data.copy(), n_fft=n_fft))
        # ======  Spectral features ======

        # ====== Temporal features advanced ======
        tempFeature[4] = np.mean(librosa.feature.zero_crossing_rate(y=epochs_data.copy(), frame_length=2048, hop_length=512))
        tempFeature[5] = self.stEnergyEntropy(epochs_data.copy())
        tempFeature[6:8] = np.mean(librosa.feature.poly_features(S=S, order=1), axis=1)
        tempFeature[8:15] = self.waveletDecomposition(signal=epochs_data.copy())
        tempFeature[15:17] = self.variance_and_meanof_vertex_and_vertex_slope(signal=epochs_data.copy())
        # ====== Temporal features advanced ======

        # ======= Temporal features Simple =====
        tempFeature[17] = self.rmsValue(epochs_data.copy())
        tempFeature[18] = self.stEnergy(epochs_data.copy())
        # ======= Temporal featuresSimple =====

        # ======= Other simple features from papers ======= --> they published this in the journal in 2012 / 2021 (shit show)
        tempFeature[19] = np.max(epochs_data.copy()) #PAV
        tempFeature[20] = np.min(epochs_data.copy()) #VAV
        tempFeature[21] = np.sum(np.abs(epochs_data.copy())) #AUC
        tempFeature[22] = np.var(epochs_data.copy()) #VAR
        tempFeature[23:29] = self.statisticalFeatures(signal=epochs_data.copy())
        # ======= Other simple features from papers =======
        
        # tempFeature[29] = self.pfd(epochs_data)
        # tempFeature[30:32] = self.hjorth(epochs_data)
        # tempFeature[32] = self.pfd(epochs_data)
        
        # ============ Entropy & Complexity markers =============
        
        # ======= Entropy Analysis =======
        
        # ====== Paramater orders for entropies ====== orders 3
        perm_ent_order = 3
        apEn_order = 3
        sampEn_order = 3
        # fmin, fmax = 2, 200
        # ====== Paramater orders for entropies ====== orders 3
        
        tempFeature[29] = ent.perm_entropy(epochs_data.copy(), order=perm_ent_order, delay=2, normalize=True)
        tempFeature[30] = ent.spectral_entropy(epochs_data.copy(), sf=Fs, method='welch', normalize=True)
    
        tempFeature[31] = ent.app_entropy(epochs_data.copy(), order=apEn_order, metric='euclidean')
        tempFeature[32] = ent.sample_entropy(epochs_data.copy(), order=sampEn_order, metric='euclidean')
    
        # hht_psd = self.hilbert_huang_transform(data=epochs_data.copy(), sample_rate=Fs, fmin=fmin, fmax=fmax)
        # tempFeature[33] = ent.perm_entropy(hht_psd, order=perm_ent_order, delay=2, normalize=True)
        # ======= Entropy Analysis =======
    
        # ======= LZCs ========
        tempFeature[33] = self.lempel_ziv_complexity(epochs_data.copy(), approach='onethreestd')
        tempFeature[34] = self.lempel_ziv_complexity(epochs_data.copy(), approach='hilbert')
        tempFeature[35] = self.lempel_ziv_complexity(epochs_data.copy(), approach='median')
        # ======= LZCs ========
        
        return tempFeature
        
    #%% =========================== Brainwave Finder =================================
    # def brainwaveFinder(self, eegSignal, Fs, density=True):
    # #    p300Signal = np.mean(eegSignals, axis=0)
    #     # eegSignal = notchFilter(eegSignal, Fs, 50, 30)
        
    #     if(density==False):
    #         deltaSignal = self.butter_bandpass_filter(eegSignal, 0.5, 3, 500, order=3)
    #         thetaSignal = self.butter_bandpass_filter(eegSignal, 3, 8, Fs, order=3)
    #         alphaSignal = self.butter_bandpass_filter(eegSignal, 8, 12, Fs, order=3)
    #         betaSignal = self.butter_bandpass_filter(eegSignal, 12, 38, Fs, order=3)
    #         gammaSignal = self.butter_bandpass_filter(eegSignal, 38, 48, Fs, order=3)
            
    #         deltaSignalEnergy = np.sum(deltaSignal**2)
    #         thetaSignalEnergy = np.sum(thetaSignal**2)
    #         alphaSignalEnergy = np.sum(alphaSignal**2)
    #         betaSignalEnergy = np.sum(betaSignal**2)
    #         gammaSignalEnergy = np.sum(gammaSignal**2)
            
    #         allEnergies = np.array([deltaSignalEnergy, thetaSignalEnergy, alphaSignalEnergy, betaSignalEnergy, gammaSignalEnergy])
            
    #         return allEnergies
            
    #     else:
    #         deltaSignal = self.butter_bandpass_filter(eegSignal, 0.5, 3, 500, order=3)
    #         thetaSignal = self.butter_bandpass_filter(eegSignal, 3, 8, Fs, order=3)
    #         lowalphaSignal = self.butter_bandpass_filter(eegSignal, 8, 10, Fs, order=3)
    #         highalphaSignal = self.butter_bandpass_filter(eegSignal, 10, 12, Fs, order=3)
    #         lowbetaSignal = self.butter_bandpass_filter(eegSignal, 12, 16, Fs, order=3)
    #         betaSignal = self.butter_bandpass_filter(eegSignal, 16, 20, Fs, order=3)
    #         highbetaSignal = self.butter_bandpass_filter(eegSignal, 20, 30, Fs, order=3)
    #         sigmaslowSignal = self.butter_bandpass_filter(eegSignal, 12, 14, Fs, order=3)
    #         sigmasfastSignal = self.butter_bandpass_filter(eegSignal, 14, 16, Fs, order=3)
    #         lowgammaSignal = self.butter_bandpass_filter(eegSignal, 30, 40, Fs, order=3)
    #         highgammaSignal = self.butter_bandpass_filter(eegSignal, 40, 48, Fs, order=3)
        
    #         deltaSignalEnergy = np.sum(deltaSignal**2)
    #         thetaSignalEnergy = np.sum(thetaSignal**2)
    #         lowalphaSignalEnergy = np.sum(lowalphaSignal**2)
    #         highalphaSignalEnergy = np.sum(highalphaSignal**2)
    #         lowbetaSignalEnergy = np.sum(lowbetaSignal**2)
    #         betaSignalEnergy = np.sum(betaSignal**2)
    #         highbetaSignalEnergy = np.sum(highbetaSignal**2)
    #         sigmaslowSignalSignalEnergy = np.sum(sigmaslowSignal**2)
    #         sigmasfastSignalEnergy = np.sum(sigmasfastSignal**2)
    #         lowgammaSignalEnergy = np.sum(lowgammaSignal**2)
    #         highgammaSignalEnergy = np.sum(highgammaSignal**2)
            
    #         allEnergies = np.array([deltaSignalEnergy, thetaSignalEnergy, lowalphaSignalEnergy, highalphaSignalEnergy, \
    #                                 lowbetaSignalEnergy, betaSignalEnergy, highbetaSignalEnergy, sigmaslowSignalSignalEnergy, \
    #                                 sigmasfastSignalEnergy, lowgammaSignalEnergy, highgammaSignalEnergy])
            
    #         return allEnergies
    
    def welchMaxPowerofBrainwaves(self, eegSignal, Fs, freqBandsIndexes, Nfft = 2 ** 15, window='hann', density=True):
        '''Apply Welch to see the dominant Max power in each freq band''' 
        ff, Psd = welch(x=eegSignal, fs=Fs, window=window, nperseg=512, nfft=Nfft)
       
        if(density==False):
            Pow_max_Total = np.max(Psd[np.arange(freqBandsIndexes['Delta'][0], freqBandsIndexes['Gamma'][-1]+1)])
            Pow_max_Delta = np.max(Psd[freqBandsIndexes['Delta']])
            Pow_max_Theta = np.max(Psd[freqBandsIndexes['Theta']])
            Pow_max_Alpha = np.max(Psd[freqBandsIndexes['Alpha']])
            Pow_max_Beta = np.max(Psd[freqBandsIndexes['Beta']])
            Pow_max_Gamma = np.max(Psd[freqBandsIndexes['Gamma']])
            
            allWelchPowers = np.array([Pow_max_Total, Pow_max_Delta, Pow_max_Theta, Pow_max_Alpha, \
                                       Pow_max_Beta, Pow_max_Gamma])
            return allWelchPowers
                
        else:
            Pow_max_Total = np.max(Psd[np.arange(freqBandsIndexes['Delta'][0], freqBandsIndexes['Beta'][-1]+1)])
            Pow_max_Delta = np.max(Psd[freqBandsIndexes['Delta']])
            Pow_max_Theta = np.max(Psd[freqBandsIndexes['Theta']])
            Pow_max_Lowalpha = np.max(Psd[freqBandsIndexes['LowAlpha']])
            Pow_max_Highalpha = np.max(Psd[freqBandsIndexes['HighAlpha']])
            Pow_max_LowBeta = np.max(Psd[freqBandsIndexes['LowBeta']])
            Pow_max_Beta = np.max(Psd[freqBandsIndexes['Beta']])
            Pow_max_HighBeta = np.max(Psd[freqBandsIndexes['HighBeta']])
            Pow_max_SigmaSlow = np.max(Psd[freqBandsIndexes['Sigma_slow']])
            Pow_max_SigmaFast = np.max(Psd[freqBandsIndexes['Sigma_fast']])
            Pow_max_LowGamma = np.max(Psd[freqBandsIndexes['LowGamma']])
            Pow_max_HighGamma = np.max(Psd[freqBandsIndexes['HighGamma']])
            
            allWelchPowers = np.array([Pow_max_Total, Pow_max_Delta, Pow_max_Lowalpha, Pow_max_Highalpha, \
                                       Pow_max_LowBeta, Pow_max_Beta, Pow_max_HighBeta, Pow_max_SigmaSlow, \
                                       Pow_max_SigmaFast, Pow_max_LowGamma, Pow_max_HighGamma])
            return allWelchPowers
        
    def welchBrainWaveFinder(self, eegSignal, Fs, Nfft = 2 ** 15, window='hann'):
        
        eeg_bands = {'Delta' : (0.5, 3),
                     'Theta' : (3  , 7),
                     'Alpha' : (8, 12),
                     'Beta' : (13,30),
                     'Gamma' : (30,50)}
        
        nfft = int(2 ** (np.ceil(np.log2(len(eegSignal))) + 2))
        fm, _ = periodogram(x = eegSignal, fs = Fs, nfft = nfft , window = window)  
        freqBandsIndexes = dict()
        for band in eeg_bands:
            freqBandsIndexes[band] = np.where((fm >= eeg_bands[band][0]) & (fm <= eeg_bands[band][1]))[0]  
            
        '''Apply Welch to see the dominant Max power in each freq band''' 
        ff, Psd = welch(x=eegSignal, fs=Fs, window=window, nperseg=512, nfft=Nfft)
            
        Pow_Total = np.sum(Psd[np.arange(freqBandsIndexes['Delta'][0], freqBandsIndexes['Gamma'][-1]+1)])
        Pow_Delta = np.sum(Psd[freqBandsIndexes['Delta']])
        Pow_Theta = np.sum(Psd[freqBandsIndexes['Theta']])
        Pow_Alpha = np.sum(Psd[freqBandsIndexes['Alpha']])
        Pow_Beta = np.sum(Psd[freqBandsIndexes['Beta']])
        Pow_Gamma = np.sum(Psd[freqBandsIndexes['Gamma']])
        
        allWelchPowers = np.array([Pow_Total, Pow_Delta, Pow_Theta, Pow_Alpha, \
                                   Pow_Beta, Pow_Gamma])
        return allWelchPowers
    #%% ========== Statistical Features ==========
    def statisticalFeatures(self, signal):
        ''' Statisctical features'''
        kurt     = kurtosis(signal, fisher = False)
        skewness = skew(signal)
        mean     = np.mean(signal)
        median   = np.median(signal)
        std      = np.std(signal)
        ''' Coefficient of variation '''
        coeff_var = std / mean
        
        allData = np.array([kurt, skewness, mean, median, std, coeff_var])
        return allData
    
    #%% ========== Normalizations =============
    def zscore_norm(self, data, axis=1):
        normed_data = zscore(a=data, axis=axis)
        
        meann = np.mean(data, axis=axis)
        stdd = np.std(data, axis=axis)
        
        return normed_data, meann, stdd
    #%% ======== Wavelet Decomposition ========
    def waveletDecomposition(self, signal):
        ''' Wavelet Decomposition ''' 
        cA,cD=pywt.dwt(signal,'coif1')
        # cA_values.append(cA)
        
        cA_mean = np.mean(cA)
        cA_std = np.std(cA)
        cA_Energy = np.sum(np.square(cA))
        cD_mean = np.mean(cD)
        cD_std = np.std(cD)
        cD_Energy = np.sum(np.square(cD))
        Entropy_D = np.sum(np.square(cD) * np.log(np.square(cD)))
        Entropy_A = np.sum(np.square(cA) * np.log(np.square(cA)))
        
        allData = np.array([cA_mean, cA_std, cA_Energy, cD_mean, cD_std, Entropy_D, Entropy_A])
        return allData
    #%% ========= First and second difference mean and max ======
    def firstSecondDiff_MeanMax(self, signal):
        ''' First and second difference mean and max '''
        sum1  = 0.0
        sum2  = 0.0
        Max1  = 0.0
        Max2  = 0.0
        first_diff = np.zeros(len(signal)-1)
        
        for j in range(len(signal)-1):
                sum1     += abs(signal[j+1]-signal[j])
                first_diff[j] = abs(signal[j+1]-signal[j])
                
                if first_diff[j] > Max1: 
                    Max1 = first_diff[j] # fi
                    
        for j in range(len(signal)-2):
                sum2 += abs(first_diff[j+1]-first_diff[j])
                if abs(first_diff[j+1]-first_diff[j]) > Max2 :
                	Max2 = first_diff[j+1]-first_diff[j] 
                    
        diff_mean1 = sum1 / (len(signal)-1)
        diff_mean2 = sum2 / (len(signal)-2) 
        diff_max1  = Max1
        diff_max2  = Max2
        
        allData = np.array([diff_mean1, diff_mean2, diff_max1, diff_max2])
        return allData
    #%% ============== Variance and mean of vertex and vertex slope ===========
    def variance_and_meanof_vertex_and_vertex_slope(self, signal):
        ''' Variance and Mean of Vertex to Vertex Slope '''
        t_max   = argrelextrema(signal, np.greater)[0]
        amp_max = signal[t_max]
        t_min   = argrelextrema(signal, np.less)[0]
        amp_min = signal[t_min]
        tt      = np.concatenate((t_max,t_min),axis=0)
        if len(tt)>0:
            tt.sort() #sort on the basis of time
            h=0
            amp = np.zeros(len(tt))
            res = np.zeros(len(tt)-1)
            
            for l in range(len(tt)):
                    amp[l] = signal[tt[l]]
                    
            out = np.zeros(len(amp)-1)     
             
            for j in range(len(amp)-1):
                out[j] = amp[j+1]-amp[j]
            amp_diff = out
            
            out = np.zeros(len(tt)-1)  
            
            for j in range(len(tt)-1):
                out[j] = tt[j+1]-tt[j]
            tt_diff = out
            
            for q in range(len(amp_diff)):
                    res[q] = amp_diff[q]/tt_diff[q] #calculating slope        
            
            slope_mean = np.mean(res) 
            slope_var  = np.var(res)   
        else:
            slope_var, slope_mean = 0, 0
            
        allData = np.array([slope_mean, slope_var])
        return allData
    #%% =============== ERD/ERs Calculation ===================
    def ERDS(self, raw):
        raw
    
    #%% ============== Hypnogram Processor ============
    def hypnogram_comparator(self, file_paths, compare_strings_left, compare_strings_right, convenient_one=None):
        
        file_list = list()
        for file in file_paths:
            temp_file = open(file, 'r')
            temp_file = temp_file.readlines()
            
            temp_array = np.empty(shape=(0,2), dtype=int)
            for i in range(len(temp_file)):
                if(temp_file[i][0] == '-'):
                    temp_array = np.row_stack((temp_array, np.array([int(temp_file[i][0:2]), int(temp_file[i][3])])))
                else:
                    temp_array = np.row_stack((temp_array, np.array([int(temp_file[i][0]), int(temp_file[i][2])])))
                
            file_list.append(temp_array)
            
        # np.array([int(temp_file[i][0]), int(temp_file[i][-2])])
        
        # ===== Detect Indexes =======
        compare_strings_left_founding_0 = np.argwhere(file_list[0][:,0] == compare_strings_left)
        compare_strings_left_founding_1 = np.argwhere(file_list[1][:,0] == compare_strings_left)
        
        compare_strings_right_founding_0 = np.argwhere(file_list[0][:,1] == compare_strings_right)
        compare_strings_right_founding_1 = np.argwhere(file_list[1][:,1] == compare_strings_right)
        
        if(convenient_one is not None):
            non_artifact_strings_left = np.argwhere(file_list[convenient_one][:,1] == 0) #0 is non-artifact
        
        questionmark_strings_left_0 = np.argwhere(file_list[0][:,0] == -1)
        questionmark_strings_left_1 = np.argwhere(file_list[1][:,0] == -1)
        # ===== Detect Indexes =======
        
        # ================= Common Indexes =====================
        common_indexes_sleep_stage = set(list(compare_strings_left_founding_0.flatten())) & \
                                     set(list(compare_strings_left_founding_1.flatten()))
        
        common_indexes_artifacts = set(list(compare_strings_right_founding_0.flatten())) & \
                                   set(list(compare_strings_right_founding_1.flatten()))
          
        common_indexes_questionmarks = set(list(questionmark_strings_left_0.flatten())) & \
                                       set(list(questionmark_strings_left_1.flatten()))
        
        if(convenient_one is not None):
                common_indexes_sleep_stage_noartifact = common_indexes_sleep_stage & \
                                                        set(list(non_artifact_strings_left.flatten()))
        else:
            common_indexes_sleep_stage_noartifact = None
        
        if(len(common_indexes_sleep_stage) > 0):
            common_indexes_sleep_stage = np.array(list(common_indexes_sleep_stage)) + 1
            common_indexes_sleep_stage = np.sort(common_indexes_sleep_stage)
        else:
            common_indexes_sleep_stage = 'None'
            
        if(len(common_indexes_artifacts) > 0):
            common_indexes_artifacts = np.array(list(common_indexes_artifacts)) + 1
            common_indexes_artifacts = np.sort(common_indexes_artifacts)
        else:
            common_indexes_artifacts = 'None'
            
        if(len(common_indexes_questionmarks) > 0):
            common_indexes_questionmarks = np.array(list(common_indexes_questionmarks)) + 1
            common_indexes_questionmarks = np.sort(common_indexes_questionmarks)
        else:
            common_indexes_questionmarks = 'None'
            
        if(common_indexes_sleep_stage_noartifact is not None):
           common_indexes_sleep_stage_noartifact = np.array(list(common_indexes_sleep_stage_noartifact)) + 1
           common_indexes_sleep_stage_noartifact = np.sort(common_indexes_sleep_stage_noartifact)
        else:
            common_indexes_sleep_stage_noartifact = 'None'
        # ===================== Common Indexes ===================
        
        lengths = {
                   'Amount of Left Strings 1st' : len(compare_strings_left_founding_0),\
                   'Amount of Left Strings 2nd' : len(compare_strings_left_founding_1),\
                   'Amount of Right Strings 1st' : len(compare_strings_right_founding_0),\
                   'Amount of Right Strings 2nd' : len(compare_strings_right_founding_1),\
                   'Common Amount of Left Strings' : len(common_indexes_sleep_stage),\
                   'Common Amount of Right Strings' : len(common_indexes_artifacts),\
                   'Common Amount of Indexes Sleep Stage & No Artifact' : len(common_indexes_sleep_stage_noartifact)
                  }
            
        indexes = {
                   'Common Indexes Sleep Stage' : common_indexes_sleep_stage,\
                   'Common Indexes Artifacts' : common_indexes_artifacts,\
                   'Common Indexes Question Marks' : common_indexes_questionmarks,\
                   'Common Indexes Sleep Stage & No Artifact' : common_indexes_sleep_stage_noartifact 
                  }
        
        return indexes, lengths
    #%% ======================== EDF Reading & Decomposing ===========================
    def EDFDecomposer(self, multipleFolders, writing_directory):
        
        os.chdir(writing_directory) #write everything into that directory
        
        if(len(multipleFolders) == 1):
            multipleFolders = multipleFolders, #change string into tuple
        open('meta_edf_decomposition.csv', 'w', newline='')
        file_overall = open("meta_edf_decomposition_overall.txt","w") 
        
        #====== Initial Definitions ======
        allFs = list()
        all_datachannels_all_folders = list()
        common_channels_all_folders = list()
        allLengthHours = list()
        allInfo = list()
        overallAmountofEDFs = 0    
        #====== Initial Definitions ======
    
        for folder_path in multipleFolders:
            files = list()
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".edf"):
                    files.append(file)
            
            #===== Definitions =====
            dataChannels = list()
            dataSamplingRates = list()
            lengthSeconds = list()
            count = len(files)
            dataInfos = list()
            tempList = list()
            #===== Definitions =====
            
            overallAmountofEDFs += count
            
            #======= Read EDF File =============
            for i in range(count):
                data = mne.io.read_raw_edf(folder_path + '/' + files[i], preload=False)
                # dataSets.append(data)
                dataInfo = data.info
                dataInfos.append(dataInfo)
                dataChannels.append(dataInfo['ch_names'])
                dataSamplingRates.append(dataInfo['sfreq'])
                lengthSeconds.append(len(list(data[0])[0].flatten()) / dataInfo['sfreq'])
                
                allFs.append(dataInfo['sfreq'])
                # allDataChannels.append(dataInfo['ch_names'])
            #======= Read EDF File =============
            
            lengthMinutes, lengthHours = list(), list()
            for i in range(count):
                lengthMinutes.append(lengthSeconds[i] / 60)
                lengthHours.append(lengthMinutes[i] / 60)
                allLengthHours.append(lengthMinutes[i] / 60)
                
            #====== Add items to list =======
            tempList.append(dataInfo)
            tempList.append(dataChannels)
            tempList.append(dataSamplingRates)
            tempList.append(lengthHours)
            
            allInfo.append(tempList)
            #====== Add items to list =======
                
            '''===== Write information to a CSV file =========='''
            folder_name = folder_path.split('/')[-1]
            row_list = [[folder_name, 'File ID', 'Channel Size', 'Channels', 'Fs', 'Length(minutes)', 'Length(hours)']]
            for i in range(count):
                row_list.append(['', files[i].split('.edf')[0], str(len(dataChannels[i])), dataChannels[i], str(dataSamplingRates[i]), \
                                str(round(lengthMinutes[i],2)), str(round(lengthHours[i],2))])
                
            with open('meta_edf_decomposition.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)    
            
            with open('meta_edf_decomposition.csv', 'a+', newline='') as file:
                writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
                writer.writerow('')
                writer.writerow('')
            '''===== Write information to a CSV file =========='''
             
            '''============ Calculation of statistics for each folder ==============='''
            amountofEDF = count
            minHours, maxHours = round(min(lengthHours),2), round(max(lengthHours),2)
            varietyOfFs, counts = np.unique(dataSamplingRates, return_counts=True)
            most_used_Fs_of_folder = varietyOfFs[np.argmax(counts)] 
            
            #===== All Data Channels Appending ====
            alldatachannels_one_folder = list()
            for dataChannels_one_file in dataChannels:
                for channel in dataChannels_one_file:
                    alldatachannels_one_folder.append(channel)
                    all_datachannels_all_folders.append(channel)
            #===== All Data Channels Appending ====
            
            unique_channels_one_folder = np.unique(alldatachannels_one_folder)
            
            #===== Common channel finding for a folder =====
            if(len(dataChannels) >= 2):
                common_channels_one_folder=list(set(dataChannels[0]).intersection(dataChannels[1]))
                for i in range(1, len(dataChannels)-1):
                    common_channels_one_folder=list(set(common_channels_one_folder).intersection(dataChannels[i+1]))
                common_channels_all_folders.append(common_channels_one_folder) #it is already string array
            
            else:
                common_channels_one_folder = dataChannels[0] #it is list and to string array by [0]
                common_channels_all_folders.append(common_channels_one_folder) 
                
            #===== Common channel finding for a folder =====
            
            '''============ Calculation of statistics for each folder==============='''                
            #===== Write Overall Information of single folder to a text file ========
            file_overall.write(folder_name + ' :\n')
            file_overall.write('Amount of EDF files : ' + str(amountofEDF) + '\n')
            file_overall.write('Minimum length of EDF files : ' + str(minHours) + ' hours\n')
            file_overall.write('Maximum length of EDF files : ' + str(maxHours) + ' hours\n')
            file_overall.write('Fs variations :' + str(varietyOfFs) + ' Hz\n')
            file_overall.write('Most Used Fs :' + str(most_used_Fs_of_folder) + ' Hz\n')
            file_overall.write('Unique channels :' + str(unique_channels_one_folder) + '\n')
            file_overall.write('Common channels :' + str(common_channels_one_folder) + '\n')
            file_overall.write('Common channel amount :' + str(len(common_channels_one_folder)) + '\n')
            file_overall.write('\n') #gap     
            file_overall.write('****************************************') #gap for next folder
            file_overall.write('\n') #gap for next folder    
            #===== Write Overall Information of single folder to a text file ========
    
        '''======= Write Overall Information of whole folder to a text file ========='''    
        
        #=============== Overall Statistics ===============
        overall_unique_dataChannels = np.unique(all_datachannels_all_folders)
        unique_Fs_of_all_folders, counts = np.unique(allFs, return_counts=True)
        most_used_Fs_of_all_folders = unique_Fs_of_all_folders[np.argmax(counts)]   
        #===== Common channel finding for a folder =====
    
        if(len(common_channels_all_folders) >= 2):
            overall_common_channels_all_folders=list(set(common_channels_all_folders[0]).\
                                                     intersection(common_channels_all_folders[1]))
            for i in range(1, len(common_channels_all_folders)-1):
                overall_common_channels_all_folders=list(set(overall_common_channels_all_folders).\
                                                         intersection(common_channels_all_folders[i+1]))
                    
            # overall_common_channels_all_folders = common_channels_all_folders #it is already string array
        else:
            overall_common_channels_all_folders = common_channels_all_folders[0]  #it is list and to string array by [0]
        #===== Common channel finding for a folder =====
        
        #=============== Overall Statistics ===============
        
        file_overall.write('Overall :\n')
        file_overall.write('Amount of EDF files :' + str(overallAmountofEDFs) + '\n')
        file_overall.write('Minimum length of whole EDF files ' + str(round(min(allLengthHours),2)) + ' hours\n')
        file_overall.write('Maximum Length of whole EDF files ' + str(round(max(allLengthHours),2)) + ' hours\n')
        file_overall.write('Total Fs variations :' + str(unique_Fs_of_all_folders) + ' Hz\n')
        file_overall.write('Most Used Fs :' + str(most_used_Fs_of_all_folders) + ' Hz\n')
        file_overall.write('Unique channels :' + str(overall_unique_dataChannels) + '\n')
        file_overall.write('Common channels :' + str(overall_common_channels_all_folders) + '\n')
        file_overall.write('Common channel amount :' + str(len(overall_common_channels_all_folders)) + '\n')
        
        '''======= Write Overall Information of whole folder to a text file ========='''
        
        file_overall.close() #finalize
        
        return allInfo, overall_unique_dataChannels, common_channels_all_folders, overall_common_channels_all_folders
    #%% ======================== Plot Confusion Matrix =======================
    
    def plot_confusion_matrix(self, cm, classes, title, normalize=False, cmap=plt.cm.Blues, saving_directory=None, dpi=400):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size=30)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=20)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", size=20)
    
        plt.tight_layout()
        plt.ylabel('True label', size=20)
        plt.xlabel('Predicted label', size=20)
        
        if(saving_directory is not None):
            self.save_figure(saving_directory=saving_directory, explanation=title, dpi=dpi)
        
    #%% ================= Feature Extraction ==================
    def featureExtraction(self, X_train, Fs, nfft=None, window='hann', numofMFCC=10, baselineinterval=None):
                  
        if(baselineinterval is not None):
            X_train -= np.mean(baselineinterval)
        # if(prenorm==True):
        #     X_train = zscore(X_train, axis=2)
        
        sampleAmount = len(X_train)
        amountofchannel = np.size(X_train, 1)
        
        # Defining EEG bands:
        eeg_bands = {'Delta' : (0.5, 3),
                      'Theta' : (3  , 8),
                      'LowAlpha' : (8  , 10),
                      'HighAlpha' : (10, 12),
                      'LowBeta' : (12 , 16),
                      'Beta' : (16 , 20),
                      'HighBeta' : (20, 30),
                      'Sigma_slow': (12 , 14),
                      'Sigma_fast': (14 , 16),
                      'LowGamma' : (30 ,40),
                      'HighGamma' : (40, 48)}
        # Defining freq. resoultion
        
        if(nfft is None):
            nfft = int(2 ** (np.ceil(np.log2(len(X_train[0,0]))) + 2))
        
        fm, _ = periodogram(x = X_train[0,0], fs = Fs, nfft = nfft , window = window)  
        freq_ix = dict()
        # Finding the index of different freq bands with respect to "fm" #
        for band in eeg_bands:
            freq_ix[band] = np.where((fm >= eeg_bands[band][0]) &   
                                (fm <= eeg_bands[band][1]))[0]    
        
        numofFeature = 53
        featureSet = np.zeros((sampleAmount, numofFeature))
        for i in range(sampleAmount):
            tempFeatureSet = np.zeros((amountofchannel, numofFeature))
            for j in range(amountofchannel):
                # sample = X_train[i].flatten() #scalar sample
                sample = X_train[i,j]        
                
                start = time.time()
    
                tempFeature = np.zeros((numofFeature))
                tempFeature[0:10] = np.mean(librosa.feature.mfcc(y=sample, sr=Fs, n_mfcc = numofMFCC), axis=1)
                tempFeature[10:11] = np.mean(librosa.feature.spectral_centroid(y=sample, sr=Fs, n_fft=256))
                tempFeature[11:12] = np.mean(librosa.feature.spectral_bandwidth(y=sample, sr=Fs, n_fft=256))
                S = np.abs(librosa.stft(sample))
                sampleFFT = chettoAudio.FFT(sample, Fs)[0] 
                tempFeature[12:14] = np.mean(librosa.feature.poly_features(S=S, order=1), axis=1)
                tempFeature[14:15] = np.mean(librosa.feature.spectral_rolloff(y=sample, n_fft=256))
                #tempFeature[56:61], tempFeature[61:66] = chettoEEG.binPower(sample, freqBins, Fs) #power/power-ratio
                tempFeature[15:16] = np.mean(librosa.feature.zero_crossing_rate(y=sample, frame_length=2048, hop_length=512))
                try:
                    tempFeature[16:17] = self.pfd(sample)
                except:
                    tempFeature[16:17] = None
                try:
                    tempFeature[17:19] = self.hjorth(sample)
                except:
                    tempFeature[17:19] = None
                try:
                    tempFeature[19:20] = self.hurst(sample)
                except:
                    tempFeature[19:20] = None
                # tempFeature[20:21] = chettoAudio.stEnergy(sample)
                tempFeature[20:21] = chettoAudio.stEnergyEntropy(sample)
                tempFeature[21:22] = chettoAudio.stSpectralEntropy(sampleFFT)
    #            tempFeature[63:64] = chettoAudio.F0UsingAutocorrelation(sample, self.Fs)
                # tempFeature[23:24] = chettoAudio.rmsValue(sample)
                # tempFeature[24:35] = self.brainwaveFinder(eegSignal=sample, Fs=Fs, density=True)
                tempFeature[22:33] = self.welchMaxPowerofBrainwaves(eegSignal=sample, Fs=Fs, \
                                                                          freqBandsIndexes=freq_ix)
                try:
                    tempFeature[33:36] = chettoAudio.lpc(signal=sample.flatten(), Fs=Fs)[0]
                except:
                    tempFeature[33:36] = None
                tempFeature[36:40] = self.statisticalFeatures(signal=sample) #6
                tempFeature[40:41] = np.mean(librosa.feature.rms(y=sample, frame_length=2048, hop_length=512, center=True, \
                                                          pad_mode='reflect')[0])
                    
                tempFeature[41:44] = librosa.feature.spectral_flatness(y=sample, S=None, n_fft=nfft, hop_length=512, win_length=None,\
                                                      window='hann', center=True)
                tempFeature[44:51] = self.waveletDecomposition(signal=sample)
                tempFeature[51:53] = self.variance_and_meanof_vertex_and_vertex_slope(signal=sample)
                
                tempFeatureSet[j] = tempFeature
                
                end = time.time()
                print(end - start)
                
            featureSet[i] = np.mean(tempFeatureSet, 0)
            print('Sample Averaging Completed!')
            
        #==== Change NaN Values ====
        aa, bb = np.where(np.isnan(featureSet))
        for j in np.arange(int(len(aa))):
            featureSet[aa[j],bb[j]] = np.nanmean(featureSet[:,bb[j]])
        #==== Change NaN Values ====
            
        return featureSet
#%% ================ Alternative Feature Extractor ===========================
    def FeatureExtraction_per_subject(self, X_train, Fs, duration=3, nfft=None, prenorm=False, window='hann', numofMFCC=10, new_order=None):
        
        # if(baselineinterval is not None):
        #     X_train -= np.mean(baselineinterval)
        if(prenorm==True):
            X_train = zscore(X_train, axis=2)
        
        sampleAmount = len(X_train)
        amountofchannel = np.size(X_train, 1)
        
        # Defining EEG bands:
        eeg_bands = {'Delta' : (0.5, 4),
                     'Theta_low' : (4  , 6),
                     'Theta_high': (6  , 7),
                     'Alpha'     : (8  , 12),
                     'Beta'      : (12 , 30),
                     'Sigma_slow': (10 , 12),
                     'Sigma': (12 , 15),
                     'Gamma_low' : (30 ,40),
                     'Gamma_high' : (40, 48)}
        # Defining freq. resoultion
        
        if(nfft is None):
            nfft = int(2 ** (np.ceil(np.log2(len(X_train[0,0]))) + 2))
        
        fm, _ = periodogram(x = X_train[0,0], fs = Fs, nfft = nfft , window = window)  
        freq_ix = dict()
        freq_ix_welch     = dict()
        
        # Finding the index of different freq bands with respect to "fm" #
        for band in eeg_bands:
            freq_ix[band] = np.where((fm >= eeg_bands[band][0]) &   
                                (fm <= eeg_bands[band][1]))[0]    
            
        window_len = 4 # secs
        ff, _      = welch(x = X_train[0,0], fs = Fs, window = 'hann', nperseg = Fs*window_len)
            
         # Finding the index of different freq bands with respect to "fm" WELCH#
        for band in eeg_bands:
            freq_ix_welch[band] = np.where((ff >= eeg_bands[band][0]) &   
                               (ff <= eeg_bands[band][1]))[0]    
        
        numofFeature = 43 #84
        featureSet = np.zeros((sampleAmount, numofFeature))
        all_feature_set = np.zeros((sampleAmount, amountofchannel, numofFeature))
        diff_feature_set = np.zeros((sampleAmount, 5, numofFeature))
        for iii in range(sampleAmount):
            tempFeatureSet = np.zeros((amountofchannel, numofFeature))
            for jjj in range(amountofchannel):
            
                start = time.time()    
            
                data = X_train[iii,jjj]
                            
                ### Initialization for wavelet 
                
                # 4th appr coef
                cA_values4  = []
                
                # 1st to 4th det coef
                cD_values4  = []
                cD_values3  = []
                cD_values2  = []
                cD_values1  = []
                
                # mean and std of appr coef
                cA_mean4    = []
                cA_std4     = []
                
                # mean and std of det coefs
                cD_mean4    = []
                cD_std4     = []
                cD_mean3    = []
                cD_std3     = []
                cD_mean2    = []
                cD_std2     = []
                cD_mean1    = []
                cD_std1     = []
                
                # Energy of appr coefs
                cA_Energy4  = []
                
                # Energy of det coefs
                cD_Energy4  = []
                cD_Energy3  = []
                cD_Energy2  = []
                cD_Energy1  = []
                
                # Entropy of appr coef
                Entropy_A4  = []
                
                # Entropy of det coefs
                Entropy_D4  = []
                Entropy_D3  = []
                Entropy_D2  = []
                Entropy_D1  = []
    
                first_diff = np.zeros(len(data)-1)
                
                ''' Power of signal --> Peridogram with padding'''
                # Compute the "total" power inside the investigational window
                # _ , pxx = periodogram(x = data, fs = Fs, nfft = nfft , window = window) 
                # freq_resolu_per= fm[1] - fm[0]
                
                # pow_total      = simps(pxx, dx = freq_resolu_per)
                # Pow_Delta      = simps(pxx[freq_ix['Delta']], dx = freq_resolu_per) 
                # Pow_Theta_low  = simps(pxx[freq_ix['Theta_low']], dx = freq_resolu_per) 
                # Pow_Theta_high = simps(pxx[freq_ix['Theta_high']], dx = freq_resolu_per) 
                # Pow_Alpha      = simps(pxx[freq_ix['Alpha']], dx = freq_resolu_per) 
                # Pow_Beta       = simps(pxx[freq_ix['Beta']], dx = freq_resolu_per)  
                # Pow_Sigma      = simps(pxx[freq_ix['Sigma']], dx = freq_resolu_per) 
                # Pow_Sigma_slow = simps(pxx[freq_ix['Sigma_slow']], dx = freq_resolu_per)  
                # Pow_Gamma_low  = simps(pxx[freq_ix['Gamma_low']], dx = freq_resolu_per)  
                # Pow_Gamma_high  = simps(pxx[freq_ix['Gamma_high']], dx = freq_resolu_per)  
                
                
                '''Power ratio in differnt freq ranges (Periodogram)''' 
                # Total pow is defined form 0.5 - 20 Hz
                # Pow_Delta_ratio      = Pow_Delta / pow_total
                # Pow_Theta_low_ratio  = Pow_Theta_low / pow_total
                # Pow_Theta_high_ratio = Pow_Theta_high / pow_total
                # Pow_Alpha_ratio      = Pow_Alpha / pow_total
                # Pow_Beta_ratio       = Pow_Beta / pow_total
                # Pow_Sigma_ratio      = Pow_Sigma / pow_total
                # Pow_Sigma_slow_ratio = Pow_Sigma_slow / pow_total
                # Pow_Gamma_low_ratio  = Pow_Gamma_low / pow_total
                # Pow_Gamma_high_ratio = Pow_Gamma_high / pow_total
                
                '''Apply WELCH to see the ABSOLUTE power in each freq band'''
                # window_len = 4 # secs
                # ff, Psd             = welch(x = data, fs = Fs, window = 'hann', nperseg = Fs*window_len)
                # freq_resolu_welch   = ff[1] - ff[0]
                
                # Pow_welch_Total       = simps(Psd, dx = freq_resolu_welch)
                # Pow_welch_Delta       = simps(Psd[freq_ix_welch['Delta']], dx = freq_resolu_welch)
                # Pow_welch_Theta_low   = simps(Psd[freq_ix_welch['Theta_low']], dx = freq_resolu_welch)
                # Pow_welch_Theta_high  = simps(Psd[freq_ix_welch['Theta_high']], dx = freq_resolu_welch)
                # Pow_welch_Alpha       = simps(Psd[freq_ix_welch['Alpha']], dx = freq_resolu_welch)
                # Pow_welch_Beta        = simps(Psd[freq_ix_welch['Beta']], dx = freq_resolu_welch)
                # Pow_welch_Sigma       = simps(Psd[freq_ix_welch['Sigma']], dx = freq_resolu_welch)
                # Pow_welch_Sigma_slow  = simps(Psd[freq_ix_welch['Sigma_slow']], dx = freq_resolu_welch)
                # Pow_welch_Gamma_low   = simps(Psd[freq_ix_welch['Gamma_low']], dx = freq_resolu_welch)
                # Pow_welch_Gamma_high  = simps(Psd[freq_ix_welch['Gamma_high']], dx = freq_resolu_welch)
                
                '''Apply WELCH to see the RELATIVE power in each freq band'''
    
                # Pow_welch_Delta_rel       = Pow_welch_Delta / Pow_welch_Total
                # Pow_welch_Theta_low_rel   = Pow_welch_Theta_low / Pow_welch_Total
                # Pow_welch_Theta_high_rel  = Pow_welch_Theta_high / Pow_welch_Total
                # Pow_welch_Alpha_rel       = Pow_welch_Alpha / Pow_welch_Total
                # Pow_welch_Beta_rel        = Pow_welch_Beta / Pow_welch_Total
                # Pow_welch_Sigma_rel       = Pow_welch_Sigma / Pow_welch_Total
                # Pow_welch_Sigma_slow_rel  = Pow_welch_Sigma_slow / Pow_welch_Total
                # Pow_welch_Gamma_low_rel   = Pow_welch_Gamma_low / Pow_welch_Total
                # Pow_welch_Gamma_high_rel  = Pow_welch_Gamma_high / Pow_welch_Total
                
                ''' Spectral Entropy '''
                Entropy_Welch = spectral_entropy(x = data, sf=Fs, method='welch', nperseg = Fs* window_len)
                Entropy_fft   = spectral_entropy(x = data, sf=Fs, method='fft')
                   
                ''' Wavelet Decomposition ''' 
                # Extract 4 det compositions
                coeffs = pywt.wavedec(data, 'db10', level=4)
                cA4, cD4, cD3, cD2, cD1 = coeffs
                
                # Appending appr values
                cA_values4.append(cA4)
                
                # Appending det coefs values
                cD_values4.append(cD4)
                cD_values3.append(cD3)
                cD_values2.append(cD2)
                cD_values1.append(cD1)
                
                # Calculate mean and std of appr coefs
                cA_mean4.append(np.mean(cA_values4))
                cA_std4.append(np.std(cA_values4))
                
                # Calculate mean of det coefs
                cD_mean4.append(np.mean(cD_values4))
                cD_mean3.append(np.mean(cD_values3))
                cD_mean2.append(np.mean(cD_values2))
                cD_mean1.append(np.mean(cD_values1))
                
                # Calculate std of det coefs
                cD_std4.append(np.std(cD_values4))
                cD_std3.append(np.std(cD_values3))
                cD_std2.append(np.std(cD_values2))
                cD_std1.append(np.std(cD_values1))
                
                # Calculate energy of appr coefs
                cA_Energy4.append(np.sum(np.square(cA_values4)))
                
                # Calculate energy of det coefs
                cD_Energy4.append(np.sum(np.square(cD_values4)))
                cD_Energy3.append(np.sum(np.square(cD_values3)))
                cD_Energy2.append(np.sum(np.square(cD_values2)))
                cD_Energy1.append(np.sum(np.square(cD_values1)))
                
                # Entropy of appr coefs
                Entropy_A4.append(np.sum(np.square(cA_values4) * np.log(np.square(cA_values4))))
                
                # Entropy of det coefs
                Entropy_D4.append(np.sum(np.square(cD_values4) * np.log(np.square(cD_values4))))
                Entropy_D3.append(np.sum(np.square(cD_values3) * np.log(np.square(cD_values3))))
                Entropy_D2.append(np.sum(np.square(cD_values2) * np.log(np.square(cD_values2))))
                Entropy_D1.append(np.sum(np.square(cD_values1) * np.log(np.square(cD_values1))))
    
                
                ''' Hjorth Parameters '''
                hjorth_activity     = np.var(data)
                diff_input          = np.diff(data)
                diff_diffinput      = np.diff(diff_input)
                hjorth_mobility     = np.sqrt(np.var(diff_input)/hjorth_activity)
                hjorth_diffmobility = np.sqrt(np.var(diff_diffinput)/np.var(diff_input))
                hjorth_complexity   = hjorth_diffmobility / hjorth_mobility
                 
                ''' Statisctical features'''
                Kurt     = kurtosis(data, fisher = False)
                Skewness = skew(data)
                #Mean     = np.mean(data)
                #Median   = np.median(data)
                #Std      = np.std(data)
                ''' Coefficient of variation '''
                #coeff_var = Std / Mean
                
                ''' First and second difference mean and max '''
                sum1  = 0.0
                sum2  = 0.0
                Max1  = 0.0
                Max2  = 0.0
                for j in range(len(data)-1):
                        sum1     += abs(data[j+1]-data[j])
                        first_diff[j] = abs(data[j+1]-data[j])
                        
                        if first_diff[j] > Max1: 
                            Max1 = first_diff[j] # fi
                            
                for j in range(len(data)-2):
                        sum2 += abs(first_diff[j+1]-first_diff[j])
                        if abs(first_diff[j+1]-first_diff[j]) > Max2 :
                        	Max2 = first_diff[j+1]-first_diff[j] 
                            
                diff_mean1 = sum1 / (len(data)-1)
                diff_mean2 = sum2 / (len(data)-2) 
                diff_max1  = Max1
                diff_max2  = Max2
                
                ''' Variance and Mean of Vertex to Vertex Slope '''
                t_max   = argrelextrema(data, np.greater)[0]
                amp_max = data[t_max]
                t_min   = argrelextrema(data, np.less)[0]
                amp_min = data[t_min]
                tt      = np.concatenate((t_max,t_min),axis=0)
                if len(tt)>0:
                    tt.sort() #sort on the basis of time
                    h=0
                    amp = np.zeros(len(tt))
                    res = np.zeros(len(tt)-1)
                    
                    for l in range(len(tt)):
                            amp[l] = data[tt[l]]
                            
                    out = np.zeros(len(amp)-1)     
                     
                    for j in range(len(amp)-1):
                        out[j] = amp[j+1]-amp[j]
                    amp_diff = out
                    
                    out = np.zeros(len(tt)-1)  
                    
                    for j in range(len(tt)-1):
                        out[j] = tt[j+1]-tt[j]
                    tt_diff = out
                    
                    for q in range(len(amp_diff)):
                            res[q] = amp_diff[q]/tt_diff[q] #calculating slope        
                    
                    slope_mean = np.mean(res) 
                    slope_var  = np.var(res)   
                else:
                    slope_var, slope_mean = 0, 0
                    
                ''' Spectral mean '''
                # Spectral_mean = 1 / (freq_ix['Beta'][-1] - freq_ix['Delta'][0]) * (Pow_Delta + 
                #         Pow_Theta_low + Pow_Theta_high + Pow_Alpha + Pow_Beta + 
                #         Pow_Sigma) 
                """ 
                ''' Correlation Dimension Feature '''
                try:
                    cdf = nolds.corr_dim(data,1)
                except np.linalg.LinAlgError:
                    cdf = np.NaN
                  
                """ 
                
                ''' Hurst component '''
                try:
                    Hurst = pyeeg.hurst(data)
                except np.linalg.LinAlgError:
                    Hurst = np.NaN
                    
                ''' Detrended Fluctuation Analysis ''' 
                try:
                    DFA = pyeeg.dfa(data)
                except np.linalg.LinAlgError:
                    DFA = np.NaN
                
                '''Compute Petrosian Fractal Dimension '''
                try: 
                    PFD = pyeeg.pfd(data, D=None)
                except np.linalg.LinAlgError:
                    PFD = np.NaN
                
                '''Waveform length(WL)'''
                WL = sum(abs(np.diff(data)))
                
                '''Zerocrossing(ZC) '''
                zero_crossings = np.where(np.diff(np.signbit(data)))[0]
                num_ZC = (len(zero_crossings))
                
                ''' Mean absolute value ''' 
                MAV = sum(np.abs(data)) / len(data)
                
                '''Simple Square Integral (SSI)'''
                SSI = sum(np.abs(data)**2)
                
                ''' Root mean square '''
                rms = np.sqrt(1 / len(data) * sum(np.abs(data)**2))
    
                ''' Spectral edge frequency features --> SEF50 and SEF95'''
                # Imtiaz et al. proposed the freq band of investigation: 8 - 16 Hz
                data_SEF =  self.butter_bandpass_filter(data=data, lowcut=8, highcut=16, fs=Fs, order=2)
                
                # compute fft^2
                FFT_ = abs(fft(data, n = None))
                FFT_ = FFT_[0:int(len(FFT_)/2)+1] 
                FFT_ = abs(FFT_ ** 2)
                
                # Compute frequency samples
                freq_fft, _ = periodogram(x = data_SEF, fs = Fs, nfft = None , window = window)  
                
                # Defining accumulative and total power
                acc_pow = 0
                tot_pow = 0
                
                # Calculating summation of all powers
                for i,j in enumerate(freq_fft):
                    tot_pow = tot_pow + FFT_[i]
                
                # defining SEF50
                for i,j in enumerate(freq_fft):
                    acc_pow = acc_pow + FFT_[i]    
                    if acc_pow >= .5 * tot_pow:
                        SEF50 = j
                        break
                    
                # Defining SEF 95   
                acc_pow = 0
                for i,j in enumerate(freq_fft):
                    acc_pow = acc_pow + FFT_[i]    
                    if acc_pow >= .95 * tot_pow:
                        SEF95 = j
                        break
                del acc_pow, tot_pow
                
                ''' Spectral edge frequency features --> SEFd'''
                
                # definig subepochs : create window size of 3 secs
                time_win     = duration
                samp_per_win = int(Fs * time_win)
                
                # each column is a subepoch of 2s
                data_SEF_per_win = np.reshape(data_SEF, (samp_per_win, int(len(data_SEF) / samp_per_win)), order='F' )
                
                SEFds = []
                for j in np.arange(0, np.shape(data_SEF_per_win)[1]):
                    
                    # Calculating fft
                    data_tmp = data_SEF_per_win[:,j]
                    C        = abs(fft(data_tmp, n = 512))
                    C        = C[0:int(len(C)/2)+1] 
                    C        = abs(C ** 2)
                    freqs, _ = periodogram(x = data_tmp, fs = Fs, nfft = 512 , window = window)  
                    
                    # Calculating SEFds
                    acc_pow = 0
                    tot_pow = 0
                    # Calculating summation of all powers
                    for i,j in enumerate(freqs):
                        tot_pow = tot_pow + C[i]
                    
                    # defining SEF50
                    for i,j in enumerate(freqs):
                        acc_pow = acc_pow + C[i]    
                        if acc_pow >= .5 * tot_pow:
                            SEF50_tmp = j
                            break
                    # Defining SEF 95   
                    acc_pow = 0
                    for i,j in enumerate(freqs):
                        acc_pow = acc_pow + C[i]    
                        if acc_pow >= .95 * tot_pow:
                            SEF95_tmp = j
                            break
                    SEFd_tmp = SEF95_tmp - SEF50_tmp
                    # Comuting SEFds array
                    SEFds.append(SEFd_tmp)
                    del acc_pow, tot_pow, SEF95_tmp, SEF50_tmp
                    
                SEFd = 1 / len(SEFds)  * sum(SEFds)
                
                ''' Wrapping up featureset '''
                # feat = [pow_total, Pow_Delta, Pow_Theta_low, Pow_Theta_high, Pow_Alpha,
                #         Pow_Beta, Pow_Sigma, Pow_Sigma_slow, Pow_Delta_ratio, Pow_Theta_low_ratio, 
                #         Pow_Theta_high_ratio, Pow_Alpha_ratio,
                #         Pow_Beta_ratio, Pow_Sigma_ratio, Pow_Sigma_slow_ratio, Pow_Gamma_low_ratio, Pow_Gamma_high_ratio,
                #         cA_mean4[0], cA_std4[0],
                #         cD_mean4[0],cD_mean3[0], cD_mean2[0], cD_mean1[0], cD_std4[0],
                #         cD_std3[0], cD_std2[0], cD_std1[0], cA_Energy4[0], cD_Energy4[0],
                #         cD_Energy3[0], cD_Energy2[0], cD_Energy1[0], Entropy_A4[0],
                #         Entropy_D4[0], Entropy_D3[0], Entropy_D2[0], Entropy_D1[0],
                #         Entropy_Welch, Entropy_fft, Kurt, Skewness, Mean, Median, 
                #         Spectral_mean, hjorth_activity, hjorth_mobility, 
                #         hjorth_complexity, Std, coeff_var, diff_mean1, diff_mean2, 
                #         diff_max1, diff_max2, slope_mean, slope_var, Pow_welch_Total, 
                #         Pow_welch_Delta, Pow_welch_Theta_low, Pow_welch_Theta_high, 
                #         Pow_welch_Alpha, Pow_welch_Beta, Pow_welch_Sigma, Pow_welch_Sigma_slow,
                #         Pow_welch_Gamma_low, Pow_welch_Gamma_high,
                #         Pow_welch_Delta_rel, Pow_welch_Theta_low_rel, Pow_welch_Theta_high_rel, 
                #         Pow_welch_Alpha_rel, Pow_welch_Beta_rel, Pow_welch_Sigma_rel, 
                #         Pow_welch_Sigma_slow_rel, Pow_welch_Gamma_low_rel, Pow_welch_Gamma_high_rel,
                #         SEF50, SEF95, SEFd, PFD, Hurst,
                #         WL, num_ZC, MAV,  SSI, rms]
                feat = [cA_mean4[0], cA_std4[0], cD_mean4[0],cD_mean3[0], cD_mean2[0], cD_mean1[0], cD_std4[0],
                        cD_std3[0], cD_std2[0], cD_std1[0], cA_Energy4[0], cD_Energy4[0],
                        cD_Energy3[0], cD_Energy2[0], cD_Energy1[0], Entropy_A4[0],
                        Entropy_D4[0], Entropy_D3[0], Entropy_D2[0], Entropy_D1[0],
                        Entropy_Welch, Entropy_fft, Kurt, Skewness, 
                        hjorth_activity, hjorth_mobility, hjorth_complexity, diff_mean1, diff_mean2, 
                        diff_max1, diff_max2, slope_mean, slope_var, SEF50, SEF95, SEFd, PFD, Hurst,
                        WL, num_ZC, MAV,  SSI, rms]
                
                tempFeatureSet[jjj] = np.array(feat)
                
                end = time.time()
                print(end - start)
                
            featureSet[iii] = np.mean(tempFeatureSet, 0)
            print('Sample Averaging Completed!')
            all_feature_set[iii] = tempFeatureSet
            #diff_feature_set[iii] = tempFeatureSet[0:6] - tempFeatureSet[9:15]
            diff_feature_set[iii] = tempFeatureSet[new_order[0:5]] - tempFeatureSet[new_order[-5:]]
            
                # Features = np.row_stack((Features,feat))
        
        #==== Change NaN Values ====
        aa, bb = np.where(np.isnan(featureSet))
        for j in np.arange(int(len(aa))):
            featureSet[aa[j],bb[j]] = np.nanmean(featureSet[:,bb[j]])
        #==== Change NaN Values ====

        #==== Change NaN Values ====
        all_feature_set = all_feature_set.reshape(sampleAmount, amountofchannel*43)
        aa, bb = np.where(np.isnan(all_feature_set))
        for j in np.arange(int(len(aa))):
            all_feature_set[aa[j],bb[j]] = np.nanmean(all_feature_set[:,bb[j]])
        #==== Change NaN Values ====
        
        #==== Change NaN Values ====
        diff_feature_set = diff_feature_set.reshape(sampleAmount, 5*43)
        aa, bb = np.where(np.isnan(diff_feature_set))
        for j in np.arange(int(len(aa))):
            diff_feature_set[aa[j],bb[j]] = np.nanmean(diff_feature_set[:,bb[j]])
        #==== Change NaN Values ====
            
        # if(featurenorm == True):
        #     featureSet_normed = zscore(featureSet, axis=0)
        #     return featureSet, featureSet_normed  
        # else:
            
        return featureSet, all_feature_set, diff_feature_set
        #%% Normalizing features        

#%% ============ LRLR Marker Selection & Cut =============
    '''self.EEG_cutter_saver(file_path=file_path, saving_directory=saving_directory, explanation='xxxx'',\
                             channel_names=['EOG-1','EOG-2','F3'], channel_indexes=np.array([1,3,4]),\
                             time_interval_seconds=np.array([[56785,56788],[23456,23459]]), low_cut=0.1, high_cut=30) '''
    def EEG_cutter_saver(self, file_path, time_interval_seconds, channel_indexes=None,\
                         saving_directory=None, explanation=None, channel_names=None, low_cut=0.1, high_cut=30):
        
        #==== Initialize ======
        data = mne.io.read_raw_edf(file_path)
        Fs = data.info['sfreq']
        data = data.get_data()
        nyquist = Fs / 2
        #==== Initialize ======
        
        #==== Pre-process =========
        if(high_cut < nyquist): #it should be less than nyquist frequency
            raw_data = self.butter_bandpass_filter(data=data, lowcut=low_cut, highcut=high_cut, fs=Fs)
        if(nyquist > 50): #notch filter at 50 Hz
            raw_data = self.notchFilter(data=raw_data, Fs=Fs, f0=50, Q=30)
        #==== Pre-process =========
        
        custom_data = np.empty(shape=[0,np.shape(raw_data)[1]])
        if(channel_indexes is None):
            custom_data = data
        else:
            custom_data = np.row_stack((custom_data, raw_data[channel_indexes,:]))

        seconds = time_interval_seconds * Fs
        seconds = seconds.astype(int)        
        eeg_chunk = list()
        for i in range(len(time_interval_seconds)): #how many cut operation will be done!
            #==== Cutter =====
            temp_data = custom_data[:, seconds[i,0]: seconds[i,1]]
            eeg_chunk.append(temp_data)
            #==== Cutter =====
                
            if(saving_directory is not None):
                #==== Save Figure =====
                time_interval = len(temp_data[0]) / Fs
                time_interval = np.linspace(0, time_interval, num=len(temp_data[0]))  # seconds
                
                color_chunk = ("red", "green", "yellow", "darkorchid", "cyan")
                plt.figure()
                for i in range(len(channel_indexes)):
                    plt.plot(time_interval, temp_data[i], color=color_chunk[i], linewidth=2)
                
                plt.legend(channel_names, prob={'size':20})
                plt.xlabel('Time [Seconds]', size=25)
                plt.ylabel('Amplitude [uV]', size=25)
                plt.title(explanation)
        
                #=== Maximize ====
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(32, 18)
                #=== Maximize ====
                plt.show()
                
                plt.savefig(saving_directory + '/' + explanation, pad_inches=0, bbox_inches='tight', dpi=400)
                print('Figure has saved successfully!')
                #===== Save Figure ======
                plt.close()
 
        return eeg_chunk
    
    def multi_file_EEG_cutter_saver(self, folder_path, time_interval_seconds, channel_indexes=None, selected_data=None, \
                                    file_explanations=None, saving_directory=None, explanation=None, channel_names=None, \
                                    low_cut=0.1, high_cut=30):
        
        files = list()
        for file in os.listdir(folder_path):
            if file.endswith(".edf"):
               files.append(file)
            
        if(selected_data is not None): #for taking only sub-set of all dataset
            files = [files[i] for i in selected_data] #get only selected files
               
        if(file_explanations is None):
            file_explanations = np.empty(shape=[0,1], dtype=str)
            for i in range(len(files)):
                file_explanations = np.append(file_explanations, files[i].split('.edf')[0])
        
        
        total_periods = dict()
        if(channel_indexes is not None):
            if(len(channel_indexes.shape) > 1): #none channel index has no shape!
                for i in range(len(files)): #same channels are in different indexes for different files
                    temp_periods = self.EEG_cutter_saver(file_path=folder_path + '/' + files[i], channel_indexes=channel_indexes[i],\
                                                         time_interval_seconds=time_interval_seconds[i], saving_directory=saving_directory,\
                                                         explanation=explanation, channel_names=channel_names,\
                                                         low_cut=low_cut, high_cut=high_cut)
                    total_periods[file_explanations[i]] = temp_periods 
                    
        else:
            for i in range(len(files)):
                temp_periods = self.EEG_cutter_saver(file_path=folder_path + '/' + files[i], channel_indexes=channel_indexes, \
                                                     time_interval_seconds=time_interval_seconds[i], saving_directory=saving_directory,\
                                                     explanation=explanation, channel_names=channel_names,\
                                                     low_cut=low_cut, high_cut=high_cut)
                total_periods[file_explanations[i]] = temp_periods 

        return total_periods
        
 
    def LRLRMarker_cut_data_by_time_interval(self, folder_path, LRLR_timeIntervals, explanation, \
                                             eog_indexes, low_cut=0.1, high_cut=30, saving_directory=None):
        
        files = list()
        for file in os.listdir(folder_path):
            if file.endswith(".edf"):
               files.append(file)
                
        #===== Definitions =====
        allFs = list()
        dataChannels = list()
        dataSamplingRates = list()
        lengthSeconds = list()
        count = len(files)
        dataInfos = list()
        dataSets = list()
        #===== Definitions =====
        
        #==== Read EDF Files =======
        for i in range(count):
            data = mne.io.read_raw_edf(folder_path + '/' + files[i])
            dataSets.append(data)
            dataInfo = data.info
            dataInfos.append(dataInfo)
            dataChannels.append(dataInfo['ch_names'])
            dataSamplingRates.append(int(dataInfo['sfreq']))
            lengthSeconds.append(len(list(data[0])[0].flatten()) / dataInfo['sfreq'])
            
            allFs.append(dataInfo['sfreq'])
        #==== Read EDF Files =======
        
        #===== LRLR Marker ========
        lrlr_chunk = list()
        for i in range(count):
            T = 30 #secs
            len_epoch   = int(allFs[i] * T)
            temp_raw_data = dataSets[i].get_data()
            temp_raw_data = self.butter_bandpass_filter(data=temp_raw_data, lowcut=low_cut, highcut=high_cut, fs=dataSamplingRates[i])
            temp_raw_data = temp_raw_data[:, 0:temp_raw_data.shape[1] - temp_raw_data.shape[1] % len_epoch] #cut the tail
        
            #===== Seconds ========
            if(np.size(eog_indexes) > 2):
                temp_raw_data_EOGs = np.row_stack((temp_raw_data[eog_indexes[i,0],:], temp_raw_data[eog_indexes[i,1],:]))
            else:
                temp_raw_data_EOGs = np.row_stack((temp_raw_data[eog_indexes[0],:], temp_raw_data[eog_indexes[1],:]))
                
            for j in range(len(LRLR_timeIntervals[i])):
                seconds = np.array([LRLR_timeIntervals[i][j,0], LRLR_timeIntervals[i][j,1]]) * allFs[i]
                seconds = seconds.astype(int)
                ld_mark = temp_raw_data_EOGs[:,seconds[0]:seconds[1]]
                lrlr_chunk.append(ld_mark)
            #===== Seconds ========
            
                if(saving_directory is not None):
                    #==== Plot Figure =====
                    time_interval = len(ld_mark[0]) / allFs[i]
                    time_interval = np.linspace(0, time_interval, num=len(ld_mark[0]))  # seconds
                    
                    plt.figure()
                    plt.plot(time_interval, ld_mark[0] + (max(np.abs(ld_mark[0])) - min(np.abs(ld_mark[1]))),\
                             color='blue', linewidth=2)
                    plt.plot(time_interval, ld_mark[1], color='red', linewidth=2)
                    plt.title('LRLR Marker', size=30)
                    plt.legend(['EOG-0', 'EOG-1'], prop={'size': 20})
                    plt.xlabel('Time [Seconds]', size=25)
                    plt.ylabel('Amplitude [uV]', size=25)
                    #==== Plot Figure =====
                    
                    #=== Maximize ====
                    figure = plt.gcf()  # get current figure
                    figure.set_size_inches(32, 18)
                    plt.show()
                    #=== Maximize ====
                                        
                    #===== Save Figure ======
                    if(saving_directory != None):
                   
                       plt.savefig(saving_directory + '/' + explanation + '_' + files[i].split('.edf')[0] + '_' + str(j), pad_inches=0, bbox_inches='tight', dpi=400)
                       print('Figure has saved successfully!')
                       #===== Save Figure ======
                       plt.close()
                    #===== Save Figure ======
                
        return lrlr_chunk
#%% =========== Signal Processing Methods ====================   
    def bandpower(data, sf, band, window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band.
    
        Parameters
        ----------
        data : 1d-array
            Input signal in the time-domain.
        sf : float
            Sampling frequency of the data.
        band : list
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative : boolean
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.
    
        Return
        ------
        bp : float
            Absolute or relative band power.
        """
    
        band = np.asarray(band)
        low, high = band
    
        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
    
        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data, sf, nperseg=nperseg)
    
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
    
        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
    
        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)
    
        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp
    
    def welch_periodogram_calculation(self, data_array, Fs):
        
        brainwave_names = ['Delta', 'Theta_low', 'Theta_high', 'Alpha', 'Beta', 'Mu', 'Sigma', 'Sigma_slow',\
                           'Gamma', 'High_gamma']
        
        # Defining EEG bands:
        eeg_bands = {brainwave_names[0] : (0.5, 4), #Delta
                     brainwave_names[1] : (4  , 6), #Theta_low
                     brainwave_names[2] : (6  , 8), #Theta_high
                     brainwave_names[3] : (8  , 12), #Alpha
                     brainwave_names[4] : (16 , 24), #Beta
                     brainwave_names[5] : (7.5, 12.5), #Mu
                     brainwave_names[6] : (12 , 15), #Sigma
                     brainwave_names[7] : (10 , 12), #Sigma_slow
                     brainwave_names[8] : (25 , 39), #Gamma
                     brainwave_names[9] : (40, 50)} #High_gamma
             
        particular_gamma_freq = 40 #Hz
        
        #======= Power Calculation =========
        all_data_band_powers = list()
        all_data_band_powers_relative = list() #relative band powers E.g : delta_power / all_power
        
        for i in range(len(data_array)):
            temp_all_band_powers = np.empty(shape=[0,1])
            temp_all_band_powers_relative = np.empty(shape=[0,1])
            for j in range(len(brainwave_names)):
                temp_power = self.bandpower(data=data_array[i], sf=Fs, band=eeg_bands[brainwave_names[j]], \
                                             window_sec=1, relative=False)
                temp_power_relative = self.bandpower(data=data_array[i], sf=Fs, band=eeg_bands[brainwave_names[j]], \
                                  window_sec=1, relative=True)
                    
                temp_all_band_powers = np.append(temp_all_band_powers, temp_power)
                temp_all_band_powers_relative = np.append(temp_all_band_powers_relative, temp_power_relative)
                
            all_data_band_powers.append(temp_all_band_powers)
            all_data_band_powers_relative.append(temp_all_band_powers_relative)
        #======= Power Calculation =========
        
        return all_data_band_powers, all_data_band_powers_relative
        
    def spectrogram_creation(self, data_array, Fs, explanation='', plot=False, saving_directory=None, nperseg=256, \
                             noverlap=None, nfft=None, average=False):
        # nperseg int(Fs*30)
        
        dimension = len(data_array.shape)
        
        if(average == True):
            
            avg_spectrogram = 0
            for i in range(len(data_array)):
                f, t, Sxx = spectrogram_lspopt(x=data_array[i], fs=Fs, c_parameter=20.0, nperseg=nperseg, \
                                               scaling='density', noverlap=noverlap, nfft=nfft)
                Sxx = 10 * np.log10(Sxx) #power to db
                avg_spectrogram += Sxx
                
            avg_spectrogram /= len(data_array)
            
            if(plot):
                #==== 1st Way =======
                plt.figure()
                ax = plt.axes()
                plt.pcolormesh(t, f, Sxx)
                plt.ylabel('Frequency [Hz]', size=20)
                plt.xlabel('Time [sec]', size=20)
                plt.title(explanation + ' Multi-taper Spectrogram', size=25)
                ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
                
                plt.colorbar()
                #==== 1st Way =======
                
                #=== Maximize ====
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(32, 18)
                plt.show()
                #=== Maximize ====
                
                #===== Save Figure ======
                if(saving_directory != None):
                
                   plt.savefig(saving_directory + '/' + explanation + '_' + str(i), pad_inches=0, \
                               bbox_inches='tight', dpi=400)
                   print('Figure has saved successfully!')
                   plt.close()
                #===== Save Figure ======
            
            return avg_spectrogram
            
        else:
            
            spectrogam_list = list()
            if(dimension == 2):
                for i in range(len(data_array)):
                    f, t, Sxx = spectrogram_lspopt(x=data_array[i], fs=Fs, c_parameter=20.0, nperseg=nperseg, \
                                                   scaling='density', noverlap=noverlap, nfft=nfft)
                    Sxx = 10 * np.log10(Sxx) #power to db
                    spectrogam_list.append(Sxx)
                    
                    if(plot):
                        #==== 1st Way =======
                        plt.figure()
                        ax = plt.axes()
                        plt.pcolormesh(t, f, Sxx)
                        plt.ylabel('Frequency [Hz]', size=20)
                        plt.xlabel('Time [sec]', size=20)
                        plt.title(explanation + ' Multi-taper Spectrogram', size=25)
                        ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
                        
                        plt.colorbar()
                        #==== 1st Way =======
                        
                        #=== Maximize ====
                        figure = plt.gcf()  # get current figure
                        figure.set_size_inches(32, 18)
                        plt.show()
                        #=== Maximize ====
                        
                        #===== Save Figure ======
                        if(saving_directory != None):
                        
                           plt.savefig(saving_directory + '/' + explanation + '_' + str(i), pad_inches=0, \
                                       bbox_inches='tight', dpi=400)
                           print('Figure has saved successfully!')
                           plt.close()
                        #===== Save Figure ======
                        
            elif(dimension == 1):
                
                f, t, Sxx = spectrogram_lspopt(x=data_array, fs=Fs, c_parameter=20.0, nperseg=nperseg, \
                                               scaling='density', noverlap=noverlap, nfft=nfft)
                Sxx = 10 * np.log10(Sxx) #power to db
                spectrogam_list.append(Sxx)
                
                if(plot):
                    #==== 1st Way =======
                    plt.figure()
                    ax = plt.axes()
                    plt.pcolormesh(t, f, Sxx)
                    plt.ylabel('Frequency [Hz]', size=20)
                    plt.xlabel('Time [sec]', size=20)
                    plt.title(explanation + ' Multi-taper Spectrogram', size=25)
                    ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
                    
                    plt.colorbar()
                    #==== 1st Way =======
                       
            return spectrogam_list
        
    def spectrum_3Methods(self, data, sf, window_sec, band=None, dB=False, plot=False, saving_directory=None,\
                          explanation=''):
        """Plot the periodogram, Welch's and multitaper PSD.
    
        Requires MNE-Python >= 0.14.
    
        Parameters
        ----------
        data : 1d-array
            Input signal in the time-domain.
        sf : float
            Sampling frequency of the data.
        band : list
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds for Welch's PSD
        dB : boolean
            If True, convert the power to dB.
        """

        sns.set(style="white", font_scale=1.2)
        # ===== Compute the PSD ==========
        freqs, psd = periodogram(data, sf)
        freqs_welch, psd_welch = welch(data, sf, nperseg=window_sec*sf)
        psd_mt, freqs_mt = psd_array_multitaper(data, sf, adaptive=True,
                                                normalization='full', verbose=0)
        # ===== Compute the PSD ==========
        sharey = False
    
        # Optional: convert power to decibels (dB = 10 * log10(power))
        if dB:
            psd = 10 * np.log10(psd)
            psd_welch = 10 * np.log10(psd_welch)
            psd_mt = 10 * np.log10(psd_mt)
            sharey = True
    
        if(plot == True):
            # ========== Start Plot =======================
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=sharey)
            # Stem
            sc = 'slategrey'
            ax1.stem(freqs, psd, linefmt=sc, basefmt=" ", markerfmt=" ")
            ax2.stem(freqs_welch, psd_welch, linefmt=sc, basefmt=" ", markerfmt=" ")
            ax3.stem(freqs_mt, psd_mt, linefmt=sc, basefmt=" ", markerfmt=" ")
            # Line
            lc, lw = 'k', 2
            ax1.plot(freqs, psd, lw=lw, color=lc)
            ax2.plot(freqs_welch, psd_welch, lw=lw, color=lc)
            ax3.plot(freqs_mt, psd_mt, lw=lw, color=lc)
            # Labels and axes
            ax1.set_xlabel('Frequency (Hz)')
            if not dB:
                ax1.set_ylabel('Power spectral density (V^2/Hz)')
            else:
                ax1.set_ylabel('Decibels (dB / Hz)')
            ax1.set_title('Periodogram')
            ax2.set_title('Welch')
            ax3.set_title('Multitaper')
            if band is not None:
                ax1.set_xlim(band)
            ax1.set_ylim(ymin=0)
            ax2.set_ylim(ymin=0)
            ax3.set_ylim(ymin=0)
            sns.despine()
            # ========== Start Plot =======================
            
            #===== Save Figure ======
            if(saving_directory != None):
            
               plt.savefig(saving_directory + '/' + explanation, pad_inches=0, \
                           bbox_inches='tight', dpi=400)
               print('Figure has saved successfully!')
            #===== Save Figure ======
            plt.close()
            
        return {'Periodogram' : psd, 'Welch' : psd_welch, 'Multitaper' : psd_mt}
    
    def nextpow2(self, p):     
        n = 2
        power=1
        while p > n:  
            n *= 2
            power += 1
        return power
#%% ============= MNE Functions =========

## ================== EEG File Reading ==============================
    def read_edf_file(self, file_path, preload=False):
        raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=False)
        print('EDF reading completed!')
        return raw
    
    def read_brainvision_file(self, file_path, preload=True):
        raw = mne.io.read_raw_brainvision(vhdr_fname=file_path, preload=preload, verbose=False)
        print('Brainvision File reading completed!')
        return raw
#%% ==================== Event / Epoch Creation ===============        
    def event_epoch_creation(self, raw, event_id, period_interval, duration, picks, overlap, tmin=0, tmax = None, baseline=(0, 0.2), \
                             single_event_id = None, XDAWN = False, crop_length=0):

        if(tmax is None):
           tmax = 4 - 1. / raw.info['sfreq']

        # ====== Event Creation ======= 
        total_events = np.empty(shape=[0,3])
        for i in range(len(period_interval)):
                
            if(np.size(period_interval[i,0]) > 1): #if there are multiple intervals in a file
                for j in range(len(period_interval[i])):
                    temp_event = mne.make_fixed_length_events(raw=raw, id=i, start=period_interval[i,j][0]-crop_length, \
                                 stop=period_interval[i,j][1]-crop_length, duration=duration, overlap=overlap)
                    total_events = np.row_stack((total_events, temp_event))
            else:
                if(single_event_id is not None):
                    temp_event = mne.make_fixed_length_events(raw=raw, id=single_event_id, start=period_interval[i,0] - crop_length, 
                                                              stop=period_interval[i,1] - crop_length, \
                                                              duration=duration, overlap=overlap)
                else:
                    temp_event = mne.make_fixed_length_events(raw=raw, id=i, start=period_interval[i,0] - crop_length, 
                                                              stop=period_interval[i,1] - crop_length, \
                                                              duration=duration, overlap=overlap)
                total_events = np.row_stack((total_events, temp_event))
        # ====== Event Creation ======= 
        
        #===== Epoching ======
        epochs = mne.Epochs(raw=raw, events=total_events.astype(int), picks=picks, tmin=tmin, tmax=tmax, event_id=event_id, \
                            preload=True, baseline=baseline, proj=True, verbose=False)
        # print(epochs.info)
        print('Epoching Completed')
        #===== Epoching ======
        
        if(XDAWN == True):
            signal_cov = compute_raw_covariance(raw)
        
            xd = Xdawn(n_components=2, signal_cov=signal_cov) # Xdawn instance
            xd.fit(epochs) # Fit xdawn
            epochs = xd.apply(epochs)
        
        return epochs, total_events.astype(int)
    
    def event_epoch_creation_rawconcattype(self, raw, event_id, period_interval, duration, picks, overlap, tmin=0, \
                                           tmax = None, baseline=(0, 0.2)):

        if(tmax is None):
           tmax = 4 - 1. / raw.info['sfreq']
          
        # ====== Event Creation ======= 
        total_events = np.empty(shape=[0,3])
        for j in range(len(period_interval)):
            for i in range(len(event_id)):
                temp_event = mne.make_fixed_length_events(raw=raw, id=i, start=period_interval[j,i,0], stop=period_interval[j,i,1], \
                                                          duration=duration, overlap=overlap)
                total_events = np.row_stack((total_events, temp_event))
        # ====== Event Creation ======= 
        
        #===== Epoching ======
        epochs = mne.Epochs(raw=raw, events=total_events.astype(int), picks=picks, tmin=tmin, tmax=tmax, event_id=event_id, \
                            preload=True, baseline=baseline, proj=True, verbose=False)
        # print(epochs.info)
        print('Epoching Completed')
        #===== Epoching ======
        
        return epochs, total_events.astype(int)
#%% ========================= MNE Preprocessing ==================
    def sensor_location_update(self, raw, renamed_channels=None, channel_types=None, picked_channels=None, pick_types=pick_types, \
                               montage=None):

        '''Input information example:
        - renamed_channels = {'EEG F3-A2':'F3', 'EEG F4-A1':'F4'} 
        - channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F3':'eeg'}
        - picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4']
        - pick_types = ['eeg','eog']
        '''

        # === Setup layout ====
        # montage_list = mne.channels.get_builtin_montages() #all montages list
        # layout = mne.channels.read_layout('EEG1005')
        # === Setup layout ====
        
        # ======== Change Channel Type & Name & Picking channels =======
        if(renamed_channels is not None):
            raw.rename_channels(mapping=renamed_channels)
        if(picked_channels is not None):
            raw.pick_channels(ch_names=picked_channels)
        if(channel_types is not None):
            raw.set_channel_types(mapping=channel_types)
        # if(pick_types is not None):
        #     raw.pick_types(eeg=True, eog=True, ecg=True, emg=True)
        
        print(raw.info)
        
        if(montage is not None):
            ten_twenty_montage = mne.channels.make_standard_montage(montage) #this is chosen
            raw.set_montage(ten_twenty_montage) #final aim is to make it valid
        # ======== Change Channel Type & Name =======
        
        return raw        
    
    def LUCIRETA_sensor_location_update(self, raw, count, drop_channels=None):
        
        #Mistake Correction
        if(count==0):
            raw.rename_channels(mapping={'FFC!h':'FFC1h'})
        
        montage = mne.channels.make_standard_montage('standard_1005') #this is chosen
        montage_chnames = montage.ch_names
       
        ch_names = raw.info['ch_names']
        
        #===== Find Channel Indexes to Select ========
        ch_indexes = np.zeros(len(ch_names)) - 1
        ch_indexes = ch_indexes.astype(int)
        to_be_deleted = list()
        for i in range(len(ch_names)):
            try:
                ch_indexes[i] = montage_chnames.index(ch_names[i])
            except:
                # print(ch_names[i])
                to_be_deleted.append(i)
                
        ch_indexes = np.delete(ch_indexes, to_be_deleted)
        ch_names = np.delete(ch_names, to_be_deleted)
        ch_names = ch_names.tolist()
        #===== Find Channel Indexes to Select ========

        #=== montage update ======
        montage.dig = montage.dig[3:]
        montage.ch_names = [montage.ch_names[index] for index in ch_indexes]
        montage.dig = [montage.dig[index] for index in ch_indexes]
                  
        raw.drop_channels(ch_names=['FFC1h','E','SI5','SI3','SI6','SI4','IIz','Events/Markers','EEG Mark1','EEG Mark2'])
        # raw.set_eeg_reference(ref_channels=['TP7', 'TP9'])
        raw.set_channel_types({'IO':'eog','ECG':'ecg','SM1':'emg','SM2':'emg','SM3':'emg'})
        raw.set_montage(montage)
        
        raw.drop_channels(ch_names=['F1','F10','F8','F9','FFC3h', 'FFC5h', 'I2', 'P2', 'PO10', 'PO8'])
        #=== montage update ======     
        
        return raw
        
    def ICA(self, raw, eog_like_channel=None, ecg_like_channel=None, manuel_components=None, n_components=30, \
            auto_eog=False, auto_ecg=False):
        
        raw_c = raw.copy()
        
        if(type(raw_c) == list):
            
            icas = list()
            for i in range(len(raw_c)):
                ica = ICA(n_components=n_components, random_state=97, verbose=False, max_iter='auto', method='infomax')
                ica.fit(raw_c[i])
                icas.append(ica)
                
            if(eog_like_channel is not None):
                eog_inds, eog_scores = ica.find_bads_eog(raw_c, ch_name=eog_like_channel)
                corrmap(icas, template=(0, eog_inds[0]), threshold=0.9, label='blink', plot=False)
                
                #exclude blink artifact IC from all ICs at once by using correlation map that gets template IC from 1st raw data
                for i in range(len(icas)):
                    icas[i].exclude = icas[i].labels_['blink']
                    icas[i].apply(raw_c[i])  
                    
            else: #choosing manuel components to delete for whole raw data
                for i in range(len(icas)):
                    icas[i].exclude = manuel_components
                    icas[i].apply(raw_c[i])
                    
        else: #if there is only one raw data
            
            ica = ICA(n_components=n_components, random_state=97, verbose=False, max_iter='auto', method='infomax')
            ica.fit(raw_c)
            
            if(eog_like_channel is not None): #first check if EOG like channel is not none than check for auto eog
                eog_inds, eog_scores = ica.find_bads_eog(raw_c, ch_name=eog_like_channel)
            elif(auto_eog == True):
                print('Within the auto EOG')
                eog_inds, eog_scores = ica.find_bads_eog(raw_c)
                    
                try:
                    ica.exclude = eog_inds
                    ica.apply(raw_c)
                    print('EOG component is excluded')
                    print(eog_inds)
                except:
                    print('No EOG component found to be excluded')
                
            if(ecg_like_channel is not None): #first check if ECG like channel is not none than check for auto ecg
                ecg_inds, ecg_scores = ica.find_bads_ecg(raw_c, ch_name=ecg_like_channel)
            elif(auto_ecg == True):
                ecg_inds, ecg_scores = ica.find_bads_ecg(raw_c)
                    
                try:
                    ica.exclude = ecg_inds
                    ica.apply(raw_c)
                except:
                    print('No ECG component found to be excluded')
                
            if(eog_like_channel is None and ecg_like_channel is None and manuel_components is not None):
                ica.exclude = manuel_components
                ica.apply(raw_c)
                
        return raw_c
    
    def eeg_read_sensorupdate_file(self, file_path, picks='eeg', count=0, renamed_channels=None, channel_types=None, picked_channels=None, \
                            drop_channels=None, eeg_reference=None, data_format='edf', \
                            Lucireta=False, input_as_list=False, preload=False, montage=None):
        
        # ========================================== Preprocess ========================================================
        
        if(data_format == 'edf'):
            raw = self.read_edf_file(file_path=file_path, preload=preload)
        elif(data_format == 'eeg'):
            raw = self.read_brainvision_file(file_path=file_path)
        elif(data_format == 'raw'):
            raw = file_path
        
        #==== Drop Channel ====
        if(drop_channels is not None):
            raw.drop_channels(ch_names=drop_channels)
        #==== Drop Channel ====
    
        # ===================== Sensor Info Update ==================
        if(Lucireta==False):
            if(input_as_list == True):
                if(renamed_channels is not None or channel_types is not None or picked_channels is not None or picks is not None):
                    raw = self.sensor_location_update(raw=raw, renamed_channels=renamed_channels[count], channel_types=channel_types[count], \
                                                      picked_channels=picked_channels[count], pick_types=picks, montage=montage)
            else:
                if(renamed_channels is not None or channel_types is not None or picked_channels is not None or picks is not None):
                    raw = self.sensor_location_update(raw=raw, renamed_channels=renamed_channels, channel_types=channel_types, \
                                                      picked_channels=picked_channels, pick_types=picks, montage=montage)
        else:
            raw = self.LUCIRETA_sensor_location_update(raw=raw, drop_channels=drop_channels, count=count)
        
        #==== Set EEG Reference ====
        if(eeg_reference is not None):
            raw.set_eeg_reference(ref_channels=eeg_reference)
            print('EEG references are set!')
        #==== Set EEG Reference ====
        
        # ===================== Sensor Info Update ==================
        
        return raw
        
    def eeg_preprocess(self, raw, f_notch=None, f_min=None, f_max=None, ICA=False, SSP=False, CSD=False, resample=None,\
                       eog_projs=None, ecg_projs=None, count=0, unit_normalization=False, period_interval=None, crop=False,\
                       manuel_components=None, auto_eog=False, auto_ecg=False, normalization=None):
        
        if(unit_normalization == True):
            print('Unit normalization is being processed...')
            try:
                raw._data = raw._data / raw._raw_extras[0]['units'][0] #unit normalization
            except:
                raw._data = raw._data / raw._cals[:,None] #unit normalization for brainvision data
                print('Brainvision protocol is applied')
                
        if(crop == True):
            raw.crop(tmin=np.min(period_interval), tmax=np.max(period_interval))
        
        if(resample is not None):
            try:
                raw.resample(resample, npad='auto')
                print('Resampling completed')
            except:
                print('No resampling has been done')
            # if(resample[count] != 0):
            #     raw.resample(resample[count], npad='auto')
                      
        if(f_notch != None):
            raw.notch_filter(f_notch, filter_length='auto', phase='zero', verbose=False)
        
        if(f_min is not None or f_max is not None):
            raw.filter(l_freq=f_min, h_freq=f_max, verbose=False)
            print('Band-pass filtering completed')
        # raw = self.SSP_artifact_removal(raw=raw)
        
        #==== Projectiles =====
        # if(ecg_projs is not None or eog_projs is not None):
            # raw.info['projs'] += eog_projs + ecg_projs
            # raw.apply_proj()
        #==== Projectiles =====
        
        #===== Get Channel Types =====
        eog_indexes = mne.pick_types(raw.info, eog=True)
        eeg_indexes = mne.pick_types(raw.info, eeg=True)
        ecg_indexes = mne.pick_types(raw.info, ecg=True)
        emg_indexes = mne.pick_types(raw.info, emg=True)
        print('Amount of EEG :' + str(len(eeg_indexes)))
        #===== Get Channel Types =====
    
        eog_index_of_first, ecg_index_of_first = 0, 0
        if(count == 0):
            eog_index_of_first = len(eog_indexes)
            ecg_indexes_of_first = len(ecg_indexes)
            
        # ============= Addititonal Preprocessing ===================
        if(ICA == True):
            n_components = np.min((len(eeg_indexes)-1, 30))
            raw = self.ICA(raw=raw, n_components=n_components, manuel_components=manuel_components, auto_eog=auto_eog, auto_ecg=auto_ecg)
        if(SSP == True):
            # if(count == 0): #only create SSP of the 1st Data
            if(len(eog_indexes) > 0):
               eog_projs, _ = mne.preprocessing.compute_proj_eog(raw, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True, verbose=False)
            print('EOG Projs Created')
            if(len(ecg_indexes) > 0):
               ecg_projs, _ = mne.preprocessing.compute_proj_ecg(raw, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True, verbose=False)
            print('ECG Projs Created')
            
            if(len(eog_indexes) > 0):
                # raw.info['projs'] += eog_projs
                raw.add_proj(eog_projs)
            if(len(ecg_indexes) > 0):
                # raw.info['projs'] += ecg_projs
                raw.add_proj(ecg_projs)
                
            if(len(eog_indexes) > 0 or len(ecg_indexes) > 0):
                raw.apply_proj()
        
        # ====== Isolate EEG Channels from EMG & EOG & ECG =======
        # raw.pick_types(eeg=True)
        # ====== Isolate EEG Channels from EMG & EOG & ECG =======
        
        if(CSD == True):
            raw = self.CSD(raw)
        if(normalization == 'zscore'):
            raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)
        elif(normalization == 'robustzscore'):
            raw = self.robustZScore(raw)
        # ============= Addititonal Preprocessing ===================

        # ========================================== Preprocess ========================================================
        
        return raw
        
    def fake_epoching(self, raw, t_begin, t_finish, picks='eeg', baseline=None):
        
        event_id = {'Whole_data':0}
        events =  np.zeros(shape=(1,3)).astype(int)
        events[0,0] = t_begin * raw.info['sfreq']

        time_second = (t_finish - t_begin) - 1 / raw.info['sfreq']
       
        #Deletion
        # raw.del_proj()
        raw.set_annotations(None)
        
        epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=time_second, baseline=baseline, preload=True, reject=None, \
                            reject_by_annotation=None, picks=picks)
            
        print(epochs.drop_log)
        
        # if(tmin is not None and tmax is not None):
        #     epochs.crop(tmin=tmin, tmax=tmax)
        
        return epochs
    
    def epoch_concatenation(self, epochs_list):
        ''' epoch_list : [epochs_0, epochs_1, epochs_2] '''
        epochs = mne.concatenate_epochs(epochs_list)
        return epochs
    
    def factor_statistics_of_file(self, file_path):
        statistics = dict()
        raw = self.read_edf_file(file_path=file_path, preload=True)
        statistics['mean'] = np.mean(raw._data)
        statistics['min'] = np.min(raw._data)
        statistics['max'] = np.max(raw._data)   
        statistics['std'] = np.std(raw._data)
        statistics['absolute_factor'] = raw._raw_extras[0]['physical_max'][0] * raw._raw_extras[0]['units'][0]
        statistics['unit'] = raw._raw_extras[0]['units'][0]
        return statistics
    
    def factor_statistics_of_dataset(self, folder_path):

        files = list()
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".edf"):
                files.append(file)
                       
        statistics = dict()
        statistics['mean'] = np.zeros(len(files))
        statistics['min'] = np.zeros(len(files))
        statistics['max'] = np.zeros(len(files))
        statistics['std'] = np.zeros(len(files))

        count = 0
        for file in files:
            file_path = folder_path + '/' + file
            raw = self.read_edf_file(file_path=file_path, preload=True)        
            statistics['mean'][count] = np.mean(raw._data)
            statistics['min'][count] = np.min(raw._data)
            statistics['max'][count] = np.max(raw._data)   
            statistics['std'][count] = np.std(raw._data)
            count += 1
            
        statistics['grand_mean'] = np.mean(statistics['mean'])
        statistics['grand_min'] = np.mean(statistics['min'])
        statistics['grand_max'] = np.mean(statistics['max'])
        statistics['grand_std'] = np.mean(statistics['std'])
            
        return statistics
    
    def edf_deep_info_retrieval_of_dataset(self, folder_path):
        files = list()
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".edf"):
                files.append(file)
                
        deep_info = dict()
        deep_info['unit'] = list()
        deep_info['file_name'] = list()
        for file in files:
            file_path = folder_path + '/' + file
            raw = self.read_edf_file(file_path=file_path, preload=False)        
            deep_info['unit'].append(raw.info['chs'][0]['unit'])
            deep_info['file_name'].append(file)
            
        deep_info['absolute_factor'] = raw._raw_extras[0]['physical_max'][0] * raw._raw_extras[0]['units'][0]
        deep_info['unit'] = raw._raw_extras[0]['units'][0]
        return deep_info
        
    def eeg_neighbourhoodinterval_of_given_folder(self, folder_path, event_id,\
                                              period_interval, picks='eeg', renamed_channels=None, channel_types=None, drop_channels=None, \
                                              picked_channels=None, eeg_reference=None, duration=4, overlap=2, tmin=0, tmax = None, \
                                              baseline=(0, 0.2), f_min=0.1, f_max=48, f_notch = None, ICA=True, SSP=False, CSD=False, \
                                              Lucireta=False, input_as_list=False, resample=None, single_event_id=None,\
                                              preload=False):
        
        files = list()
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".edf"):
                files.append(file)
        
        count = 0
        
        cropped_raw = list()
        new_periods = list()
        fake_epochs = list()
        # eog_projs, ecg_projs = None, None
        for file in files:
            file_path = folder_path + '/' + file
            
            temp_period = period_interval[count]
            
            temp_period = np.array([temp_period[1], temp_period[1], temp_period[1]]) #elbet ise yarayacak
            difference = temp_period[1,1] - temp_period[1,0]
            temp_period[0] -= difference
            temp_period[2] += difference
        
            raw = self.eeg_preprocess_file(file_path=file_path, renamed_channels=renamed_channels,\
                                       channel_types=channel_types, picks=picks, drop_channels=drop_channels, \
                                       picked_channels=picked_channels, eeg_reference=eeg_reference, \
                                       f_min=f_min, f_max=f_max, f_notch=f_notch, ICA=ICA, SSP=SSP, \
                                       CSD=CSD, Lucireta=Lucireta, count=count, input_as_list=input_as_list,\
                                       resample=resample, preload=preload)
    
            beginner = temp_period[0,0]
            finish = temp_period[2,1]
            print("Interval between {:.2f} and {:.2f}".format(beginner, finish))
            
            raw.pick(picks)
            # raw.crop(tmin=beginner, tmax=finish)
            raw_data = raw.get_data()
            new_periods.append(temp_period)
            cropped_raw.append(raw_data)
            
            count += 1
            
            # === Fake Epoching for Upcoming MNE Functions ====
            fake_epochs.append(self.fake_epoching(raw, tmin=beginner, tmax=finish))
            # === Fake Epoching for Upcoming MNE Functions ====
            
            print('File :' + str(count) + ' Completed!')
            
        return cropped_raw, fake_epochs, new_periods
    
    def eeg_epoching_pipeline_of_given_folder(self, folder_path, event_id,\
                                              period_interval, picks=['eeg'], renamed_channels=None, channel_types=None, drop_channels=None, \
                                              picked_channels=None, eeg_reference=None, duration=4, overlap=2, tmin=0, tmax = None, \
                                              baseline=(0, 0.2), f_min=0.1, f_max=48, f_notch = None, ICA=True, SSP=False, CSD=False, \
                                              XDAWN=False, Lucireta=False, input_as_list=False, resample=None, single_event_id=None,\
                                              preload=False, crop=False, unit_normalization=False, data_format='edf', montage='standard_1020',\
                                              normalization=None, auto_eog=False, auto_ecg=False):
                                              
        files = list()
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".vhdr"):
                files.append(file)
        
        count = 0
        crop_length = 0

        epochs_list = list()
        total_events = np.empty(shape=[0,3])
        eog_projs, ecg_projs = None, None
        
        #======= Other Info =========
        other_info = dict()
        other_info['projs'] = list()
        #======= Other Info =========
        
        for file in files:
            file_path = folder_path + '/' + file
            
            print('Epoching : ' + file + ' has begun')
            
            raw = self.eeg_read_sensorupdate_file(file_path=file_path, renamed_channels=renamed_channels,\
                                                  channel_types=channel_types, picks=picks, drop_channels=drop_channels, \
                                                  picked_channels=picked_channels, eeg_reference=eeg_reference, \
                                                  Lucireta=Lucireta, count=count, input_as_list=input_as_list,\
                                                  preload=preload, data_format=data_format, montage=montage)
                
            raw = self.eeg_preprocess(raw=raw, f_notch=f_notch, f_min=f_min, f_max=f_max, ICA=ICA, SSP=SSP, CSD=CSD, resample=resample,\
                                      eog_projs=eog_projs, ecg_projs=ecg_projs, count=0, period_interval=period_interval[count], crop=crop,\
                                      unit_normalization=unit_normalization, normalization=normalization, auto_eog=auto_eog, auto_ecg=auto_ecg)
                
            if(crop == True):
                crop_length = np.min(period_interval[count])
               
            # ===== Epoch / Event Creation ======
            temp_epochs, temp_events = self.event_epoch_creation(raw=raw, event_id=event_id, period_interval=period_interval[count], \
                                               duration=duration, picks=picks, overlap=overlap, tmin=tmin, tmax=tmax, \
                                               baseline=baseline, single_event_id=single_event_id, XDAWN=XDAWN, crop_length=crop_length)
                
            other_info['projs'].append(temp_epochs.info['projs'])
            temp_epochs.info['projs'] = [] #reset projs info for concatenation reasons
            # ===== Epoch / Event Creation ======
        
            count += 1
            epochs_list.append(temp_epochs)
            total_events = np.row_stack((total_events, temp_events))
        
            print('Epoching : ' + file + ' is completed')    
            
            gc.collect()
        
        total_epochs = self.epoch_concatenation(epochs_list = epochs_list)
        # total_epochs = []
        
        return total_epochs, total_events, epochs_list, other_info
    
    # def eeg_epoching_of_given_folder_rawconcattype(self, folder_path, renamed_channels, channel_types, picked_channels, picks, \
    #                                                event_id, period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
    #                                                baseline=(0, 0.2), f_min=0.1, f_max=48, f_notch = None):
        
    #     files = list()
    #     all_raw = list()
    #     all_raw_length = list()
    #     for file in os.listdir(folder_path):
    #         if file.endswith(".edf"):
    #             files.append(file)
    #             temp_raw = self.read_edf_file(file_path=folder_path + '/' + file)
    #             all_raw_length.append(np.size(temp_raw._data,1))
    #             all_raw.append(temp_raw)
                 
    #     # ==== Raw Concatenation =====
    #     raw = concatenate_raws(all_raw)
    #     del all_raw, temp_raw
    #     # ==== Raw Concatenation =====
        
    #     # ========= Sensor Info Update ======
    #     if(renamed_channels is not None and channel_types is not None and picked_channels is not None):
    #         raw = self.sensor_location_update(raw=raw, renamed_channels=renamed_channels, channel_types=channel_types, \
    #                                           picked_channels=picked_channels)
    #     # ========= Sensor Info Update ======
        
    #     # ==== Preprocess =====
    #     if(f_notch != None):
    #         raw.notch_filter(f_notch, filter_length='auto', phase='zero')
            
    #     raw.filter(l_freq=f_min, h_freq=f_max)
    #     raw = self.SSP_artifact_removal(raw=raw)
    #     # ==== Preprocess =====
        
    #     # ========= Epoch / Event Creation ==========
    #     #==== Event Merger ====
    #     # total_period_interval = np.empty(shape=[0,3])
    #     for i in range(1, len(files)):
    #         period_interval[i] += all_raw_length[i-1]
    #     #==== Event Merger ====
        
    #     epochs, events = self.event_epoch_creation_rawconcattype(raw=raw, event_id=event_id, period_interval=period_interval, \
    #                                                duration=duration, picks=picks, overlap=overlap, tmin=tmin, tmax=tmax, baseline=baseline)
    #     # ========= Epoch / Event Creation ==========
        
#%% =========== MNE + Python Combination Functions ==============
    def psd_spectrum_standardized_stage_comparison_plot(self, epochs, events, explanation, saving_directory=None, \
                                                        fmin=0.1, fmax=48, n_overlap=0, nfft=None, picks=None, channelbychannel=False,\
                                                        standardization=False, smoothing=False, psd_type='periodogram', custom_ax=None):
        Fs = epochs.info['sfreq']
        if(nfft is None):
            nfft = 2 ** (self.nextpow2(Fs) + 2)
        epochs_data = epochs.get_data() 
        channel_names = epochs.ch_names
        
        # ===== Compute the PSD ==========
        if(psd_type == 'periodogram'):
            freqs, psd = periodogram(x=epochs_data, fs=Fs, nfft=nfft)
        elif(psd_type == 'multitaper'):
            psd, freqs = psd_array_multitaper(x=epochs_data, sfreq=Fs, adaptive=True, fmin=fmin, fmax=fmax,\
                                              normalization='full', verbose=0)
        elif(psd_type == 'welch'):
            psd, freqs = mne.time_frequency.psd_welch(inst=epochs, fmin=fmin, fmax=fmax, tmin=None, \
                                                      tmax=None, n_fft=nfft, n_overlap=0, n_per_seg=50, \
                                                      picks='all', proj=False, n_jobs=1, \
                                                      reject_by_annotation=True, average='mean', verbose=None)
        else:
            psd = 0 #dummy variable
            
        #==== Get some Channels ====
        if(picks is not None):
            psd = psd[:,picks,:]
            channel_names = [channel_names[i] for i in picks]
        #==== Get some Channels ====
        
        #==== Min / Max border finding =====
        high_threshold, low_threshold = max(np.argwhere(freqs <= fmax))[0], min(np.argwhere(freqs >= fmin))[0]
        psd = psd[:,:,low_threshold:high_threshold]
        freqs = freqs[low_threshold:high_threshold]
        #==== Min / Max border finding =====
        
        # ===== Compute the PSD ==========
        
        # Convert power to dB scale.
        psd = 10 * np.log10(psd)
        # print(psd)

        if(standardization == True):        
            psd += abs(np.min(psd, axis=-1, keepdims=True)) #to make smallest power as 0
            psd /= np.sum(psd, axis=-1, keepdims=True)
  
        num_of_channels = np.size(psd,1)
  
        psd_avg = np.zeros((num_of_channels, 3, np.size(psd,2)))
        psd = np.transpose(a=psd, axes=(1,0,2))
        
        #==== Normalization ======
        for i in range(num_of_channels):
            for j in range(3): #num of stage
                temp_avg = np.mean(psd[i,np.where(events[:,2] == j),:], axis=1)
                if(smoothing==True):
                    temp_avg = self.envelopeCreator(timeSignal=temp_avg, degree=6)
                psd_avg[i,j,:] = temp_avg
        print('Normalization done!')
        #==== Normalization ======  
        
        #==== Grand Average ======
        psd_grand_avg = np.mean(psd_avg, axis=0)
        #==== Grand Average ======
    
        if(channelbychannel==True):
            
            #==== Subplots ======
            num_plot_x, num_plot_y = 2, 3
            fig, axs = plt.subplots(num_plot_x, num_plot_y)
            fig.subplots_adjust(hspace=1)
            fig.suptitle('Multitaper Periodogram of ' + explanation, size=30) 
            #==== Subplots ======
            
            #=============== Plot channel by channel ========================
            for i in range(np.size(psd_avg,0)):
            # i=0
                plt.figure()
                ax = plt.axes()
                
                #=== Main Plots =====
                ax.plot(freqs, psd_avg[i, 0], color='blue',
                        ls='-', label='REM', linewidth=3)
                ax.plot(freqs, psd_avg[i, 1], color='red',
                        ls='-', label='Lucid', linewidth=3)
                ax.plot(freqs, psd_avg[i, 2], color='green',
                        ls='-', label='Wake', linewidth=3)
                #=== Main Plots =====
                
                #===== Subplots ========
                ax_x, ax_y = int(i/(num_of_channels/num_plot_x)), i%num_plot_y #axis finder
                
                axs[ax_x, ax_y].plot(freqs, psd_avg[i, 0], color='blue',
                        ls='-', label='REM', linewidth=3)
                axs[ax_x, ax_y].plot(freqs, psd_avg[i, 1], color='red',
                        ls='-', label='Lucid', linewidth=3)
                axs[ax_x, ax_y].plot(freqs, psd_avg[i, 2], color='green',
                        ls='-', label='Wake', linewidth=3)
                #===== Subplots ========
    
                if(psd_type == 'periodogram'):
                    ax.set_title('Channel : ' + channel_names[i] + ', Periodogram of ' + explanation, size=25)
                if(psd_type == 'welch'):
                    ax.set_title('Channel : ' + channel_names[i] + ', Welch Periodogram of ' + explanation, size=25)
                if(psd_type == 'multitaper'):
                    ax.set_title('Channel : ' + channel_names[i] + ', Multitaper Periodogram of ' + explanation, size=25)
                    axs[ax_x, ax_y].set_title('Channel : ' + channel_names[i], size=25)
         
                ax.set_xlabel('Frequency (Hz)', size=20)
                axs[ax_x, ax_y].set_xlabel('Frequency (Hz)', size=20)
                if(standardization == True):
                    ax.set_ylabel(ylabel='Power [%]', size=20)
                    axs[ax_x, ax_y].set_ylabel(ylabel='Power [%]', size=20)
                else:
                    ax.set_ylabel(ylabel='Power [dB]', size=20)
                    axs[ax_x, ax_y].set_ylabel(ylabel='Power [dB]', size=20)
                ax.legend(loc='upper right', prop={'size': 20, 'weight':3})
                axs[ax_x, ax_y].legend(loc='upper right', prop={'size': 20, 'weight':3})
                ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
                axs[ax_x, ax_y].tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
                plt.show()
                
                #===Save Figure ====
                if(saving_directory is not None):
                    self.save_figure(saving_directory, explanation, extra=channel_names[i])
                #===Save Figure ====
            
                #=============== Plot channel by channel ========================
            
        #=== Save All Channels at Once ====
        if(saving_directory is not None):
            fig.set_size_inches(32, 18)
            fig.savefig('Multitaper Periodogram of All Channels.jpeg', pad_inches=1, bbox_inches='tight', dpi=400)
            plt.close(fig)
        #=== Save All Channels at Once ====
            
        #========== Plot Grand Average ============
        if(custom_ax is None):
            plt.figure()
            ax = plt.axes()
        else:
            ax = custom_ax
        
        # psd_grand_avg = (psd_grand_avg - np.mean(psd_grand_avg)) / np.std(psd_grand_avg)    
        
        ax.plot(freqs, psd_grand_avg[0], color='blue',
                ls='-', label='REM', linewidth=3)
        ax.plot(freqs, psd_grand_avg[1], color='red',
                ls='-', label='Lucid', linewidth=3)
        ax.plot(freqs, psd_grand_avg[2], color='green',
                ls='-', label='Wake', linewidth=3)
        
        if(psd_type == 'periodogram'):
            ax.set_title('Grand Average of All Periodograms : ' + explanation, size=25)
        if(psd_type == 'welch'):
            ax.set_title('Grand Average of All Welch Periodograms : ' + explanation, size=25)
        if(psd_type == 'multitaper'):
            ax.set_title(explanation, size=25)
 
        ax.set_xlabel('Frequency (Hz)', size=20)
        if(standardization == True):
            ax.set_ylabel(ylabel='Power [%]', size=20)
        else:
            ax.set_ylabel(ylabel='Power [dB]', size=20)
        ax.legend(loc='upper right', prop={'size': 20, 'weight':3})
        ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
        plt.show()
        #========== Plot Grand Average ============
        
        #===Save Figure ====
        if(saving_directory is not None):
            self.save_figure(saving_directory, explanation, extra='Grand Avg')
        #===Save Figure ====
        
        return psd_grand_avg, freqs
        
    def psd_spectrum_standardized_lucid_REM_ratio(self, epochs, events, saving_directory=None, \
                                                  fmin=0.1, fmax=50, nfft=256, n_overlap=128, picks=None, \
                                                  standardization=False, smoothing=False, psd_type='periodogram', \
                                                  explanation=None, ylim=[0.9, 1.1], custom_ax=None, channelbychannel=True,\
                                                  label='Grand Avg', color='#f1ff1b', linestyle=(0, (5, 1)), plot=True):
        
        Fs = epochs.info['sfreq']
        nfft = 2 ** (self.nextpow2(p=Fs) + 2)
        epochs_data = epochs.get_data()
        channel_names = epochs.ch_names
        
        # ===== Compute the PSD ==========
        if(psd_type == 'peridogram'):
            freqs, psd = periodogram(x=epochs_data, fs=Fs, nfft=nfft)
        elif(psd_type == 'multitaper'):
            psd, freqs = psd_array_multitaper(x=epochs_data, sfreq=Fs, adaptive=True, fmin=fmin, fmax=fmax,\
                                              normalization='full', verbose=0)
        elif(psd_type == 'welch'):
            psd, freqs = mne.time_frequency.psd_welch(inst=epochs, fmin=fmin, fmax=fmax, tmin=None, \
                                                         tmax=None, n_fft=nfft, n_overlap=0, n_per_seg=50, \
                                                         proj=False, n_jobs=1, \
                                                         picks='all', reject_by_annotation=True, average='mean', verbose=None)
        else:
            psd = 0 #dummy variable
        
        #==== Get some Channels ====
        if(picks is not None):
            psd = psd[:,picks,:]
            channel_names = [channel_names[i] for i in picks]
        #==== Get some Channels ====
            
        #==== Min / Max border finding =====
        high_threshold, low_threshold = max(np.argwhere(freqs <= fmax))[0], min(np.argwhere(freqs >= fmin))[0]
        #=== For each Freq Band ====
        freq_bands = np.zeros((7,2)).astype(int)
        freq_bands[0,0], freq_bands[0,1] = min(np.argwhere(freqs >= 1))[0], max(np.argwhere(freqs < 4))[0]
        freq_bands[1,0], freq_bands[1,1] = min(np.argwhere(freqs >= 4))[0], max(np.argwhere(freqs < 8))[0] 
        freq_bands[2,0], freq_bands[2,1] = min(np.argwhere(freqs >= 8))[0], max(np.argwhere(freqs < 12))[0]
        freq_bands[3,0], freq_bands[3,1] = min(np.argwhere(freqs >= 12))[0], max(np.argwhere(freqs < 30))[0]
        freq_bands[4,0], freq_bands[4,1] = min(np.argwhere(freqs >= 30))[0], max(np.argwhere(freqs < 40))[0]
        freq_bands[5,0], freq_bands[5,1] = min(np.argwhere(freqs >= 40))[0], max(np.argwhere(freqs < 45))[0]
        freq_bands[6,0], freq_bands[6,1] = min(np.argwhere(freqs >= 45))[0], max(np.argwhere(freqs < 48))[0]
        #=== For each Freq Band ===
        psd = psd[:,:,low_threshold:high_threshold]
        freqs = freqs[low_threshold:high_threshold]
        #==== Min / Max border finding =====
        
        # ===== Compute the PSD ==========
        
        # Convert power to dB scale.
        psd = 10 * np.log10(psd)

        if(standardization == True):        
            psd += abs(np.min(psd, axis=-1, keepdims=True)) #to make smallest power as 0
            psd /= np.sum(psd, axis=-1, keepdims=True)
  
        num_of_channels = np.size(psd,1)
  
        psd_avg = np.zeros((num_of_channels, 3, np.size(psd,2)))
        psd = np.transpose(a=psd, axes=(1,0,2))
        
        #==== Normalization ======
        for i in range(num_of_channels):
            for j in range(3): #num of stage
                temp_avg = np.mean(psd[i,np.where(events[:,2] == j),:], axis=1)
                if(smoothing==True):
                    temp_avg = self.envelopeCreator(timeSignal=temp_avg, degree=3)
                psd_avg[i,j,:] = temp_avg
        #==== Normalization ======    

        #=== Lucid Ratios ===
        lucid_rem_ratio = psd_avg[:,1,:] / psd_avg[:,0,:]
        grand_avg_lucid_rem_ratio = np.mean(lucid_rem_ratio, axis=0)
        sort_channels_per_band = list()
        for i in range(7):
            temp_band = np.sum(lucid_rem_ratio[:,freq_bands[i,0]:freq_bands[i,1]], axis=1)
            temp_sorted_channels = [channel_names[i] for i in np.argsort(temp_band * -1)]
            sort_channels_per_band.append(temp_sorted_channels)
        # lucid_wake_ratio = welch_avg[:,1,:] / welch_avg[:,2,:]
        #=== Lucid Ratios ===
          
        if(plot == True):
            #=============================== Plot ======================================
            #=== Plot Lucidity / REM ====
            
            #===== Pseudo-random Color Generation ======
            random.seed(313)
            number_of_colors = np.size(psd_avg,0) 
            color_chunk = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
            #===== Pseudo-random Color Generation ======
            
            if(custom_ax is None):
                plt.figure()
                ax = plt.axes()
            else:
                ax = custom_ax
    
            if(channelbychannel == True):
                for i in range(np.size(psd_avg,0)):
                    ax.plot(freqs, lucid_rem_ratio[i], color=color_chunk[i],
                            ls='-', linewidth=3)
                
            if(psd_type == 'periodogram'):
                ax.set_title(explanation + ' Periodogram of Lucidity / REM', size=25)
            elif(psd_type == 'welch'):
                ax.set_title(explanation + ' Welch Periodogram of Lucidity / REM', size=25)
            elif(psd_type == 'multitaper'):
                ax.set_title(explanation, size=25)
            ax.set_xlabel('Frequency (Hz)', size=20)
            ax.set_ylabel('Power Lucid / REM', size=20)
            ax.tick_params(labelsize=15) #change size of tick parameters on x and y axes
            ax.plot(freqs, np.ones(len(freqs)), ls='--', linewidth=5, color='black')
            
            ax.plot(freqs, grand_avg_lucid_rem_ratio, ls=linestyle, linewidth=7, color=color, label=label)
            #==== Frequency Limit Drawer ======
            amp_linspace = np.linspace(np.min(lucid_rem_ratio), np.max(lucid_rem_ratio), num=len(freqs))
            ax.plot(np.ones(len(freqs)), amp_linspace, ls='--',  linewidth=3, color='#5f6567')
            ax.plot(np.ones(len(freqs))*4, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
            ax.plot(np.ones(len(freqs))*8, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
            ax.plot(np.ones(len(freqs))*12, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
            ax.plot(np.ones(len(freqs))*30, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
            ax.plot(np.ones(len(freqs))*40, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
            ax.plot(np.ones(len(freqs))*45, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
            #==== Frequency Limit Drawer ======
            
            #===== Texts ======
            ax.text(1.2, 1.03, s='δ (1–4 Hz)', color='black', fontsize=20)
            ax.text(4.8, 1.03, s='θ (4–8 Hz)', color='black', fontsize=20)
            ax.text(8.5, 1.03, s='α (8–12 Hz)', color='black', fontsize=20)
            ax.text(20.2, 1.03, s='β (12-30 Hz)', color='black', fontsize=20)
            ax.text(32.5, 1.03, s='γ1 (30–40 Hz)', color='black', fontsize=20)
            ax.text(40.3, 1.03, s='γ2 (40-45 Hz)', color='black', fontsize=20)
            ax.text(45.5, 1.03, s='γ3 (45+ Hz)', color='black', fontsize=20)
            #===== Texts ======
            
            ax.set_ylim(ylim[0], ylim[1])
            
            #==== Channel Band Power Sorted Text ======
            # xlabel = [2.3, 5.8, 9.7, 13.6, 17.6, 23.4, 31, 40, 46.2]
            # for i in range(9):
            #     for j in range(len(channel_names)):
            #         if(j == 0):
            #             ax.text(xlabel[i], 1.019-j*0.0015, s=sort_channels_per_band[i][j], fontsize=15, color='black', fontweight='bold')
            #         else:
            #             ax.text(xlabel[i], 1.019-j*0.0015, s=sort_channels_per_band[i][j], fontsize=15, color='black')
            #==== Channel Band Power Sorted Text ======
            
            ax.legend(loc='upper left', prop={'size': 20, 'weight':3})
            plt.show()
            
            #===Save Figure ====
            if(saving_directory is not None):
                self.save_figure(saving_directory, explanation=explanation, dpi=400)
            #===Save Figure ====
         
            #=== Plot Lucidity / REM ===  
            
            # #=== Plot Lucidity / Wake ===
            # plt.figure()
            # ax = plt.axes()
            # for i in range(np.size(welch_avg,0)):
            #     ax.plot(welch_spectrum_freqs, lucid_wake_ratio[i], color=color_chunk[i],
            #             ls='-', label=eeg_channels[i], linewidth=3)
            #     ax.set_title('Channel : ' + eeg_channels[i] + ', PSD of Lucidity / Wake', size=25)
            #     ax.set_xlabel('Frequency (Hz)', size=20)
            #     ax.set_ylabel('Power Lucid / Wake', size=20)
            #     ax.legend(loc='upper right', prop={'size': 20, 'weight':3})
            #     ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
            #     plt.show()
            # #=== Plot Lucidity / Wake ===  
            #=============================== Plot ======================================
        
        return lucid_rem_ratio, grand_avg_lucid_rem_ratio, freqs
#%% ============= Spectral Connectivity Analysis =================

    def spectral_coherence_avgofeachband(self, epochs, eventnames, method='coh', mode='multitaper'):
        
        fmin = (2, 4, 8, 12, 30, 36) #hz
        fmax = (4, 7, 12, 30, 36, 45) #hz
        sfreq = epochs.info['sfreq']  # the sampling frequency
        
        cohs = list()
        for i in range(len(eventnames)):
            coh, freqs, times, n_epochs, n_tapers = spectral_connectivity_epochs(epochs[eventnames[i]], method, mode=mode, indices=None,\
                                                                  sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, n_jobs=1)
            cohs.append(coh)    
            
        return cohs, freqs, times
            
    def spectral_coherence_nonaveraged(self, epochs, eventnames, method='coh', mode='multitaper'):
        fmin = 0.1 #hz
        fmax = 48 #hz
        sfreq = epochs.info['sfreq']  # the sampling frequency
        
        cohs = list()
        for i in range(len(eventnames)):
            coh, freqs, times, n_epochs, n_tapers = spectral_connectivity_epochs(epochs[eventnames[i]], method=method, mode=mode, indices=None,\
                                                              sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=False, n_jobs=1)
            cohs.append(coh)
            
        return cohs, freqs, times
        
    def spectral_coherence_analysis(self, epochs, eventnames, saving_directory):
    
        spectral_coherence_eachband, freqs, times = self.spectral_coherence_avgofeachband(epochs=epochs, eventnames=eventnames, method='coh', mode='multitaper')
    
        spectral_coherence_spectrum_coh, freqs, times = self.spectral_coherence_nonaveraged(epochs=epochs, eventnames=eventnames, method='coh', mode='multitaper')
        spectral_coherence_spectrum_imcoh, freqs, times = self.spectral_coherence_nonaveraged(epochs=epochs, eventnames=eventnames, method='imcoh', mode='multitaper')
        spectral_coherence_spectrum_wpli, freqs, times = self.spectral_coherence_nonaveraged(epochs=epochs, eventnames=eventnames, method='wpli', mode='multitaper')
        
        coherences = [spectral_coherence_spectrum_coh, spectral_coherence_spectrum_imcoh, spectral_coherence_spectrum_wpli]
        ylabel = ['Power Coherence', 'Imaginary Coherence', 'Weighted Phase Lag Index Coherence']
        title = ['Power Coherence Analysis', 'Imaginary Power Coherence Analysis', 'Weighted Phase Lag Index Coherence Analysis']
        for i in range(3):
            intervalLength = 31
            REM_coherence  = self.envelopeCreator(coherences[i][0][1,0],intervalLength=intervalLength)
            lucid_coherence  = self.envelopeCreator(coherences[i][1][1,0],intervalLength=intervalLength)
            wake_coherence  = self.envelopeCreator(coherences[i][2][1,0],intervalLength=intervalLength)
            
            band_indexes = [np.argwhere(freqs==4)[0,0], np.argwhere(freqs==8)[0,0], np.argwhere(freqs==12)[0,0], np.argwhere(freqs==16)[0,0],\
                            np.argwhere(freqs==16)[0,0], np.argwhere(freqs==20)[0,0], np.argwhere(freqs==28)[0,0], np.argwhere(freqs==36)[0,0],\
                            np.argwhere(freqs==45)[0,0]]
    
            self.coherence_plot(REM_coherence, lucid_coherence, wake_coherence, freqs, title[i], band_indexes, ylabel[i], saving_directory)
        
        return spectral_coherence_spectrum_coh, spectral_coherence_spectrum_imcoh, spectral_coherence_spectrum_wpli
    
#%% ==================== Fractal Analysis ======================


#%% ============ MNE Plot Functions ===========

    def plot_epochs(self, epochs, picks='eeg', scalings=50):
        epochs.plot(scalings=scalings, show=True, block=False, n_epochs=10, title='Overlapping Events', picks=picks) #plot epoch raw
       
    def plot_image(self, epochs, picks): 
        epochs.plot_image(picks=picks) #Epoch time-power image
        
    def plot_PSD_topomap(self, epochs, ch_type='eeg'):
        epochs.plot_psd_topomap(ch_type=ch_type, dB=True, proj=True)
        
    def plt_evoked_topomap(self, raw, times, explanation, events, event_id, picks, tmin=-1, tmax=4, saving_directory=None, CSD=False,\
                           performance_mode=False):
        
        if(performance_mode == False):
        
            if(CSD==True):
                raw = self.CSD(raw)
                picks = 'csd'
                explanation = 'CSD ' + explanation
                
            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, picks=picks, preload=True)
            del raw
        
        else:  #if performance_mode is true, than raw will be equal to epochs
            epochs = raw.copy()
            if(CSD==True):
                picks = 'csd'
                explanation = 'CSD ' + explanation
                
            del raw
            
        #===== Evoke Response Creation =======
        # evoked = epochs['T1'].average()
        evoked = epochs.average()
        evoked.apply_baseline((None,0))
        #===== Evoke Response Creation =======
        
        #=== Evoked Topomap Plot ======
        evoked.plot_topomap(colorbar=True, times=times)
        plt.title(explanation, size=25)
        # plt.title('Scalp Topomaps of CSD no: ' + str(i), size=25)
        #=== Evoked Topomap Plot ======
        
        #=== Maximize ====
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18)
        #=== Maximize ====
        
        #=== Save Topomap =======
        if(saving_directory is not None):
            plt.savefig(saving_directory + '/' + 'Topomap ' + explanation, pad_inches=0.5, bbox_inches='tight', dpi=400)
            print('Figure has saved successfully!')
            # plt.close()
        #=== Save Topomap =======
        
        del epochs, evoked
        gc.collect()
        

    def PSD_of_all_channels(self, epochs):
        plt.figure()
        ax = plt.axes()
        epochs.plot_psd(ax=ax, dB=True, picks=['eeg','eogz'], xscale='linear', estimate='power')
        ax.set_title(label='PSD of Different Channels', size=25)
        ax.set_xlabel(xlabel='Frequency (Hz)', size=20)
        ax.set_ylabel(ylabel='µV^2/Hz (dB)', size=20)
        # ax.legend(ax.lines[2::3], stages, prop={'size': 20, 'weight':3})
        ax.grid(linewidth=1.2) #change grid line width
        ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
        
        for line in plt.gca().lines: #change linewidth of axes plotted lines
            line.set_linewidth(3.)
        gc.collect()
            
    def PSD_of_all_stages(self, epochs, event_id, explanation, picks=None, saving_directory=None, fmin=0.1, fmax=49):
        
        if(picks is None):
            picks = 'eeg'
            
        # stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # random color code generation
        stage_colors = ['red','blue','green']
        stages = sorted(event_id.keys())
        
        plt.figure()
        ax = plt.axes()
        for stage, color in zip(stages, stage_colors):
            epochs[stage].plot_psd(area_mode=None, color=color, ax=ax, fmin=fmin, fmax=fmax, show=False,\
                                        average=True, spatial_colors=False, picks=picks, dB=True, estimate='power') #EOG-1, EOG-2
                
        ax.set_title(label=explanation, size=25)
        ax.set_xlabel(xlabel='Frequency (Hz)', size=20)
        ax.set_ylabel(ylabel='µV^2/Hz (dB)', size=20)
        legends = ax.legend(ax.lines[2::3], stages, prop={'size': 20, 'weight':6})
        ax.grid(linewidth=1.2) #change grid line width
        ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
        
        for line in plt.gca().lines: #change linewidth of axes plotted lines
            line.set_linewidth(3.)
            
        for line in legends.get_lines():
            line.set_linewidth(3.0)
            
        #=== Save figure ====
        if(saving_directory is not None):
            self.save_figure(saving_directory=saving_directory, explanation=explanation)
            plt.close()
        #=== Save figure ====
        gc.collect()
            
    def multitaper_spectrogram_lucidity_neighbours(self, epochs, lucidity_periods, baseline_length=20, normalization=True, saving_directory=None,\
                                                   explanation=None, fmin=0.1, fmax=48, file_names=None, labels=['REM', 'Lucid', 'Wake']):
        #Multi-taper Parameters
        freqs = np.arange(fmin, fmax, 0.1)  # frequencies from 2-35Hz
        n_cycles = freqs  # use constant t/f resolution
        vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
        n_cycles = freqs * 2  # use constant t/f resolution
        
        powers = list()
        for i in range(len(epochs)):
            epoch = epochs[i]
            
            power = tfr_multitaper(epoch, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
                                   average=True)
            baseline_period = 5 #second
            beginner = power.times[0]
            finish = power.times[-1]
            states = lucidity_periods[i][1]
            difference = states[1] - states[0]
            
            power.apply_baseline([beginner, beginner+20], mode='mean')
            
            avg_power = np.mean(power._data, 0)
            power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]
            
            if(normalization == True):
                power._data = np.absolute(power._data) ** (1/2.)
                power._data = 10 * np.log10(power._data)
            
            if(file_names is not None):
                power = self.multitaper_spectrogram_MNE(power=power, state_times=states, saving_directory=saving_directory, \
                                                    fmin=fmin, fmax=fmax, explanation=file_names[i], labels=labels)
            else:
                power = self.multitaper_spectrogram_MNE(power=power, state_times=states, saving_directory=saving_directory, \
                                                    fmin=fmin, fmax=fmax, explanation=explanation + '_' + str(i), labels=labels)
            powers.append(power)
            
            gc.collect()
            
        return powers
            
    def multitaper_spectrogram_MNE(self, power, fmin, fmax, state_times=None, saving_directory=None, explanation=None, labels=['0','1','2']):
        
        freq_thesholds = np.array([4,8,12,30])
        vmin=np.min(power._data)
        vmax=np.max(power._data)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 1]})
        power.plot([0], vmin=vmin, vmax=vmax, axes=ax[0], colorbar=False, show=False)
        
        ax[0].axvline(0, linewidth=1, color="black", linestyle=":")  # event
        ax[0].legend(fontsize=15)
        fig.colorbar(ax[0].images[-1], cax=ax[-1])
        
        if(state_times is not None):
            
            beginner = power.times[0]
            finish = power.times[-1]
            difference = state_times[1] - state_times[0]
            text_location_difference = (difference * 3) * 0.033
            
            ax[0].text(beginner-text_location_difference,3, 'Delta', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            ax[0].text(beginner-text_location_difference,6, 'Theta', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            ax[0].text(beginner-text_location_difference,10, 'Alpha', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            ax[0].text(beginner-text_location_difference,18, 'Beta', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            if(fmax > 37): #some spectrum has no high gamma values so, gamma text might be flew to the top of gamma section
                ax[0].text(beginner-text_location_difference,37, 'Gamma', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            else:
                ax[0].text(beginner-text_location_difference,32, 'Gamma', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            
            ax[0].text(state_times[0] - difference/2, 49, labels[0], horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            ax[0].text(state_times[0] + difference/2, 49, labels[1], horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            ax[0].text(state_times[1] + difference/2, 49, labels[2], horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
            
            ax[0].vlines(x=state_times, ymin=0, ymax=48, colors='purple', ls='--', lw=3, label='States')
            ax[0].hlines(y=freq_thesholds, xmin=beginner, xmax=finish, colors='black', ls='--', lw=3, label='Brain waves')
        
        # ax[0].set_xlim(0, 100)
        plt.title('Power Spectrum', fontsize=13)
        plt.suptitle(explanation, fontsize=20)
        fig.show()
        
        #=== Save Figure =======
        if(saving_directory is not None):
            self.save_figure(saving_directory, explanation=explanation, dpi=200)
        #=== Save Figure =======
        
        return power
        
    def multitaper_spectrogram_all_periods(self, epochs, event_id, picks='eeg'):
        freqs = np.arange(5., 49., 0.1)
        n_cycles = freqs / 2
        vmin, vmax = -3., 3.  # Define our color limits.
        time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
        stages = sorted(event_id.keys())
        
        for i in range(len(stages)):
            power = mne.time_frequency.tfr_multitaper(epochs[stages[i]], freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, \
                                                      return_itc=False, picks=picks)
            plt.figure()
            ax = plt.axes()
            power.plot([0], baseline=(0., 0.1), mode='mean', vmin=vmin, vmax=vmax, axes=ax)
            ax.set_title(label='Multitaper Spectrogram Frontal, stage :' + stages[i], size=25)
            ax.set_xlabel(xlabel='Time (s)', size=20)
            ax.set_ylabel(ylabel='Frequency (Hz)', size=20)
            ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes  

    def plt_sensor_location(self, raw):

        fig = plt.figure()
        ax2d = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')
        raw.plot_sensors(ch_type='eeg', axes=ax2d, show_names=True)
        raw.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d', show_names=True)
        
    def plt_GFP(self, raw, picks, saving_directory, events, event_id, explanation, tmin=-1, tmax=4, baseline=None):
        
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 30),
            ('Gamma', 30, 45)
                     ]
        frequency_map = list()
        
        '''Now we can compute the Global Field Power We can track the emergence of spatial patterns compared to baseline 
        #for each frequency band, with a bootstrapped confidence interval.'''
        for band, fmin, fmax in iter_freqs:
            # (re)load the data to save memory
            raw_copy = raw.copy()
            raw_copy.load_data()
        
            # bandpass filter
            raw_copy.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                       l_trans_bandwidth=1,  # make sure filter params are the same
                       h_trans_bandwidth=1)  # in each band and skip "auto" option.
        
            # epoch
            epochs = mne.Epochs(raw_copy, events, event_id, tmin, tmax, baseline=baseline, picks=picks, preload=True)
            # remove evoked response
            epochs.subtract_evoked()
        
            # get analytic signal (envelope)
            epochs.apply_hilbert(envelope=True)
            frequency_map.append(((band, fmin, fmax), epochs.average()))
        
            del epochs
        del raw_copy, raw
        gc.collect()
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        for ((freq_name, fmin, fmax), average), color, ax in zip(
                frequency_map, colors, axes.ravel()[::-1]):
            times = average.times * 1e3
            gfp = np.sum(average.data ** 2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(-500, 0))
            ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
            ax.axhline(0, linestyle='--', color='grey', linewidth=2)
            ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                          stat_fun=self.stat_fun)
            ci_low = rescale(ci_low, average.times, baseline=(None, 0))
            ci_up = rescale(ci_up, average.times, baseline=(None, 0))
            ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
            ax.grid(True)
            ax.set_ylabel('GFP', size=20)
            ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                        xy=(0.95, 0.8),
                        horizontalalignment='right',
                        xycoords='axes fraction')
            ax.set_xlim(-1000, 4000)
        
        axes.ravel()[-1].set_xlabel('Time [ms]', size=20)
        ax.set_title('Global Field Power of Event-related Dynamics ' + explanation, size=25)
        
        ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes  
        #=== Maximiz====
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18)
        #=== Maximize ====
        
        #===== Save Figure ======
        if(saving_directory is not None):
           self.save_figure(saving_directory, explanation=explanation, dpi=400)
        #===== Save Figure ======
        gc.collect()
    
    def plt_evoked_response(self, raw, explanation, picks, events, event_id, saving_directory=None, tmin=-1, tmax=4):
        
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, picks=picks, preload=True)

        #===== Evoke Response Creation =======
        evoked = epochs.average()
        evoked.apply_baseline((None,0))
        #===== Evoke Response Creation =======
        
        #====== Plot Figure =======
        plt.figure()
        ax = plt.axes()
        evoked.plot(gfp=True, axes=ax, spatial_colors=True)
        ax.set_title(explanation, size=25)
        ax.set_xlabel(xlabel='Time (s)', size=20)
        ax.set_ylabel(ylabel='µV', size=20)
        ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes  
        #====== Plot Figure =======
        
        #=== Maximize ====
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18)
        #=== Maximize ====
        
        #===== Save Figure ======
        if(saving_directory is not None):
            plt.savefig(saving_directory + '/' + 'Evoked Response with GFP_' + explanation, pad_inches=0.5, bbox_inches='tight', dpi=400)
            print('Figure has saved successfully!')
            plt.close()
        #===== Save Figure ======
        gc.collect()
#%% ======================================== Custom Plot Functions =====================================
    def avg_multiple_spectrums_onefigure(self, spectral_list, freqs, title, xlabel, ylabel, saving_directory, explanation):
        
        plt.figure()
        ax = plt.axes()
        import random
        for i in range(len(spectral_list)):
            color = "#%06x" % random.randint(0, 0xFFFFFF)
            ax.plot(freqs, spectral_list[i], color=color, ls='-', linewidth=3, label='File_'+str(i))  
        
        #Grand AVG
        spectral_avg = np.mean(spectral_list, axis=0)
        ax.plot(freqs, spectral_avg, color='yellow', ls='-', linewidth=6, label='Grand Avg') 
        
        #==== Frequency Limit Drawer ======
        amp_linspace = np.linspace(0.9, 1.1, num=len(freqs))
        ax.plot(np.ones(len(freqs)), amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*4, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*8, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*12, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*16, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*20, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*28, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*36, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*45, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        #==== Frequency Limit Drawer ======
        
        #===== Texts ======
        ax.text(1.2, 1.03, s='δ (1–4 Hz)', color='black', fontsize=20)
        ax.text(4.8, 1.03, s='θ (4–8 Hz)', color='black', fontsize=20)
        ax.text(8.5, 1.03, s='α (8–12 Hz)', color='black', fontsize=20)
        ax.text(12.2, 1.03, s='β1 (12-16 Hz)', color='black', fontsize=20)
        ax.text(16.2, 1.03, s='β2 (16–20 Hz)', color='black', fontsize=20)
        ax.text(22.4, 1.03, s='γ1 (20–28 Hz)', color='black', fontsize=20)
        ax.text(30.5, 1.03, s='γ2 (28–36 Hz)', color='black', fontsize=20)
        ax.text(38.7, 1.03, s='γ-40Hz (36-45 Hz)', color='black', fontsize=20)
        ax.text(45.5, 1.03, s='γ+ (45+ Hz)', color='black', fontsize=20)
        #===== Texts ======
        
        ax.set_title(title, size=25)
        ax.set_xlabel(xlabel, size=20)
        ax.set_ylabel(ylabel, size=20)
        ax.set_ylim(0.7, 1.4)
        ax.legend(loc='upper left', prop={'size': 11, 'weight':3})
        ax.plot(freqs, np.ones(len(freqs)), ls='--', linewidth=5, color='black')
        
        self.save_figure(saving_directory, explanation=explanation, dpi=400)
        
    def coherence_plot(self, REM_coherence, lucid_coherence, wake_coherence, freqs, title, band_indexes, ylabel):
        plt.figure()
        ax = plt.axes()
        ax.plot(freqs, REM_coherence, marker='d', linewidth=3, color='blue', markevery=band_indexes, markersize=25)
        ax.plot(freqs, lucid_coherence, marker='d', linewidth=3, color='red', markevery=band_indexes, markersize=25)
        ax.plot(freqs, wake_coherence,  marker='d', linewidth=3, color='green', markevery=band_indexes, markersize=25)
        ax.set_xlabel('Frequency [Hz]', size=20)
        ax.set_ylabel('Multitaper PSD Coherence', size=20)
        
        def update_prop(handle, orig):
            handle.update_from(orig)
            handle.set_marker("")
        
        ax.legend(['REM', 'Lucid', 'Wake'], loc='lower right', fontsize=20, handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)})
        
        ax.set_title(title, size=25)
        
        #===== Texts ======
        ax.text(1.2, 0.95, s='δ (1–4 Hz)', color='black', fontsize=12, fontweight='bold')
        ax.text(4.8, 0.95, s='θ (4–8 Hz)', color='black', fontsize=12, fontweight='bold')
        ax.text(8.5, 0.95, s='α (8–12 Hz)', color='black', fontsize=12, fontweight='bold')
        ax.text(12.2, 0.95, s='β1 (12-16 Hz)', color='black', fontsize=12, fontweight='bold')
        ax.text(16.2, 0.95, s='β2 (16–20 Hz)', color='black', fontsize=12, fontweight='bold')
        ax.text(22.4, 0.95, s='γ1 (20–28 Hz)', color='black', fontsize=12, fontweight='bold')
        ax.text(30.5, 0.95, s='γ2 (28–36 Hz)', color='black', fontsize=12, fontweight='bold')
        ax.text(38.7, 0.95, s='γ-40Hz (36-45 Hz)', color='black', fontsize=12, fontweight='bold')
        ax.text(45.5, 0.95, s='γ+ (45+ Hz)', color='black', fontsize=12, fontweight='bold')
        #===== Texts ======
        
        #==== Frequency Limit Drawer ======
        amp_linspace = np.linspace(0.9, 1.0, num=len(freqs))
        ax.plot(np.ones(len(freqs)), amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*4, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*8, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*12, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*16, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*20, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*28, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*36, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        ax.plot(np.ones(len(freqs))*45, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
        #==== Frequency Limit Drawer ======
#%% ========== Statistical Plots ==============
    def sorted_boxplot_scores(self, results, title, xlabel, ylabel, with_diamongs=False):
        
        #=== Sort processing =====
        mean_results = np.mean(results, axis=1)
        sorted_indexes = np.argsort(-1 * mean_results) #descending order
        plot_results = np.transpose(results[sorted_indexes]) #boxplot get data column-by-column
        len_of_results = len(results)
        #=== Sort processing =====
        
        green_diamond = dict(markerfacecolor='g', marker='D')
        
        fig1, ax1 = plt.subplots()
        ax1.boxplot(plot_results, showfliers=False)
        plt.plot(mean_results[sorted_indexes], linewidth=4, color='blue')
        plt.plot(np.arange(1,len_of_results+1), np.ones(len_of_results)*0.5, ls='--', linewidth=4, color='black')
        ax1.set_title(title, size=25)
        ax1.set_xlabel(xlabel=xlabel, size=20)
        ax1.set_ylabel(ylabel=ylabel, size=20)
        ax1.tick_params(labelsize=10) #chnage size of tick parameters on x and y axes  
#%% ========= Preprocessing ============
    def SSP_artifact_removal(self, raw):
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
    
    def CSD(self, raw, stiffness=4, keepEEGFormat=False):
        
        temp_raw = raw.copy()
        
        if(keepEEGFormat == True):
            temp_raw._data = mne.preprocessing.compute_current_source_density(temp_raw, stiffness=stiffness)._data
        else:
            temp_raw = mne.preprocessing.compute_current_source_density(temp_raw, stiffness=stiffness)
            
        return temp_raw
    
#%% =========== Normalization Strategies ============
    def robustZScore(self, raw, ifnumpy=False):
        
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
#%% ======== Save Figure ===========
    def save_figure(self, saving_directory, explanation='', extra=None, dpi=200, pad_inches=0.2, ifsvg=False):
        
        #=== Maximize ====
        figure = plt.gcf()  # get current figure
        # figure.set_size_inches(32, 18)
        # mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        plt.show()
        #=== Maximize ====
        
        if(ifsvg):
            if(extra is not None):
                plt.savefig(saving_directory + '/' + explanation + '_' + extra + '.svg', pad_inches=pad_inches, bbox_inches='tight', dpi=dpi)
                print('Figure has saved successfully!')
                plt.close()
            else:
                plt.savefig(saving_directory + '/' + explanation + '.svg', pad_inches=pad_inches, bbox_inches='tight', dpi=dpi)
                print('Figure has saved successfully!')
                plt.close()
        else:
            if(extra is not None):
                plt.savefig(saving_directory + '/' + explanation + '_' + extra + '.jpeg', pad_inches=pad_inches, bbox_inches='tight', dpi=dpi)
                print('Figure has saved successfully!')
                plt.close()
            else:
                plt.savefig(saving_directory + '/' + explanation + '.jpeg', pad_inches=pad_inches, bbox_inches='tight', dpi=dpi)
                print('Figure has saved successfully!')
                plt.close()
            
#%% ============ Functions to Reduce memory Usage ===================
    def raw_file_to_epochs(self, raw_file_list, event_id, tmin=-1, tmax=4, l_pass=None, h_pass=None, CSD=False):
        
        # ========= Read Data =============
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_file_list]) #concatenate
        eegbci.standardize(raw)  # set channel names
        #=== Name Standardize ====
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage)
        raw.rename_channels(lambda x: x.strip('.'))
        # ========= Read Data =============
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                            exclude='bads')
        
        events, _ = events_from_annotations(raw, event_id=event_id)
        
        # (re)load the data to save memory
        raw.filter(l_pass, h_pass, fir_design='firwin', skip_by_annotation='edge')
        raw.load_data()
        
        if(CSD==True):
            raw = self.CSD(raw)
        
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, picks=picks, preload=True)
        
        return epochs, events, picks
#%% ===================== Rejection Functions ===================
    def convert_total_rejection_intofilebyfile_rejection(self, epochs_list, rejected_indexes_file):
        
        os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Rejected Epochs')
        getrejected_indexes = pickle.load(open(rejected_indexes_file,'rb'))
        
        epoch_lengths = np.zeros(len(epochs_list))
        for i in range(len(epochs_list)):
            epoch_lengths[i] = np.size(epochs_list[i]._data,0)
            
        begin_end_indexes = np.zeros((len(epochs_list), 2))
        begin_end_indexes[0,0], begin_end_indexes[0,1] = 0, epoch_lengths[0] 
        curr_index = epoch_lengths[0] 
        for i in range(1,len(epochs_list)):
            begin_end_indexes[i,0] = curr_index 
            curr_index += epoch_lengths[i]
            begin_end_indexes[i,1] = curr_index
        
        #==== Convert global indexes into base file by file ====
        get_rejected_indexes_list = list()
        for i in range(len(epochs_list)):
            get_rejected_indexes_list.append(getrejected_indexes[np.logical_and(getrejected_indexes > begin_end_indexes[i,0], \
                                                       getrejected_indexes < begin_end_indexes[i,1])] - begin_end_indexes[i,0])
        #==== Convert global indexes into base file by file ====
    
        return get_rejected_indexes_list
#%% =========== Auto sleep staging =================
    def yasa_advanced_sleep_staging(self, raw, eeg_channels, eog, emg, metadata=None):
        
        sls_F3 = yasa.SleepStaging(raw, eeg_name=eeg_channels[0], eog_name=eog, emg_name=emg, metadata=dict(age=21, male=False))
        sls_F4 = yasa.SleepStaging(raw, eeg_name=eeg_channels[1], eog_name=eog, emg_name=emg, metadata=dict(age=21, male=False))
        sls_C3 = yasa.SleepStaging(raw, eeg_name=eeg_channels[2], eog_name=eog, emg_name=emg, metadata=dict(age=21, male=False))
        sls_C4 = yasa.SleepStaging(raw, eeg_name=eeg_channels[3], eog_name=eog, emg_name=emg, metadata=dict(age=21, male=False))
        sls_O1 = yasa.SleepStaging(raw, eeg_name=eeg_channels[4], eog_name=eog, emg_name=emg, metadata=dict(age=21, male=False))
        sls_O2 = yasa.SleepStaging(raw, eeg_name=eeg_channels[5], eog_name=eog, emg_name=emg, metadata=dict(age=21, male=False))
        
        # Get the predicted probabilities
        proba_F3 = sls_F3.predict_proba()
        proba_F4 = sls_F4.predict_proba()
        proba_C3 = sls_C3.predict_proba()
        proba_C4 = sls_C4.predict_proba()
        proba_O1 = sls_O1.predict_proba()
        proba_O2 = sls_O2.predict_proba()
        
        proba_avg = (proba_F3 + proba_F4 + proba_C3 + proba_C4 + proba_O1 + proba_O2) / 6
        proba_avg = proba_avg.to_numpy()
        
        #softmax layer
        sleep_stages = np.argmax(proba_avg, axis=1)
        
        return sleep_stages
#%% ============ FOOF =============
    def FOOOF_group(self, spectra, freqs, fmin=2, fmax=48):
        
        fg = FOOOFGroup()
        fg.fit(freqs, spectra, [fmin, fmax])
        exps = fg.get_params('aperiodic_params', 'exponent')
        r2 = fg.get_params('r_squared')
        
        return np.mean(exps), np.mean(r2)
