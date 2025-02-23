# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import time
import generate_and_plot_signal.py as sgen
from tqdm import tqdm
from scipy.stats import linregress
from itertools import combinations
import matplotlib as plt



##################################################################################
                             # DATASET GENERATION #
##################################################################################

signals = []
labels = []
base_frequencies = []
clean_signal = []
s_rate = 500
duration = 2
N_signals = 15 # number of different signals
N_signal_noisy = 20 # number of noisy signal, i.e. number of copy of the same signal
N_max_components = 10 # maximum number of components
snr_db= np.linspace(10,100, N_signal_noisy)
slope = 0.6
noise = "white" # "colored"
    
    
for ii in tqdm(range(N_signals)):
    N = random.randint(int(N_max_components/2), N_max_components) # number of components of this signal
    freqs = np.zeros(N, dtype=int)
    
    for i in range(N):
        freqs[i]=random.randint(1,math.floor(s_rate/2)) # Nyquist's theorem
        #freqs[i]=random.randint(1,50) # Nyquist's theorem
    

        signal,t = sgen.sinusoidal_signal( N ,freqs, sampling_rate=s_rate ,ampls=None, duration=duration)

    clean_signal.append(signal)
    
    signal_power = np.mean(signal**2)
    
    for jj in range(N_signal_noisy):
        #CHOOSE THE APPROPRIATE NOISE FUNCTION
        # USING WHITE NOISE FOR NOW
        if noise == "white":
            white_noise = sgen.generate_white_noise(snr_db[jj], signal_power, duration=duration, sampling_rate=s_rate)
            signals.append(signal + white_noise)
        if noise == "colored":
            colored_noise = sgen.generate_colored_noise(snr_db[jj], slope, signal_power, duration=duration, sampling_rate=s_rate)
            signals.append(signal + colored_noise)

        labels.append(ii)  # Label each signal type
        base_frequencies.append(sorted((freqs), reverse = False))
        time.sleep(0.0001)
        
        
        
##################################################################################       
              # FFTs OF THE DIFFERENCE OF SIGNALS
##################################################################################




# PIPELINE 2: FFT OF THE SIGNAL DIFFERENCE
# i) diff = signal1 - signal2
# ii) fft_diff = fft(diff) (it is a COMPLEX number)
# iii) check if fft_diff.real and/or fft_diff.imag follow a Gaussian distribution

plot = 1  # SET TO 0 IF YOU DO NOT WANT PLOTS

# STEP i: Compute the difference between pairs of signals
diff_SAME = []
diff_DIFF = []

for i, j in tqdm(combinations(range(len(labels)), 2)):
    diff = signals[i] - signals[j]
    if labels[i] == labels[j]:  # Same signal
        diff_SAME.append(diff)
        
    if labels[i] != labels[j]:  # Different signal
        diff_DIFF.append(diff)

# STEP ii: Perform FFT on the differences
fft_diff_SAME = []
fftR_diff_SAME = []
fftI_diff_SAME = []
fft_diff_DIFF = []
fftR_diff_DIFF = []
fftI_diff_DIFF = []

# Select only positive frequencies
freqs_fft = np.fft.fftfreq(duration * s_rate, d=1 / s_rate)
pos_mask = freqs_fft > 0

# Process for SAME signals (those that are labeled as the same signal)
for i in tqdm(range(len(diff_SAME))):
    x = diff_SAME[i]
    ft = np.fft.fft(x)
    fft_diff_SAME.append(ft)
    
    # Perform linear detrending on the real part of the FFT
    slope, intercept, _, _, _ = linregress(freqs_fft[pos_mask], ft.real[pos_mask])
    detrended_R = ft.real[pos_mask] - (slope * freqs_fft[pos_mask] + intercept)
    fftR_diff_SAME.append(detrended_R)
    
    # Perform linear detrending on the imaginary part of the FFT
    slope, intercept, _, _, _ = linregress(freqs_fft[pos_mask], ft.imag[pos_mask])
    detrended_I = ft.imag[pos_mask] - (slope * freqs_fft[pos_mask] + intercept)
    fftI_diff_SAME.append(detrended_I)
    
    '''
    # Alternative linear detrend approach for real part
    ftR_norm = []
    ftR_pos = ft.real[pos_mask]
    for j in range(len(ftR_pos)):
        if ftR_pos[j] >= 0:
            ftR_norm.append(ftR_pos[j] - np.mean(np.abs(ftR_pos)))
        else:
            ftR_norm.append(ftR_pos[j] + np.mean(np.abs(ftR_pos)))
                            
    slope, intercept, _, _, _ = linregress(freqs_fft[pos_mask], ftR_norm)
    detrended_R = ftR_norm - (slope * freqs_fft[pos_mask] + intercept)
    fftR_diff_SAME.append(detrended_R)
    slope, intercept, _, _, _ = linregress(freqs_fft[pos_mask], ft.imag[pos_mask])
    detrended_I = ft.imag[pos_mask] - (slope * freqs_fft[pos_mask] + intercept)
    fftI_diff_SAME.append(detrended_I)
    
    # Z-score normalization of the real part
    mu = np.mean(ft.real[pos_mask])
    sigma = np.std(ft.real[pos_mask])
    detrended_R = (ft.real[pos_mask] - mu) / sigma
    fftR_diff_SAME.append(detrended_R)

    # Z-score normalization of the imaginary part
    mu = np.mean(ft.imag[pos_mask])
    sigma = np.std(ft.imag[pos_mask])
    detrended_I = (ft.imag[pos_mask] - mu) / sigma
    fftI_diff_SAME.append(detrended_I)
    '''
    
# Process for DIFFERENT signals (those that are labeled as different)
for i in tqdm(range(len(diff_DIFF))):
    x = diff_DIFF[i]
    ft = np.fft.fft(x)
    fft_diff_DIFF.append(ft)
    
    # Perform linear detrending on the real part of the FFT
    slope, intercept, _, _, _ = linregress(freqs_fft[pos_mask], ft.real[pos_mask])
    detrended_R = ft.real[pos_mask] - (slope * freqs_fft[pos_mask] + intercept)
    fftR_diff_DIFF.append(detrended_R)
    
    # Perform linear detrending on the imaginary part of the FFT
    slope, intercept, _, _, _ = linregress(freqs_fft[pos_mask], ft.imag[pos_mask])
    detrended_I = ft.imag[pos_mask] - (slope * freqs_fft[pos_mask] + intercept)
    fftI_diff_DIFF.append(detrended_I)

    '''
    # Z-score normalization for real part
    mu = np.mean(ft.real[pos_mask])
    sigma = np.std(ft.real[pos_mask])
    detrended_R = (ft.real[pos_mask] - mu) / sigma
    fftR_diff_DIFF.append(detrended_R)

    # Z-score normalization for imaginary part
    mu = np.mean(ft.imag[pos_mask])
    sigma = np.std(ft.imag[pos_mask])
    detrended_I = (ft.imag[pos_mask] - mu) / sigma
    fftI_diff_DIFF.append(detrended_I)
    '''

# STEP iii: Plotting the results (optional - if plot == 1)
if plot == 0:
    # TEST PLOT - ONLY RUN THIS IF YOU HAVE FEW SIGNALS AND WANT TO SEE THEM
    print(base_frequencies)
    for ii in range(len(fftR_diff_SAME)):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
        
        # Plot the real part of the FFT
        transformR = fftR_diff_SAME[ii]
        axs[0].plot(freqs_fft[pos_mask], transformR)
        axs[0].set_ylabel('Real amplitude')
        axs[0].set_xlabel('Frequency')
        axs[0].grid(True)

        # Plot the imaginary part of the FFT
        transformI = fftI_diff_SAME[ii]
        axs[1].plot(freqs_fft[pos_mask], transformI)
        axs[1].set_ylabel('Imaginary amplitude')
        axs[1].set_xlabel('Frequency')
        axs[1].grid(True)

        # Adjust layout and show the figure
        plt.tight_layout()  # Automatically adjusts spacing between subplots
        plt.show()

