# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import time
from tqdm import tqdm
import generate_and_plot_signal as sgen
import processing_functions as pro


##################################################################################
                             # DATASET GENERATION #
##################################################################################

signals = []
labels = []
base_frequencies = []
clean_signal = []
s_rate = 500
duration = 2
N_signals = 20 # number of different signals
N_signal_noisy = 20 # number of noisy signal, i.e. number of copy of the same signal
N_max_components = 40 # maximum number of components
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
              # DIFFERENCE OF FFTs FOR INDIVIDUAL SIGNALS
##################################################################################

# Steps:
# i) Compute FFT of the signal: fft_signal = fft(signal) (complex-valued)
# i-bis) Apply linear detrending to both real and imaginary parts of FFT
# ii) Compute the difference between FFTs of two signals: fft_diff = fft(signal1) - fft(signal2)



# STEP (i): Compute FFT and apply linear detrending
fft_complex, freqs_fft = pro.compute_complex_fft(signals, s_rate)
# STEP (ii): Compute FFT differences for same and different signals
fftR_diff_SAME, fftI_diff_SAME, fft_diff_SAME, fftR_diff_DIFF, fftI_diff_DIFF, fft_diff_DIFF = pro.compute_fft_differences(fft_complex, labels)

##################################################################################
                        #  CHECK THE RESULTS
##################################################################################

print(pro.perform_shapiro_test_on_concatenated_data(fftI_diff_SAME, fftR_diff_SAME))
#print(pro.perform_shapiro_test_on_concatenated_data(fftI_diff_DIFF, fftR_diff_DIFF, False))
