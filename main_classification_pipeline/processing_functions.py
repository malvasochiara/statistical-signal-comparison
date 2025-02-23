# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from itertools import combinations
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.stats import shapiro


def compute_complex_fft(signal_list, s_rate):
    """
    Computes the complex Fast Fourier Transform (FFT) of a list of signals
    and applies detrending to both the real and imaginary parts.
    
    This function removes linear trends from both components of the FFT,
    ensuring that the transformed signal is centered around zero.
    
    Parameters:
    - signal_list (list of np.ndarray): List of 1D signals.
    - s_rate (float): Sampling rate of the signals.
      
    Returns:
    - fft_complex (list of np.ndarray): List of detrended FFT signals.
    - freqs_fft (np.ndarray): Array of positive frequency components.
    """
    freqs_fft = np.fft.fftfreq(len(signal_list[0]), d=1/s_rate)  # Compute FFT frequencies
    pos_mask = freqs_fft > 0  # Consider only positive frequencies
    fft_complex = []
    
    for signal in tqdm(signal_list, desc="Computing FFT"):
        ft = np.fft.fft(signal)  # Compute FFT
        
        # Detrending the real part
        slope_real, intercept_real, _, _, _ = linregress(freqs_fft[pos_mask], np.real(ft[pos_mask]))
        detrended_real = np.real(ft[pos_mask]) - (slope_real * freqs_fft[pos_mask] + intercept_real)
        
        # Detrending the imaginary part
        slope_imag, intercept_imag, _, _, _ = linregress(freqs_fft[pos_mask], np.imag(ft[pos_mask]))
        detrended_imag = np.imag(ft[pos_mask]) - (slope_imag * freqs_fft[pos_mask] + intercept_imag)
        
        # Reconstruct the detrended complex FFT
        detrended_ft = detrended_real + 1j * detrended_imag
        
        fft_complex.append(detrended_ft)
    
    return fft_complex, freqs_fft[pos_mask]

def compute_fft_differences(fft_complex, labels_train):
    """
    Computes the difference between pairs of FFT signals and normalizes them.
    
    This function calculates the real and imaginary differences between pairs
    of FFT-transformed signals. The differences are then normalized by their
    respective standard deviations. Additionally, the magnitude of the normalized
    difference is computed.
    
    Parameters:
    - fft_complex (list of np.ndarray): List of complex FFT signals.
    - labels_train (list): List of labels associated with each FFT signal.
    
    Returns:
    - fftR_diff_SAME (list of np.ndarray): Normalized real differences for same signals.
    - fftI_diff_SAME (list of np.ndarray): Normalized imaginary differences for same signals.
    - fft_diff_SAME (list of np.ndarray): Magnitude of normalized differences for same signals.
    - fftR_diff_DIFF (list of np.ndarray): Normalized real differences for different signals.
    - fftI_diff_DIFF (list of np.ndarray): Normalized imaginary differences for different signals.
    - fft_diff_DIFF (list of np.ndarray): Magnitude of normalized differences for different signals.
    """
    fftR_diff_SAME = []
    fftI_diff_SAME = []
    fft_diff_SAME = []

    fftR_diff_DIFF = []
    fftI_diff_DIFF = []
    fft_diff_DIFF = []
    
    for i, j in tqdm(combinations(range(len(labels_train)), 2), desc="Computing FFT differences"):
        real_i, imag_i = np.real(fft_complex[i]), np.imag(fft_complex[i])
        real_j, imag_j = np.real(fft_complex[j]), np.imag(fft_complex[j])
        
        diffR = real_i - real_j
        diffI = imag_i - imag_j
        
        # Normalization AFTER computing the difference
        std_diffR = np.std(diffR)
        std_diffI = np.std(diffI)

        if std_diffR > 0 and std_diffI > 0:  # Avoid division by zero
            diffR_norm = diffR / std_diffR
            diffI_norm = diffI / std_diffI
        else:
            diffR_norm = diffR
            diffI_norm = diffI
        
        # Compute the magnitude of the normalized difference
        diff_standard = np.sqrt(diffR_norm**2 + diffI_norm**2)
        
        if labels_train[i] == labels_train[j]:  # Same signal
            fftR_diff_SAME.append(diffR_norm)
            fftI_diff_SAME.append(diffI_norm)
            fft_diff_SAME.append(diff_standard)
        else:  # Different signals
            fftR_diff_DIFF.append(diffR_norm)
            fftI_diff_DIFF.append(diffI_norm)
            fft_diff_DIFF.append(diff_standard)
    
    return fftR_diff_SAME, fftI_diff_SAME, fft_diff_SAME, fftR_diff_DIFF, fftI_diff_DIFF, fft_diff_DIFF


def perform_shapiro_test_on_concatenated_data(data_imaginary, data_real, is_normal_data=True):
    """
    Function to analyze the p-values from the Shapiro-Wilk test applied to the 
    concatenated real and imaginary parts of the input data.

    For each data pair (imaginary and real parts), the data is first standardized 
    by dividing each value by the standard deviation of that part. The real and 
    imaginary parts are then concatenated, and the Shapiro-Wilk test for normality 
    is applied to the combined data.

    Parameters:
    data_imaginary (list of np.ndarray): List of numpy arrays containing the imaginary parts of the data.
    data_real (list of np.ndarray): List of numpy arrays containing the real parts of the data.
    is_normal_data (bool): If True, p-values less than 0.05 are considered "wrong" (non-normal). 
                            If False, p-values greater than 0.05 are considered "wrong" (indicating non-normality where it should be normal).

    Returns:
    float: The percentage of p-values that are considered "wrong" based on the `is_normal_data` condition.
    """

    # List to store p-values
    p_values = []
    
    # Loop through each entry in the input lists and perform the Shapiro-Wilk test
    for i in tqdm(range(len(data_imaginary))):
        # Standardize the imaginary and real data
        standardized_imaginary = np.array(data_imaginary[i] / np.std(data_imaginary[i]))
        standardized_real = np.array(data_real[i] / np.std(data_real[i]))
        
        # Concatenate the standardized real and imaginary parts
        concatenated_data = np.concatenate((standardized_imaginary, standardized_real))
        
        # Perform Shapiro-Wilk test for normality on the concatenated data
        _, p_value = shapiro(concatenated_data)
        p_values.append(p_value)
    
    # Plotting the histogram of p-values with color depending on normality
    plt.figure(figsize=(5, 3))
    plot_color = 'purple' if is_normal_data else 'orange'  # Set color based on is_normal_data
    plt.hist(p_values, bins=20, density=True, alpha=0.5, label='Same signal', color=plot_color)
    plt.xlabel('p')
    plt.ylabel('Frequency')
    plt.title('P values')
    plt.legend()
    plt.show()

    # Count the "wrong" p-values based on the is_normal_data flag
    if is_normal_data:
        count_wrong = sum(p < 0.05 for p in p_values)  # p < 0.05 is "wrong" for normal data
    else:
        count_wrong = sum(p > 0.05 for p in p_values)  # p > 0.05 is "wrong" for non-normal data

    # Calculate and return the percentage of wrong p-values
    return (count_wrong / len(p_values)) * 100
