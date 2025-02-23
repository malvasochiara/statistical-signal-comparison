# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def sinusoidal_signal( N = 10,freqs=None, sampling_rate=250 ,ampls=None, duration=1):
    """
    Creates a signal made by a sum of sinusoidal waves.
    All input parameters can be set to None and in that case they are randomly extracted.
    
    INPUT:
        - N:             number of components for each signal. Default is 10.
        - duration:      temporal extent of the signal in seconds. Default is 1 s.
        - sampling_rate: sampling frequency of the signal in Hz. Default is 250 Hz.
        - freqs:         array of frequencies of the individual sine waves that compose the final signal.
        - ampls:         array of amplitudes of the individual sine waves that compose the final signal
        
    OUTPUT:
        - freqs:  array of frequencies of the individual sine waves that compose the final signal
        - t:      array of time points
        - signal: array containing the signal points
    """

    if freqs is not None and ampls is not None and len(ampls) != len(freqs):
          raise ValueError(" Frequencies and amplitudes must have the same length!")

    if freqs is None:
        freqs=np.linspace(1,math.floor(sampling_rate/2),N, dtype=int)
        
    if ampls == None:
        ampls=np.array(np.ones(N))
    
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    signal = np.zeros_like(t)

    # summing all the sine waves
    for f, a in zip(freqs, ampls):
        signal += a * np.sin(2 * np.pi * f * t)

    return signal, t

def cosinusoidal_signal( N = 10,freqs=None, sampling_rate=250 ,ampls=None, duration=1):
    """
    Creates a signal made by a sum of cosinusoidal waves.
    All input parameters can be set to None and in that case they are randomly extracted.
    
    INPUT:
        - N:             number of components for each signal. Default is 10.
        - duration:      temporal extent of the signal in seconds. Default is 1 s.
        - sampling_rate: sampling frequency of the signal in Hz. Default is 250 Hz.
        - freqs:         array of frequencies of the individual sine waves that compose the final signal.
        - ampls:         array of amplitudes of the individual sine waves that compose the final signal
        
    OUTPUT:
        - freqs:  array of frequencies of the individual sine waves that compose the final signal
        - t:      array of time points
        - signal: array containing the signal points
    """
    # METTI UN RAISE ERROR SE I PARAMETRI PASSATI IN INPUT NON RISPETTANO IL TEOREMA DI NYQUIST
    if freqs is not None and ampls is not None and len(ampls) != len(freqs):
          raise ValueError(" Frequencies and amplitudes must have the same length!")

    if freqs is None:
        freqs=np.linspace(1,math.floor(sampling_rate/2),N, dtype=int)
        
    if ampls == None:
        ampls=np.array(np.ones(N))
    
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    signal = np.zeros_like(t)

    # summing all the sine waves
    for f, a in zip(freqs, ampls):
        signal += a * np.cos(2 * np.pi * f * t)

    return signal, t

def mixed_trig_signal(N=10, freqs=None, sampling_rate=250, ampls=None, duration=1):
    """
    Creates a signal made by a sum of sinusoidal and cosinusoidal waves, randomly selected.
    
    INPUT:
        - N:             number of components for each signal. Default is 10.
        - duration:      temporal extent of the signal in seconds. Default is 1 s.
        - sampling_rate: sampling frequency of the signal in Hz. Default is 250 Hz.
        - freqs:         array of frequencies of the individual sine/cosine waves that compose the final signal.
        - ampls:         array of amplitudes of the individual sine/cosine waves that compose the final signal.
        
    OUTPUT:
        - t:      array of time points.
        - signal: array containing the generated signal.
    """
    if freqs is not None and ampls is not None and len(ampls) != len(freqs):
        raise ValueError("Frequencies and amplitudes must have the same length!")
    
    if freqs is None:
        freqs = np.linspace(1, math.floor(sampling_rate/2), N, dtype=int)
    
    if ampls is None:
        ampls = np.ones(N)
    
    if any(f > sampling_rate / 2 for f in freqs):
        raise ValueError("Frequencies must respect the Nyquist theorem (f < sampling_rate / 2)")
    
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    signal = np.zeros_like(t)
    
    # Randomly selecting between sine and cosine for each frequency
    trig_choices = np.random.choice([np.sin, np.cos], size=N)
    
    for func, f, a in zip(trig_choices, freqs, ampls):
        signal += a * func(2 * np.pi * f * t)
    
    return signal, t

def generate_white_noise(snr_db, signal_power=1.0, duration=1.0, sampling_rate=250):
    """
    Generates white noise based on a given SNR in decibels.
    
    INPUT:
        - snr_db:       Signal-to-noise ratio in decibels (dB).
        - signal_power: Power of the reference signal. Default is 1.0.
        - duration:     Duration of the noise in seconds. Default is 1 second.
        - sampling_rate:Sampling rate in Hz. Default is 250 Hz.
    
    OUTPUT:
        - noise:        Array of white noise values.
    """
    # Compute the number of samples
    n_samples = int(duration * sampling_rate)
    
    # Convert SNR from decibels to a linear scale
    snr_linear = 10**(snr_db / 10)
    
    # Calculate noise power based on signal power and SNR
    noise_power = signal_power / snr_linear
    
    # Generate white noise with the calculated power
    noise = np.sqrt(noise_power) * np.random.randn(n_samples)
    
    #window = scipy.signal.windows.hamming(len(noise))
    #white_noise_windowed = noise * window

    
    return noise

def generate_colored_noise(snr_db, slope, signal_power=1.0, duration=1.0, sampling_rate=250):
    # Number of samples
    n_samples = int(duration * sampling_rate)

    # Convert SNR from decibels to a linear scale
    snr_linear = 10**(snr_db / 10)
    
    # Calculate noise power based on signal power and SNR
    noise_power = signal_power / snr_linear
    
    # Generate white noise with the calculated power
    white_noise = np.sqrt(noise_power) * np.random.randn(n_samples)
    # Remove DC offset (mean)
    #white_noise -= np.mean(white_noise)
    '''
    window = scipy.signal.windows.hamming(len(white_noise))
    white_noise_windowed = white_noise * window
    '''
    white_noise_windowed = white_noise
    # FFT of white noise
    white_noise_spectrum = np.fft.fft(white_noise_windowed)
    freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)
    
    # define the line
    y = slope*freqs
    colored_noise_spectrum = white_noise_spectrum + y
    
    #colored_noise_spectrum_real = white_noise_spectrum.real + y
    
    #colored_noise_spectrum = colored_noise_spectrum_real + np.complex64(1j)*white_noise_spectrum.imag
    # IFFT to obtain time-domain colored noise
    colored_noise = np.fft.ifft(colored_noise_spectrum, n=n_samples)
    
    return colored_noise


def plot_signals_and_fft(snr_db=10, N=10, duration=1, sampling_rate=250):
    # Generate clean sinusoidal signal
    signal, t = sinusoidal_signal(N=N, sampling_rate=sampling_rate, duration=duration)
    
    # Generate white noise with the specified SNR
    noise = generate_white_noise(snr_db, signal_power=np.var(signal), duration=duration, sampling_rate=sampling_rate)
    
    # Add noise to the clean signal to create noisy signal
    noisy_signal = signal + noise
    
    # Perform FFT on both signals
    def compute_fft(signal, sampling_rate):
        N = len(signal)
        freqs = fftfreq(N, 1 / sampling_rate)
        fft_vals = fft(signal)
        return freqs[:N // 2], np.abs(fft_vals)[:N // 2]
    
    freqs, clean_fft = compute_fft(signal, sampling_rate)
    _, noisy_fft = compute_fft(noisy_signal, sampling_rate)
    
    # Create the figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot clean signal
    axs[0, 0].grid(True, zorder=0)  # Mette la griglia dietro
    axs[0, 0].plot(t, signal, label="Clean Signal", color='orange', zorder=1)  # Pone il grafico sopra la griglia
    axs[0, 0].set_title("Clean Signal")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].legend()
    
    # Plot noisy signal
    axs[0, 1].grid(True, zorder=0)
    axs[0, 1].plot(t, noisy_signal, label="Noisy Signal", color='purple', zorder=1)
    axs[0, 1].set_title("Noisy Signal (with white noise)")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].legend()
    
    # Plot FFT of clean signal
    axs[1, 0].grid(True, zorder=0)
    axs[1, 0].plot(freqs, clean_fft, label="FFT of Clean Signal", color='orange', zorder=1)
    axs[1, 0].set_title("FFT of Clean Signal")
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 0].set_ylabel("Magnitude")
    axs[1, 0].legend()
    
    # Plot FFT of noisy signal
    axs[1, 1].grid(True, zorder=0)
    axs[1, 1].plot(freqs, noisy_fft, label="FFT of Noisy Signal", color='purple', zorder=1)
    axs[1, 1].set_title("FFT of Noisy Signal")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

    
def plot_signals_and_fft_colored_noise(snr_db=10, slope=-1, N=10, duration=1, sampling_rate=250):
    # Generate clean sinusoidal signal
    signal, t = sinusoidal_signal(N=N, sampling_rate=sampling_rate, duration=duration)
    
    # Generate colored noise with the specified SNR and slope
    noise = generate_colored_noise(snr_db, slope, signal_power=np.var(signal), duration=duration, sampling_rate=sampling_rate)
    
    # Add noise to the clean signal to create noisy signal
    noisy_signal = signal + noise
    
    # Perform FFT on both signals
    def compute_fft(signal, sampling_rate):
        N = len(signal)
        freqs = fftfreq(N, 1 / sampling_rate)
        fft_vals = fft(signal)
        return freqs[:N // 2], np.abs(fft_vals)[:N // 2]
    
    freqs, clean_fft = compute_fft(signal, sampling_rate)
    _, noisy_fft = compute_fft(noisy_signal, sampling_rate)
    
    # Create the figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot clean signal
    axs[0, 0].plot(t, signal, label="Clean Signal", color='orange', zorder=1)
    axs[0, 0].set_title("Clean Signal")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True, zorder=0)  # Grid behind
    axs[0, 0].legend()
    
    # Plot noisy signal
    axs[0, 1].plot(t, noisy_signal, label="Noisy Signal", color='purple', zorder=1)
    axs[0, 1].set_title("Noisy Signal (with colored noise)")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].grid(True, zorder=0)  # Grid behind
    axs[0, 1].legend()
    
    # Plot FFT of clean signal
    axs[1, 0].plot(freqs, clean_fft, label="FFT of Clean Signal", color='orange', zorder=1)
    axs[1, 0].set_title("FFT of Clean Signal")
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 0].set_ylabel("Magnitude")
    axs[1, 0].grid(True, zorder=0)  # Grid behind
    axs[1, 0].legend()
    
    # Plot FFT of noisy signal
    axs[1, 1].plot(freqs, noisy_fft, label="FFT of Noisy Signal", color='purple', zorder=1)
    axs[1, 1].set_title("FFT of Noisy Signal")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].grid(True, zorder=0)  # Grid behind
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

