#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: daniele
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import librosa
from librosa import display
from os import walk, path, makedirs


def wav_file_list(source):
    # list all file in source directory
    filenames = []
    for (dirpath, dirnames, filenames) in walk(source):
        break
    # drop all non wav file
    wav_filenames = [f for f in filenames if f.lower().endswith('.wav')]

    return wav_filenames


# calcola uno spettrogramma
def spectrogram(filepath, fs, N, overlap, win_type='hamming'):
    # Load an audio file as a floating point time series
    x, fs = librosa.core.load(filepath, sr=fs, offset=5.0, duration=5.0)
    # Returns: np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype], dtype=64-bit complex
    X = librosa.core.stft(x, n_fft=N, window=signal.get_window(win_type, N), hop_length=N - overlap, center=False)
    # Sxx = np.abs(X)**2
    Sxx = librosa.core.power_to_db(X)

    return Sxx


# estrae gli spettrogrammi dai file contenuti in source e li salva in dest
def extract_spectrograms(source, dest, fs, N, overlap, win_type='hamming'):
    wav_filenames = wav_file_list(source)
    for w in wav_filenames:
        Sxx = spectrogram(path.join(source, w), fs, N, overlap, win_type)
        np.save(path.join(dest, w[0:-4]), Sxx)


# calcola i mel, i delta e i delta-deltas
def log_mel(filepath, fs, N, overlap, win_type='hamming', n_mels=128, fmin=0.0, fmax=None, htk=True):
    image_filename = os.path.join(dest_path, os.path.splitext(os.path.basename(filepath))[0] + '_' + str(i))
    image_filename = image_filename.replace('_16bit', '')
    # Load an audio file as a floating point time series
    x, fs = librosa.core.load(filepath, sr=fs, offset=5.0, duration=5.0)
    length = np.round(x.shape[0] / fs)
    print('File Length: {}'.format(length))
    # Power spectrum
    S = np.abs(librosa.core.stft(x, n_fft=N, window=signal.get_window(win_type, N), hop_length=N - overlap, center=False)) ** 2
    # Build a Mel filter
    mel_basis = librosa.filters.mel(fs, N, n_mels, fmin, fmax, htk)
    # Filtering
    mel_filtered = np.dot(mel_basis, S)

    coefficients = librosa.core.power_to_db(mel_filtered)
    plt.figure()
    plt.imshow(coefficients)
    plt.savefig(image_filename + '.png')

   # delta = librosa.feature.delta(mel_filtered, delta_width*2+1, order=1, axis=-1)
    #coefficients = np.concatenate((coefficients, delta))
    # add delta e delta-deltas
    # coefficients.append(librosa.feature.delta(mel_filtered, delta_width*2+1, order=1, axis=-1))
    # coefficients.append(librosa.feature.delta(mel_filtered, delta_width*2+1, order=2, axis=-1))

    return coefficients


def mfcc_e(filepath, fs, N, overlap, n_mels=26, fmin=0.0, fmax=None, htk=True, delta_width=None):
    x, fs = librosa.core.load(filepath, sr=fs, offset=5.0, duration=5.0)
    mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mels, hop_length=N - overlap, fmin=fmin, fmax=fmax, htk=htk)
    delta = librosa.feature.delta(mfcc, delta_width * 2 + 1, order=1, axis=-1)
    acc = librosa.feature.delta(mfcc, delta_width * 2 + 1, order=2, axis=-1)
    coefficients = np.vstack((mfcc, delta, acc))
    # coefficients = mfcc

    return coefficients


# estrae i mel e ci calcola i delta e i delta-deltas dai file contenuti in source e li salva in dest
# per la versione log è sufficiente eseguire librosa.logamplitude(.) alla singola sottomatrice o all'intera matrice


def extract_log_mel(source, dest, fs, N, overlap, win_type='hamming', n_mels=128, fmin=0.0, fmax=None, htk=True):
    wav_filenames = wav_file_list(source)
    for w in wav_filenames:
        mels = log_mel(path.join(source, w), fs, N, overlap, win_type, n_mels, fmin, fmax, htk)
        # np.save(path.join(dest, w[0:-4]), mels)


# calcola i m, i delta e i delta-deltas
def extract_MFCC(source, dest, fs, n_mels, N, overlap, fmin, fmax, htk, delta_width):
    wav_filenames = wav_file_list(source)
    for w in wav_filenames:
        mfcc = mfcc_e(path.join(source, w), fs, N, overlap, n_mels, fmin, fmax, htk, delta_width)
        # x, fs = librosa.core.load(path.join(source,w), sr=fs)
        # centr=librosa.feature.spectral_centroid(y=x, sr=fs, n_fft=2048, hop_length=N-overlap, freq=None)
        # mfcc = np.vstack((mfcc,centr))
        # zcr=librosa.feature.zero_crossing_rate(y=x, frame_length=N, hop_length=N-overlap, center=True)
        # mfcc = np.vstack((mfcc, zcr))
        np.save(path.join(dest, w[0:-4]), mfcc)


if __name__ == "__main__":

    root_dir = path.realpath('../../')

    wav_dir_path = path.join(root_dir, 'dataset_16bit')
    dest_path = path.join('dataset', 'spectrograms')

    if (not path.exists(dest_path)):
        makedirs(dest_path)

    window_type = 'hamming'
    fft_length = 1024
    window_length = 1024
    overlap = 256
    Fs = 44100
    n_mels = 128
    fmin = 0.0
    fmax = Fs / 2
    htk = True
    delta_width = 2
    offset = 5
    duration = 5

    # extract_spectrograms(wav_dir_path, dest_path, Fs, fft_length, overlap, window_type)
    extract_log_mel(wav_dir_path, dest_path, Fs, window_length, overlap, window_type, n_mels, fmin, fmax, htk)
    #    extract_MFCC(wav_dir_path, dest_path, Fs, n_mels, window_length, overlap, fmin, fmax, htk, delta_width)

    import os

    print(os.path.realpath('.'))
