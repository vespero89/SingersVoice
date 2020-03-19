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
import os


def wav_file_list(source):
    # list all file in source directory
    filenames = []
    for (dirpath, dirnames, filenames) in os.walk(source):
        break
    # drop all non wav file
    wav_filenames = [f for f in filenames if f.lower().endswith('.wav')]

    return wav_filenames


# calcola i mel, i delta e i delta-deltas
def log_mel(filepath, out_folder, fs, N, overlap, win_type='hamming', n_mels=128, fmin=0.0, fmax=None, htk=True):
    # Load an audio file as a floating point time series

    # x, fs = librosa.core.load(filepath, sr=fs, offset=5.0)
    # x_trimmered, index = librosa.effects.trim(x)
    # chunks = librosa.effects.split(x_trimmered, frame_length=5*fs, hop_length=1*fs)

    stream = librosa.stream(filepath, block_length=1, frame_length=5*fs, hop_length=1*fs, fill_value=0)
    i = 0
    for x_chunks in stream:
        image_filename = os.path.join(dest_path, out_folder, os.path.splitext(os.path.basename(filepath))[0] + '_' + str(i))
        image_filename = image_filename.replace('_16bit', '')
        # Power spectrum
        S = np.abs(librosa.core.stft(x_chunks, n_fft=N, window=signal.get_window(win_type, N), hop_length=N-overlap, center=False)) ** 2
        # Build a Mel filter
        mel_basis = librosa.filters.mel(fs, N, n_mels, fmin, fmax, htk)
        # Filtering
        mel_filtered = np.dot(mel_basis, S)

        coefficients = librosa.core.power_to_db(mel_filtered)
        plt.figure()
        plt.imshow(coefficients)
        plt.savefig(image_filename + '.png')
        i += 1

   # delta = librosa.feature.delta(mel_filtered, delta_width*2+1, order=1, axis=-1)
    #coefficients = np.concatenate((coefficients, delta))
    # add delta e delta-deltas
    # coefficients.append(librosa.feature.delta(mel_filtered, delta_width*2+1, order=1, axis=-1))
    # coefficients.append(librosa.feature.delta(mel_filtered, delta_width*2+1, order=2, axis=-1))

    return True


def extract_log_mel(source, dest, fs, N, overlap, win_type='hamming', n_mels=128, fmin=0.0, fmax=None, htk=True):
    wav_filenames = wav_file_list(source)
    for w in wav_filenames:
        singer_name = os.path.basename(w).split('-')[0]
        os.makedirs(os.path.join(dest, singer_name), exist_ok=True)
        mels = log_mel(os.path.join(source, w), singer_name, fs, N, overlap, win_type, n_mels, fmin, fmax, htk)
        if mels:
            print('Done {}'.format(w))
        # np.save(path.join(dest, w[0:-4]), mels)


def normalize_audio(source, dest):
    wav_filenames = wav_file_list(source)
    for w in wav_filenames:
        x, fs = librosa.load(os.path.join(source, w), sr=Fs)
        librosa.output.write_wav(os.path.join(dest, w), x, sr=Fs, norm=True)


if __name__ == "__main__":

    root_dir = os.path.realpath('../../')

    wav_dir_path = os.path.join(root_dir, 'dataset_wavs_norm')
    dest_path = os.path.join('dataset', 'spectrograms_normalized')

    if (not os.path.exists(dest_path)):
        os.makedirs(dest_path)

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

    # normalize_audio(wav_dir_path, dest_path)

    extract_log_mel(wav_dir_path, dest_path, fs=Fs, N=window_length, overlap=overlap, win_type=window_type, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk)

    import os

    print(os.path.realpath('.'))
