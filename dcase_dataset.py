import tqdm
import os
import glob
import pandas as pd
import numpy as np
import time

import torch
from dcase_utils import get_label_train as get_label_train_fn
from dcase_utils import get_label_valid as get_label_valid_fn
from sed_utils import load_wave, get_segments_and_labels
import sed_utils

class PrototypeDataset(torch.utils.data.Dataset):
    def __init__(self, wave, annotations, window_size, hop_size, sample_rate, transform=None, padding='expand', normalize=False, mean=0, std=1):
        self.normalize = normalize
        self.mean = mean
        self.std = std

        xs = []
        times = []
        for (start_time, end_time) in annotations:
            start_idx = int(np.ceil(start_time * sample_rate))
            end_idx   = int(np.floor(end_time * sample_rate))

            ann_window_size = end_idx - start_idx

            # TODO: consider how to expand this in the best way

            to_expand = window_size - ann_window_size
            if to_expand > 0:
                print("{} shorter than {}, padding beginning and end with {}.".format(ann_window_size, window_size, padding))
                if padding == 'expand':
                    print("expand")
                    start_expand = int(np.floor(to_expand/2))
                    end_expand = int(np.ceil(to_expand/2))
                    wave_segment = wave[start_idx-start_expand:end_idx+end_expand]
                elif padding == 'zeros':
                    print("zeros")
                    wave_segment = wave[start_idx:end_idx]
                    to_pad = int(np.ceil(to_expand/2))
                    wave_segment = np.pad(wave_segment, to_pad)
                elif padding == 'repeat':
                    print("repeat")
                    wave_segment = wave[start_idx:end_idx]
                    nb_repeats = int(np.ceil(to_expand / ann_window_size))
                    wave_segment = np.repeat(wave_segment, nb_repeats)[:window_size]
                else:
                    raise ValueError("{} padding scheme not defined.".format(padding))

            else:
                wave_segment = wave[start_idx:end_idx]

            wave_segments, segment_times = sed_utils.split_into_segments(wave_segment, sample_rate, hop_size, window_size)
            segment_times = [(x[0] + start_time, x[1] + start_time) for x in segment_times]
            xs.append(wave_segments)
            times.append(segment_times)

        self.x = np.concatenate(xs)
        self.times = np.concatenate(times)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
            if self.normalize:
                x = x - self.mean
                x = x / self.std
                #x = np.squeeze(x)

        return x

class BioacousticDataset(torch.utils.data.Dataset):
    """Bioacoustic dataset."""
    def __init__(self, csv_paths, window_size, hop_size, sample_rate, n_classes, n_time, n_shot=1000000, n_background=1000000, transform=None, normalize=True):
        """
        Args:
                csv_paths       : All annotation files.
                window_size     : The number of samples for each input segment, should be on form 2^i.
                hop_size        : The number of samples until the next window, should be on form window_size / (2^i).
                n_classes       : The number of classes in the annotations.
                n_time          : The number of perdictions for each segment, should be on form 2^i
                n_shot          : The maximum number of annotatated segments to use.
                n_background    : The maximum number of background segments to load.
                transform       : Optional transform to be applied on a sample.
        """
        self.normalize = normalize
        self.sample_rate = sample_rate
        self.transform = transform
        get_label_fn = get_label_train_fn

        wav_paths = [x.replace('csv', 'wav') for x in csv_paths]

        # I need to optimize this memory problem
        self.csv_paths = csv_paths
        self.wav_paths = wav_paths
        self.transform = transform

        # this is problematic for memory
        xs = []
        ys = []

        sig_segss = []
        sig_seg_targetss = []
        sig_intervalss = []

        bg_segss = []
        bg_seg_targetss = []
        bg_intervalss = []

        sample_rates = []

        for wav_path, csv_path in tqdm.tqdm(list(zip(wav_paths, csv_paths))):
            # load the wave file
            wave, sample_rate = load_wave(wav_path)
            sample_rates.append(sample_rate)
            annotation_df = pd.read_csv(csv_path)

            sig_segs, sig_seg_targets, sig_intervals, bg_segs, bg_seg_targets, bg_intervals = sed_utils.get_segments_and_labels(
                    wave, sample_rate, annotation_df, n_shot, n_background, hop_size, window_size, n_classes, n_time, get_label_fn
            )

            sig_segss.append(sig_segs)
            sig_seg_targetss.append(sig_seg_targets)
            sig_intervalss.append(sig_intervals)
            
            if len(bg_segs) > 0:
                bg_segss.append(bg_segs)
                bg_seg_targetss.append(bg_seg_targets)
                bg_intervalss.append(bg_intervals)

        assert(len(list(set(sample_rates))) == 1)
        assert(sample_rates[0] == self.sample_rate)

        if len(bg_segss) > 0:
            x_bg = np.concatenate(bg_segss)
            y_bg = np.concatenate(bg_seg_targetss)
            bg_intervals = np.concatenate(bg_intervalss)
        else:
            n_background = 0
            print("There were no background in: ", csv_paths)

        x_sig = np.concatenate(sig_segss)
        y_sig = np.concatenate(sig_seg_targetss)
        sig_intervals = np.concatenate(sig_intervalss)

        if n_background > 0:
            self.x_bg = x_bg
            self.y_bg = y_bg

            self.x_sig = x_sig
            self.y_sig = y_sig
            self.intervals = np.concatenate((sig_intervals, bg_intervals))
        else:
            self.x_sig = x_sig
            self.y_sig = y_sig
            self.intervals = sig_intervals
            self.x_bg = np.array([])
            self.y_bg = np.array([])

        # compute the mean and std of the transforms
        # only reason to keep wav formats is for future data augmentation
        if self.transform:
            x_sig_tf = []
            x_bg_tf = []
            for wav in tqdm.tqdm(self.x_sig):
                x_sig_tf.append(np.expand_dims(self.transform(wav), axis=0))
            for wav in tqdm.tqdm(self.x_bg):
                x_bg_tf.append(np.expand_dims(self.transform(wav), axis=0))

            #self.x_sig_tf = np.concatenate(x_sig_tf)
            #if n_background > 0:
            #    self.x_bg_tf = np.concatenate(x_bf_tf)
            #else:
            #    self.x_bg_tf = np.array([])

            if self.normalize:
                x = np.concatenate(x_sig_tf + x_bg_tf)
                print(x.shape)

                self.mean = np.mean(x, axis=0, keepdims=False)
                self.std  = np.std(x, axis=0, keepdims=False)

    def __len__(self):
            return len(self.x_sig) + len(self.x_bg)

    def __getitem__(self, idx):
            if torch.is_tensor(idx):
                    idx = idx.tolist()

            # Keep track of both background and signal
            if idx < len(self.x_sig):
                x = self.x_sig[idx]
                #x_tf = self.x_sig_tf[idx]
                y = self.y_sig[idx]
            else:
                x = self.x_bg[idx-len(self.x_sig)]
                #x_tf = self.x_bg_tf[idx-len(self.x_sig)]
                y = self.y_bg[idx-len(self.x_sig)]

            if self.transform:
                x = self.transform(x)
                #x = x_tf
                if self.normalize:
                    x = x - self.mean
                    x = x / self.std
                    #x = np.squeeze(x)


            return x, y
