import tqdm
import os
import glob
import pandas as pd
import numpy as np
import time

import torch
from dcase_utils import get_label_train as get_label_train_fn
from dcase_utils import get_label_valid as get_label_valid_fn
from sed_utils import load_wave, get_segments_and_labels, get_segments_and_labels_new
import sed_utils

class PrototypeDataset(torch.utils.data.Dataset):
    def __init__(self, wave, annotations, window_size, hop_size, sample_rate, transform=None):

        xs = []
        times = []
        for (start_time, end_time) in annotations:
            start_idx = int(np.ceil(start_time * sample_rate))
            end_idx   = int(np.floor(end_time * sample_rate))

            ann_window_size = end_idx - start_idx

            # TODO: consider how to expand this in the best way
            if window_size - ann_window_size > 0:
                to_pad = window_size - ann_window_size
            else:
                to_pad = 0

            wave_segment = wave[start_idx:end_idx]
            wave_segment = np.pad(wave_segment, int(np.ceil(to_pad / 2)))
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
        return x

class BioacousticDatasetNew(torch.utils.data.Dataset):
    """Bioacoustic dataset."""
    def __init__(self, csv_paths, window_size, hop_size, sample_rate, n_classes, n_time, n_shot=1000000, n_background=1000000, transform=None, cache=True, is_validation_data=False, use_old=True):
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
                cache           : If the dataset should be cached to disk.
                is_validation_data : If the dataset should be loaded as validation data.
        """
        self.sample_rate = sample_rate
        self.transform = transform

        if is_validation_data:
            get_label_fn = get_label_valid_fn
        else:
            get_label_fn = get_label_train_fn

        wav_paths = [x.replace('csv', 'wav') for x in csv_paths]

        # I need to optimize this memory problem
        #n_segments = 0
        #t1 = time.time()
        #for wav_path in wav_paths:
        #    wave, sample_rate = sed_utils.load_wave(wav_path)
        #    segments, segment_intervals = sed_utils.split_into_segments(wave, sample_rate, hop_size, window_size)
        #    n_segments += len(segments)

        #t2 = time.time()
        #print("time: ", t2-t1)
        #print("n_segments: ", n_segments)

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

        for wav_path, csv_path in tqdm.tqdm(list(zip(wav_paths, csv_paths)), disable=is_validation_data):
            # load the wave file
            wave, sample_rate = load_wave(wav_path)
            sample_rates.append(sample_rate)
            annotation_df = pd.read_csv(csv_path)

            sig_segs, sig_seg_targets, sig_intervals, bg_segs, bg_seg_targets, bg_intervals = sed_utils.get_segments_and_labels_new(
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

    def __len__(self):
            return len(self.x_sig) + len(self.x_bg)

    def __getitem__(self, idx):
            if torch.is_tensor(idx):
                    idx = idx.tolist()

            # Keep track of both background and signal
            if idx < len(self.x_sig):
                x = self.x_sig[idx]
                y = self.y_sig[idx]
            else:
                x = self.x_bg[idx-len(self.x_sig)]
                y = self.y_bg[idx-len(self.x_sig)]

            if self.transform:
                    x = self.transform(x)

            return x, y

#class BioacousticDataset(torch.utils.data.Dataset):
#        """Bioacoustic dataset."""
#
#        def __init__(self, root_dir, window_size, hop_size, sample_rate, n_classes, n_time, n_shot=1000000, n_background=1000000, transform=None, cache=True, is_validation_data=False, use_old=True):
#                """
#                Args:
#                        root_dir        : Training_Set with all wav files and annotations.
#                        window_size : The number of samples for each input segment, should be on form 2^i.
#                        hop_size        : The number of samples until the next window, should be on form window_size / (2^i).
#                        n_classes       : The number of classes in the annotations.
#                        n_time          : The number of perdictions for each segment, should be on form 2^i
#                        n_shot          : The maximum number of annotatated segments to use.
#                        n_background    : The maximum number of background segments to load.
#                        transform       : Optional transform to be applied on a sample.
#                        cache           : If the dataset should be cached to disk.
#                        is_validation_data : If the dataset should be loaded as validation data.
#                """
#                self.sample_rate = sample_rate
#                self.transform = transform
#
#                if is_validation_data:
#                    get_label_fn = get_label_valid_fn
#                    assert(root_dir.endswith('.csv'))
#                    csv_paths = [root_dir]
#                    basename = os.path.basename(root_dir)
#                    cache_dir = os.path.join('cache/valid', basename.split('.')[0])
#                else:
#                    get_label_fn = get_label_train_fn
#                    cache_dir = 'cache/train'
#                    assert(root_dir.endswith('/'))
#                    csv_paths = glob.glob(os.path.join(root_dir, '*/*.csv'))
#
#                x_bg_path = 'x_bg_{}_{}_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, n_background, n_shot, sample_rate)
#                x_sig_path = 'x_sig_{}_{}_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, n_background, n_shot, sample_rate)
#                y_bg_path = 'y_bg_{}_{}_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, n_background, n_shot, sample_rate)
#                y_sig_path = 'y_sig_{}_{}_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, n_background, n_shot, sample_rate)
#                intervals_path = 'intervals_{}_{}_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, n_background, n_shot, sample_rate)
#                if cache and os.path.exists(os.path.join(cache_dir, x_bg_path)):
#                    #print("loading cached data ...")
#                    self.x_bg = np.load(os.path.join(cache_dir, x_bg_path))
#                    self.y_bg = np.load(os.path.join(cache_dir, y_bg_path))
#                    self.x_sig = np.load(os.path.join(cache_dir, x_sig_path))
#                    self.y_sig = np.load(os.path.join(cache_dir, y_sig_path))
#                    self.intervals = np.load(os.path.join(cache_dir, intervals_path))
#                else:
#                    #print("building dataset ...")
#
#                    wav_paths = [x.replace('csv', 'wav') for x in csv_paths]
#                    
#                    self.csv_paths = csv_paths
#                    self.wav_paths = wav_paths
#                    self.transform = transform
#
#                    xs = []
#                    ys = []
#                    
#                    sig_segss = []
#                    sig_seg_targetss = []
#                    sig_intervalss = []
#                    
#                    bg_segss = []
#                    bg_seg_targetss = []
#                    bg_intervalss = []
#                    
#                    sample_rates = []
#
#                    # TODO: possibly a loop over the data first to compute the allocation needed
#                    # 1. compute the allocated array needed
#                    # 2. allocate the array
#                    # 3. populate the array
#                    
#                    for wav_path, csv_path in tqdm.tqdm(list(zip(wav_paths, csv_paths)), disable=is_validation_data):
#                        #print("------------------------------------------")
#                        #print("- csv file: ", os.path.basename(csv_path))
#                        #print("------------------------------------------")
#                        
#                        t1 = time.time()
#                        wave, sample_rate = load_wave(wav_path)
#                        sample_rates.append(sample_rate)
#                        annotation_df = pd.read_csv(csv_path)
#                        t2 = time.time()
#                        #print("data loading time: ", t2-t1)
#
#                        t1 = time.time()
#                        if use_old:
#                            sig_segs, sig_seg_targets, sig_intervals, bg_segs, bg_seg_targets, bg_intervals = get_segments_and_labels(
#                                    wave, sample_rate, annotation_df, n_shot, n_background, hop_size, window_size, n_classes, n_time, get_label_fn
#                            )
#                        else:
#                            sig_segs, sig_seg_targets, sig_intervals, bg_segs, bg_seg_targets, bg_intervals = get_segments_and_labels_new(
#                                    wave, sample_rate, annotation_df, n_shot, n_background, hop_size, window_size, n_classes, n_time, get_label_fn
#                            )
#
#                        t2 = time.time()
#                        #print("annotation loading time: ", t2-t1)
#
#                        sig_segss.append(sig_segs)
#                        sig_seg_targetss.append(sig_seg_targets)
#                        sig_intervalss.append(sig_intervals)
#                        #print("sig_segs: ", len(sig_segs))
#                        
#                        if len(bg_segs) > 0:
#                            #print("bg_segs: ", bg_segs.shape)
#                            bg_segss.append(bg_segs)
#                            bg_seg_targetss.append(bg_seg_targets)
#                            bg_intervalss.append(bg_intervals)
#                    
#                    #print("")
#                    assert(len(list(set(sample_rates))) == 1)
#                    assert(sample_rates[0] == self.sample_rate)
#                    
#                    if len(bg_segss) > 0:
#                        x_bg = np.concatenate(bg_segss)
#                        y_bg = np.concatenate(bg_seg_targetss)
#                        bg_intervals = np.concatenate(bg_intervalss)
#                    else:
#                        n_background = 0
#                        print("There were no background in: ", root_dir)
#                    
#                    x_sig = np.concatenate(sig_segss)
#                    #print("x_sig: ", x_sig.shape)
#                    y_sig = np.concatenate(sig_seg_targetss)
#                    sig_intervals = np.concatenate(sig_intervalss)
#                    
#                    if n_background > 0:
#                        #print("x_bg: ", x_bg.shape)
#
#                        self.x_bg = x_bg
#                        self.y_bg = y_bg
#
#                        self.x_sig = x_sig
#                        self.y_sig = y_sig
#                        #self.x = np.concatenate((x_sig, x_bg))
#                        #self.y = np.concatenate((y_sig, y_bg))
#                        #self.sig_intervals = sig_intervals
#                        #self.bg_intervals  = bg_intervals
#                        self.intervals = np.concatenate((sig_intervals, bg_intervals))
#                    else:
#                        self.x_sig = x_sig
#                        self.y_sig = y_sig
#                        self.intervals = sig_intervals
#                        self.x_bg = np.array([])
#                        self.y_bg = np.array([])
#                        #self.sig_intervals = sig_intervals
#                
#                    if cache:
#                        if not os.path.exists(cache_dir):
#                            os.makedirs(cache_dir)
#                        np.save(os.path.join(cache_dir, x_bg_path), self.x_bg)
#                        np.save(os.path.join(cache_dir, y_bg_path), self.y_bg)
#                        np.save(os.path.join(cache_dir, x_sig_path), self.x_sig)
#                        np.save(os.path.join(cache_dir, y_sig_path), self.y_sig)
#                        np.save(os.path.join(cache_dir, intervals_path), self.intervals)
#
#        def __len__(self):
#                return len(self.x_sig) + len(self.x_bg)
#
#        def __getitem__(self, idx):
#                if torch.is_tensor(idx):
#                        idx = idx.tolist()
#
#                # Keep track of both background and signal
#                if idx < len(self.x_sig):
#                    x = self.x_sig[idx]
#                    y = self.y_sig[idx]
#                else:
#                    x = self.x_bg[idx-len(self.x_sig)]
#                    y = self.y_bg[idx-len(self.x_sig)]
#
#                #x = self.x[idx]
#                #y = self.y[idx]
#
#                if self.transform:
#                        x = self.transform(x)
#
#                return x, y
#
#        #def get_interval_by_idx(idx):
#        #    if idx < len(self.sig_intervals):
#        #        interval = self.sig_intervals[idx]
#        #    else:
#        #        interval = self.bg_intervals[idx-len(self.sig_intervals)]
#
#
