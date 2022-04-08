import tqdm
import os
import glob
import pandas as pd
import numpy as np

import torch
from dcase_utils import get_label as get_label_fn
from sed_utils import load_wave, get_segments_and_labels

class BioacousticBaseDataset(torch.utils.data.Dataset):
        """Bioacoustic base dataset."""

        def __init__(self, root_dir, window_size, hop_size, sample_rate, n_classes, n_time, include_background=False, transform=None, cache=True):
                """
                Args:
                        root_dir        : Training_Set with all wav files and annotations.
                        window_size : The number of samples for each input segment, should be on form 2^i.
                        hop_size        : The number of samples until the next window, should be on form window_size / (2^i).
                        n_classes       : The number of classes in the annotations.
                        n_time          : The number of perdictions for each segment, should be on form 2^i
                        include_background : If samples of assumed background (not annotated) should be included.
                        transform       : Optional transform to be applied on a sample.
                        cache           : If the dataset should be cached to disk.
                """
                self.sample_rate = sample_rate
                self.transform = transform

                x_path = 'x_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, include_background, sample_rate)
                y_path = 'y_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, include_background, sample_rate)
                if cache and os.path.exists(os.path.join('cache', x_path)):
                        self.x = np.load(os.path.join('cache', x_path))
                        self.y = np.load(os.path.join('cache', y_path))
                else:
                        csv_paths = glob.glob(os.path.join(root_dir, '*/*.csv'))[:1]
                        wav_paths = [x.replace('csv', 'wav') for x in csv_paths]
                        
                        self.csv_paths = csv_paths
                        self.wav_paths = wav_paths
                        self.transform = transform

                        xs = []
                        ys = []
                        
                        sig_segss = []
                        sig_seg_targetss = []
                        
                        bg_segss = []
                        bg_seg_targetss = []
                        
                        sample_rates = []
                        
                        for wav_path, csv_path in tqdm.tqdm(list(zip(wav_paths, csv_paths))):
                                wave, sample_rate = load_wave(wav_path)
                                sample_rates.append(sample_rate)
                                annotation_df = pd.read_csv(csv_path)

                                sig_segs, sig_seg_targets, bg_segs, bg_seg_targets = get_segments_and_labels(
                                        wave, sample_rate, annotation_df, hop_size, window_size, n_classes, n_time, get_label_fn
                                )

                                sig_segss.append(sig_segs)
                                sig_seg_targetss.append(sig_seg_targets)
                                
                                if len(bg_segs) > 0:
                                        bg_segss.append(bg_segs)
                                        bg_seg_targetss.append(bg_seg_targets)
                                
                        assert(len(list(set(sample_rates))) == 1)
                        assert(sample_rates[0] == self.sample_rate)
                        

                        x_bg = np.concatenate(bg_segss)
                        y_bg = np.concatenate(bg_seg_targetss)
                        
                        x_sig = np.concatenate(sig_segss)
                        y_sig = np.concatenate(sig_seg_targetss)
                        
                        if include_background:
                                self.x = np.concatenate((x_sig, x_bg))
                                self.y = np.concatenate((y_sig, y_bg))
                        else:
                                self.x = x_sig
                                self.y = y_sig
                        
                        if cache:
                                np.save(os.path.join('cache', x_path), self.x)
                                np.save(os.path.join('cache', y_path), self.y)

        def __len__(self):
                return len(self.x)

        def __getitem__(self, idx):
                if torch.is_tensor(idx):
                        idx = idx.tolist()

                x = self.x[idx]
                y = self.y[idx]

                if self.transform:
                        x = self.transform(x)

                return x, y
