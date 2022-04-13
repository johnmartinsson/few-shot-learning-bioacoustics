import tqdm
import os
import glob
import pandas as pd
import numpy as np

import torch
from dcase_utils import get_label_train as get_label_train_fn
from dcase_utils import get_label_valid as get_label_valid_fn
from sed_utils import load_wave, get_segments_and_labels

class BioacousticDataset(torch.utils.data.Dataset):
        """Bioacoustic dataset."""

        def __init__(self, root_dir, window_size, hop_size, sample_rate, n_classes, n_time, n_shot=1000000, n_background=1000000, transform=None, cache=True, is_validation_data=False):
                """
                Args:
                        root_dir        : Training_Set with all wav files and annotations.
                        window_size : The number of samples for each input segment, should be on form 2^i.
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
                    assert(root_dir.endswith('.csv'))
                    csv_paths = [root_dir]
                    basename = os.path.basename(root_dir)
                    cache_dir = os.path.join('cache/valid', basename)
                else:
                    get_label_fn = get_label_train_fn
                    cache_dir = 'cache/train'
                    assert(root_dir.endswith('/'))
                    csv_paths = glob.glob(os.path.join(root_dir, '*/*.csv'))

                x_path = 'x_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, n_background, sample_rate)
                y_path = 'y_{}_{}_{}_{}_{}.npy'.format(window_size, hop_size, n_classes, n_time, n_background, sample_rate)
                if cache and os.path.exists(os.path.join(cache_dir, x_path)):
                    print("loading cached data ...")
                    self.x = np.load(os.path.join(cache_dir, x_path))
                    self.y = np.load(os.path.join(cache_dir, y_path))
                else:
                    print("building dataset ...")

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
                                    wave, sample_rate, annotation_df, n_shot, n_background, hop_size, window_size, n_classes, n_time, get_label_fn
                            )

                            sig_segss.append(sig_segs)
                            sig_seg_targetss.append(sig_seg_targets)
                            
                            if len(bg_segs) > 0:
                                    bg_segss.append(bg_segs)
                                    bg_seg_targetss.append(bg_seg_targets)
                            
                    assert(len(list(set(sample_rates))) == 1)
                    assert(sample_rates[0] == self.sample_rate)
                    

                    if len(bg_segss) > 0:
                        x_bg = np.concatenate(bg_segss)
                        y_bg = np.concatenate(bg_seg_targetss)
                    else:
                        n_background = 0
                        print("There were no background in: ", root_dir)
                    
                    x_sig = np.concatenate(sig_segss)
                    y_sig = np.concatenate(sig_seg_targetss)
                    
                    if n_background > 0:
                            self.x = np.concatenate((x_sig, x_bg[:n_background]))
                            self.y = np.concatenate((y_sig, y_bg[:n_background]))
                    else:
                            self.x = x_sig
                            self.y = y_sig
                    
                    if cache:
                        if not os.path.exists(cache_dir):
                            os.makedirs(cache_dir)
                        np.save(os.path.join(cache_dir, x_path), self.x)
                        np.save(os.path.join(cache_dir, y_path), self.y)

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
