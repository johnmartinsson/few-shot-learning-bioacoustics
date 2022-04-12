import librosa
import pandas as pd
import numpy as np

import time

def get_bioacoustic_pcen_conf():
    return {
	'gain' : 0.8,
	'bias' : 10,
	'power' : 0.25,
	'time_constant' : 0.06,
	'eps' : 1e-6    
    }
def get_speech_pcen_conf():
    return {
	'gain' : 0.98,
	'bias' : 2,
	'power' : 0.5,
	'time_constant' : 0.4,
	'eps' : 1e-6    
    }

def wav_to_pcen(wav, sample_rate, conf):
    window_size = 256 # roughly 25 ms, as used when deriving default params for PCEN, int(sample_rate * 0.025)
    hop_size    = 128 # roughly 10 ms, as used when deriving default params for PCEN, int(sample_rate * 0.010)
    D = librosa.feature.melspectrogram(
        wav, 
        sr=sample_rate,
        win_length=window_size,
        hop_length=hop_size,
        n_mels=40     # used to derive default params for PCEN
    )
    S_pcen = librosa.core.pcen(
        D, 
        sr=sample_rate,
        gain=conf['gain'],
        bias=conf['bias'],
        power=conf['power'],
        time_constant=conf['time_constant'],
        eps=conf['eps']
    )
    return S_pcen

def wav_to_mel(wav, sample_rate):
    window_size = 256 # int(sample_rate * 0.025)
    hop_size    = 128 # int(sample_rate * 0.010)
    D = librosa.feature.melspectrogram(
        wav, 
        sr=sample_rate,
        win_length=window_size,
        hop_length=hop_size,
        n_mels=40
    )
    S_db = librosa.power_to_db(np.abs(D), ref=np.max)
    return S_db

# TODO: Write a test for this
def split_into_segments(wave, sample_rate, hop_size, window_size):

    N = len(wave)
    M = int(np.floor((N-window_size)/hop_size))
    index = np.array([np.arange(window_size) + hop_size*i for i in range(M+1)])
    
    # split the wave into the segments
    segments = wave[index]
    
    # compute the start and end time for each segment
    time_intervals = [(np.min(x)/sample_rate, np.max(x)/sample_rate) for x in index]
    
    return segments, time_intervals

# TODO: Write a test for this
def compute_interval_intersection(i1, i2):
    (a_s, a_e) = i1
    (b_s, b_e) = i2
    if b_s > a_e or a_s > b_e:
        return 0
    else:
        o_s = max(a_s, b_s)
        o_e = min(a_e, b_e)
        return o_e - o_s

# TODO: write a test for this
def compute_interval_union(i1, i2):
    (a_s, a_e) = i1
    (b_s, b_e) = i2
    
    o_s = min(a_s, b_s)
    o_e = max(a_e, b_e)
    
    return o_e - o_s

# TODO: Write a test for this
def compute_interval_intersection_over_union(i1, i2):
    
    intersection = compute_interval_intersection(i1, i2)
    union = compute_interval_union(i1, i2)
    
    if union == 0:
        return 0 #print(i1, i2)
    else:
        return intersection/union

# TODO: write a test for this    
def get_segment_annotation(segment_interval, annotation_interval, sample_rate, window_size):
    # time relative segment
    s_s = segment_interval[0] - segment_interval[0]
    s_e = segment_interval[1] - segment_interval[0]
    
    a_s = annotation_interval[0] - segment_interval[0]
    a_e = annotation_interval[1] - segment_interval[0]
    
    segment_annotation = np.zeros(window_size)
    segment_indices    = np.arange(window_size)
    annotation_indices = np.arange(int(np.floor(a_s * sample_rate)), int(np.floor(a_e * sample_rate)))
    segment_annotation[np.intersect1d(segment_indices, annotation_indices)] = 1
    return segment_annotation
    
def get_segments_and_labels(wave, sample_rate, annotation_df, hop_size, window_size, n_classes, n_time, get_label_fn):
    
    segments, segment_intervals = split_into_segments(wave, sample_rate, hop_size, window_size)
        
    t1 = time.time()
    
    segment_targets = np.zeros((len(segments), n_classes, n_time))
    annotation_intervals, labels = get_annotation_intervals_and_labels(annotation_df, get_label_fn)

    for seg_idx, segment_interval in enumerate(segment_intervals):
        for (annotation_interval, label) in zip(annotation_intervals, labels):
            iou = compute_interval_intersection_over_union(segment_interval, annotation_interval)
            if iou > 0:
                # update target vector
                class_idx = label
                segment_annotation = get_segment_annotation(segment_interval, annotation_interval, sample_rate, window_size)
                
                # max pooling (downsample the target vector)
                annotation_segments, _ = split_into_segments(segment_annotation, sample_rate, window_size//n_time, window_size//n_time)
                annotation = np.max(annotation_segments, axis=1)
                
                segment_targets[seg_idx, class_idx,:] += annotation
    segment_targets = np.clip(segment_targets, 0, 1) # target range [0,1]
    
    t2 = time.time()
    #print("classify_with_annotation: ", t2-t1)
    
    # bool_idx for signal and background
    signal_bool_idx     = np.sum(segment_targets[:,0:n_classes,:], axis=(1, 2)) > 0 # sum over class and time dimension
    background_bool_idx = np.sum(segment_targets[:,0:n_classes,:], axis=(1, 2)) == 0 # sum over class and time dimension

    signal_segments = segments[signal_bool_idx,:]
    signal_segment_targets = segment_targets[signal_bool_idx,:,:]
    
    background_segments = segments[background_bool_idx,:]
    background_segment_targets = segment_targets[background_bool_idx,:,:]
    
    if len(background_segments) < len(signal_segments):
        return signal_segments, signal_segment_targets, [], []
    
    # TODO: maybe thing a bit more about this. Mainly done to save memory space.
    background_random_idx = np.random.choice(np.arange(len(background_segments)), len(signal_segments)) # sample as many as signals
    background_segments = background_segments[background_random_idx]
    background_segment_targets = background_segment_targets[background_random_idx]
    
    return signal_segments, signal_segment_targets, background_segments, background_segment_targets

    
def load_wave(wav_path):
    wave, sample_rate = librosa.load(wav_path, sr=None)
    wave = wave * (2**31) # rescale according to recommendation for PCEN in librosa
    return wave, sample_rate


def get_annotation_interval(row):
    start_time = row[1][1]
    end_time = row[1][2]
    return (start_time, end_time)

def get_annotation_intervals_and_labels(annotation_df, get_label_fn):
    columns = annotation_df.columns
    annotation_intervals = []
    labels = []
    
    for row in annotation_df.iterrows():
        label = get_label_fn(row, columns)
        annotation_interval = get_annotation_interval(row)
        annotation_intervals.append(annotation_interval)
        labels.append(label)
    
    return annotation_intervals, labels

def plot_spectrogram(ax, audio_segment, sample_rate):
    D = librosa.feature.melspectrogram(audio_segment, sr=sample_rate)
    S_db = librosa.power_to_db(np.abs(D), ref=np.max)
    ax.imshow(np.flip(S_db, axis=0), aspect='auto')
