import pandas as pd
import librosa
import numpy as np
import IPython.display as ipd

def play_first_audio_events(audio_file_path, annotation_df, event_class, event_category, n=5, expand=0):
    audio, sample_rate = librosa.load(audio_file_path, None)
    event_df = annotation_df[annotation_df[event_class] == event_category][:n]
    audio_objects = []
    for index, row in event_df.iterrows():
        start_time = float(row['Starttime'])
        end_time   = float(row['Endtime'])
        
        start_idx = int((start_time-expand)*sample_rate)
        end_idx = int((end_time+expand)*sample_rate)
        
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(audio):
            end_idx = len(audio)
        
        audio_segment = audio[start_idx:end_idx]
        audio_object = ipd.Audio(audio_segment, rate=sample_rate)
        ipd.display(audio_object)
        
def get_first_audio_events(audio_file_path, annotation_df, event_class, event_category, n=5, expand=0):
    audio, sample_rate = librosa.load(audio_file_path, None)
    event_df = annotation_df[annotation_df[event_class] == event_category][:n]
    audio_segments = []
    times = []
    for index, row in event_df.iterrows():
        start_time = float(row['Starttime'])
        end_time   = float(row['Endtime'])
        
        start_idx = int((start_time-expand)*sample_rate)
        end_idx = int((end_time+expand)*sample_rate)
        
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(audio):
            end_idx = len(audio)
        
        audio_segment = audio[start_idx:end_idx]
        audio_segments.append(audio_segment)
        times.append((start_time, end_time))
    return audio_segments, times, sample_rate
        
def play_random_audio_events(audio_file_path, annotation_df, event_class, event_category, n=5, expand=0):
    audio, sample_rate = librosa.load(audio_file_path, None)
    event_df = annotation_df[annotation_df[event_class] == event_category][:n]
    audio_objects = []
    audio_segments = []
    for index, row in event_df.iterrows():
        start_time = float(row['Starttime'])
        end_time   = float(row['Endtime'])
        
        start_idx = int((start_time-expand)*sample_rate)
        end_idx = int((end_time+expand)*sample_rate)
        
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(audio):
            end_idx = len(audio)
        
        audio_segment = audio[start_idx:end_idx]
        audio_segments.append(audio_segment)

    if n > len(audio_segments):
        n = len(audio_segments)
    
    random_choice_indices = np.random.choice(np.arange(n), n, replace=False)

    for idx in random_choice_indices:
        audio_object = ipd.Audio(audio_segments[idx], rate=sample_rate)
        ipd.display(audio_object)

def get_first_audio_events_consecutive_audio(audio_file_path, annotation_df, event_class, event_category, n=5, expand=0):
    audio, sample_rate = librosa.load(audio_file_path, None)
    event_df = annotation_df[annotation_df[event_class] == event_category][:n]
    times = []

    for index, row in event_df.iterrows():
        start_time = float(row['Starttime'])
        end_time   = float(row['Endtime'])
        times.append((start_time, end_time))

    start_time = times[0][0]
    end_time = times[-1][1]

    start_idx = int((start_time-expand)*sample_rate)
    end_idx = int((end_time+expand)*sample_rate)

    audio_segment = audio[start_idx:end_idx]
    return audio_segment, times, sample_rate
