import numpy as np
import pandas as pd

def get_nshot_event_lengths(n_shots, csv_path):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df["Q"] == 'POS'].tolist()
    ann_5shot_df = ann_df.loc[ref_pos_indexes[:n_shots]]
    
    event_lengths = (ann_5shot_df['Endtime']-ann_5shot_df['Starttime'])
    
    return event_lengths

def get_nshot_gap_lengths(n_shots, csv_path):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df["Q"] == 'POS'].tolist()
    ann_5shot_df = ann_df.loc[ref_pos_indexes[:n_shots]]

    gap_lengths = (ann_5shot_df['Starttime'][1:].to_numpy()-ann_5shot_df['Endtime'][:4].to_numpy())
    
    return gap_lengths

def get_nshot_density(n_shots, csv_path):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df["Q"] == 'POS'].tolist()
    ann_5shot_df = ann_df.loc[ref_pos_indexes[:n_shots]]

    event_lengths = get_nshot_event_lengths(n_shots, csv_path)
    
    starttime = ann_5shot_df['Starttime'].iloc[0]
    endtime   = ann_5shot_df['Endtime'].iloc[n_shots-1]
    density   = event_lengths.sum() / (endtime-starttime)

    return density
