import numpy as np
import pandas as pd

def get_event_lengths(n_shots, csv_path, class_name):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df[class_name] == 'POS'].tolist()
    ann_nshot_df = ann_df.loc[ref_pos_indexes[:n_shots]]
    
    event_lengths = (ann_nshot_df['Endtime']-ann_nshot_df['Starttime'])
    
    return event_lengths

def get_density(n_shots, csv_path, class_name):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df[class_name] == 'POS'].tolist()
    ann_nshot_df = ann_df.loc[ref_pos_indexes[:n_shots]]

    event_lengths = get_event_lengths(n_shots, csv_path, class_name)
    
    starttime = ann_nshot_df['Starttime'].iloc[0]
    endtime   = ann_nshot_df['Endtime'].iloc[-1]
    density   = event_lengths.sum() / (endtime-starttime)

    return density

def get_gap_lengths(n_shots, csv_path, class_name):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df[class_name] == 'POS'].tolist()
    ann_5shot_df = ann_df.loc[ref_pos_indexes[:n_shots]]

    gap_lengths = (ann_5shot_df['Starttime'][1:].to_numpy()-ann_5shot_df['Endtime'][:-1].to_numpy())
    
    return gap_lengths

# backwards compability
def get_nshot_event_lengths(n_shots, csv_path):
    return get_event_lengths(n_shots, csv_path, 'Q')

def get_nshot_gap_lengths(n_shots, csv_path):
    return get_gap_lengths(n_shots, csv_path, 'Q')

def get_nshot_density(n_shots, csv_path):
    return get_density(n_shots, csv_path, 'Q')
