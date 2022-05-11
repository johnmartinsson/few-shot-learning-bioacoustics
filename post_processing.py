import os
import numpy as np
import pandas as pd

import stats_utils

def adaptive_network_predictions(prediction_paths, segment_lengths, csv_paths, n_shot):
    
    prediction_dfs = []
    for csv_path in csv_paths:
        key = os.path.basename(csv_path).replace('.csv', '.wav')
        
        event_lengths = stats_utils.get_nshot_event_lengths(n_shot, csv_path)
        mean_event_length = event_lengths.mean()
        
        gap_lengths = stats_utils.get_nshot_gap_lengths(n_shot, csv_path)
        mean_gap_length = gap_lengths.mean()
        
        d = mean_event_length
        
        idx = np.argmin(np.sqrt((segment_lengths - d)**2))
        
        prediction_path = prediction_paths[idx]   
        prediction_group = pd.read_csv(prediction_path).groupby('Audiofilename')
        prediction_df = prediction_group.get_group(key)
        prediction_dfs.append(prediction_df)
        
    prediction_df = pd.concat(prediction_dfs, ignore_index=True)
    return prediction_df

def merge_predictions(prediction_df):
    prediction_groups = prediction_df.groupby('Audiofilename')
    
    dfs = []
    for key in prediction_groups.groups.keys():
        prediction = prediction_groups.get_group(key)
        starttimes = prediction['Starttime'].to_numpy()
        endtimes   = prediction['Endtime'].to_numpy()

        # check if next prediction overlaps with current
        ds = starttimes[1:] < endtimes[:-1]
        ds = np.insert(ds, 0, [True])
        
        # create merge group
        gs = []
        for d in ds:
            if len(gs) == 0:
                g = 0
                gs.append(g)
            else:
                if d:
                    gs.append(g)
                else:
                    g+=1
                    gs.append(g)
        
        
        prediction['Group'] = gs
        
        
        #print(prediction)
        min_df = prediction.groupby('Group').min()
        max_df = prediction.groupby('Group').max()
        
        min_df.loc[:, ('Endtime')] = max_df['Endtime']
        dfs.append(min_df)

    merged_prediction_df = pd.concat(dfs, ignore_index=True)
    return merged_prediction_df

def adaptive_remove_unmatchable_predictions(prediction_df, csv_paths, n_shot):
    prediction_groups = prediction_df.groupby('Audiofilename')

    dfs = []
    for csv_path in csv_paths:
        key = os.path.basename(csv_path).replace('.csv', '.wav')
        if key in prediction_groups.groups.keys():
            prediction = prediction_groups.get_group(key)
            
            event_lengths = stats_utils.get_nshot_event_lengths(n_shot, csv_path)
            mean_event_length = event_lengths.mean()
            
            predicted_event_lengths = prediction['Endtime'] - prediction['Starttime']
            
            # These can not be matched
            not_too_short = predicted_event_lengths > 0.3 * mean_event_length
            not_too_long  = predicted_event_lengths < (1/0.3)*mean_event_length
            df = prediction[not_too_short & not_too_long]

            #print(df_test)
            #print(df)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
