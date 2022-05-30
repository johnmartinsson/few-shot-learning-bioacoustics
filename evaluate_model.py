import os
import numpy as np
import pandas as pd
import torch
import tqdm
import glob
import time

import dcase_dataset
import models
import post_processing as pp
import sed_utils
import dcase_evaluation
import stats_utils

def create_embeddings(model, data_loader, verbose=False):
    embeddings   = []
    for x in tqdm.tqdm(data_loader, disable=(not verbose)):
        #x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double()
        x = x.double()
        x = x.cuda()
        _, embedding = model(x)     
        embeddings.append(embedding.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    return embeddings

def evaluate(experiment_dir, conf):
    root_path = conf['root_path']
    csv_paths = conf['csv_paths']

    # make predictions
    pred_df = predict(experiment_dir, csv_paths, conf)
    pred_df.to_csv(os.path.join(experiment_dir, 'pred.csv'), index=False)

    # post-process predictions
    merged_pred_df = pp.merge_predictions(pred_df)
    merged_pred_df.to_csv(os.path.join(experiment_dir, 'merged_pred.csv'), index=False)

    unmachable_pred_df = pp.adaptive_remove_unmatchable_predictions(merged_pred_df, csv_paths, conf['n_shot'])
    unmachable_pred_df.to_csv(os.path.join(experiment_dir, 'post_processed_pred.csv'), index=False)

    pred_file_path = os.path.join(experiment_dir, 'pred.csv')
    overall_scores, scores_per_subset = dcase_evaluation.evaluate(
            pred_file_path = pred_file_path,
            ref_file_path  = root_path,
            team_name      = "TeamGBG",
            dataset        = 'VAL',
            savepath       = experiment_dir,
            metadata       = [],
            verbose        = True
    )

    post_pred_file_path = os.path.join(experiment_dir, 'post_processed_pred.csv')
    post_overall_scores, post_scores_per_subset = dcase_evaluation.evaluate(
            pred_file_path = post_pred_file_path,
            ref_file_path  = root_path,
            team_name      = "TeamGBG",
            dataset        = 'VAL',
            savepath       = experiment_dir,
            metadata       = [],
            verbose        = True
    )

    return overall_scores, scores_per_subset, post_overall_scores, post_scores_per_subset

def probability_query(query, n_prototype, p_prototype):
    """
    The pseudo-probability of the query point belonging to the positive class.
    """
    
    d_n = euclidean_distance(query, n_prototype)
    d_p = euclidean_distance(query, p_prototype)

    x = np.array([-d_n, -d_p])

    y_proba_p = np.exp(-d_p) / (np.exp(-d_p) + np.exp(-d_n))
    
    return y_proba_p

def predict(experiment_dir, csv_paths, conf, verbose=False):
    # data settings
    n_classes = conf['n_classes']
    n_time = conf['n_time']
    sample_rate = conf['sample_rate']
    n_mels = conf['n_mels']
    tf_transform_name = conf['tf_transform']
    n_shot = conf['n_shot']
    
    window_size = conf['window_size']
    hop_size = conf['hop_size']

    # model settings
    model_name = conf['model_name']
    embedding_dim = conf['embedding_dim']
    n_layer = conf['n_layer']
    channels = conf['channels']

    padding = conf['padding']
    normalize_input = conf['normalize_input']
    normalize_energy = conf['normalize_energy']

    tf_transform = sed_utils.get_tf_transform(tf_transform_name, n_mels, sample_rate, normalize=normalize_energy)

    window_sizes = np.array([256, 512, 1024, 2048, 4096, 8192])

    if normalize_input:
        mean = np.load(os.path.join(experiment_dir, "mean.npy"))
        std = np.load(os.path.join(experiment_dir, "std.npy"))
    else:
        mean = 0
        std = 1

    pos_events = []
    for csv_path in tqdm.tqdm(csv_paths, disable=(not verbose)):
        wav_path = csv_path.replace('.csv', '.wav')

        if conf['adaptive_window_size']:
            n_shot_event_lengths = stats_utils.get_nshot_event_lengths(n_shot, csv_path)
            average_event_length = np.median(n_shot_event_lengths)
            average_event_size = int(sample_rate * average_event_length)
            window_size = window_sizes[np.argmin(np.sqrt(np.power(window_sizes-average_event_size, 2)))]

            print("average event size: ", average_event_size)
            print("adaptive window size: ", window_size)
            model_name = "resnet" + "_" + str(window_size*2)
            print(model_name)
        else:
            window_size = conf['window_size']
            print("window size: ", window_size)

        ###############################################################################################################
        # Load the model
        ###############################################################################################################
        model = models.get_model(model_name, n_classes, n_time, embedding_dim=embedding_dim, n_layer=n_layer, channels=channels)
        model = model.double()
        model_path = os.path.join(experiment_dir, 'best_model.ckpt')
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
        model.eval()

        ###############################################################################################################
        # Compute the prototype and query embeddings
        ###############################################################################################################
        wave, sample_rate = sed_utils.load_wave(wav_path)

        pos_anns = stats_utils.get_positive_annotations(csv_path, n_shot, class_name='Q')
        gap_anns = stats_utils.get_gap_annotations(csv_path, n_shot, class_name='Q')
        query_anns = stats_utils.get_query_annotations(csv_path, n_shot, class_name='Q')

        query_dataset = dcase_dataset.PrototypeDataset(wave, query_anns, window_size, hop_size, sample_rate, tf_transform, normalize=normalize_input, mean=mean, std=std)
        neg_dataset = dcase_dataset.PrototypeDataset(wave, gap_anns, window_size, window_size//16, sample_rate, tf_transform, padding=padding, normalize=normalize_input, mean=mean, std=std)
        pos_dataset = dcase_dataset.PrototypeDataset(wave, pos_anns, window_size, window_size//16, sample_rate, tf_transform, padding=padding, normalize=normalize_input, mean=mean, std=std)
        
        query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=8)
        neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=64, shuffle=False, num_workers=8)
        pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=64, shuffle=False, num_workers=8)

        q_embeddings = create_embeddings(model, query_loader)
        q_embedding_times = np.array(query_dataset.times)
        
        p_embeddings = create_embeddings(model, pos_loader)
        n_embeddings = create_embeddings(model, neg_loader)

        n_prototype = np.mean(n_embeddings, axis=0)
        p_prototype = np.mean(p_embeddings, axis=0)

        ###############################################################################################################
        # Classify query embeddings
        ###############################################################################################################

        y_probas = []
        for query in q_embeddings:
            y_proba = probability_query(query, n_prototype, p_prototype)
            y_probas.append(y_proba)

        sorted_predicitions, sorted_intervals = zip(*sorted(list(zip(y_probas, q_embedding_times)), key=lambda x: x[1][0]))

        # Get the 5th annotated positive event, and set the end of that
        # event as the skiptime. (Remove all predictions before.)
        ann_df = pd.read_csv(csv_path)
        ann_df = ann_df.sort_values(by='Starttime', axis=0, ascending=True)
        nth_event = select_nth_event_with_value(ann_df, 5, value='POS')
        skiptime = nth_event['Endtime']

        for y_proba, interval in zip(sorted_predicitions, sorted_intervals):
            if y_proba > conf['classification_threshold']:
                if not interval[0] < skiptime:
                    pos_events.append({
                        'Audiofilename' : os.path.basename(csv_path).replace('.csv', '.wav'),
                        'Starttime'     : interval[0],
                        'Endtime'       : interval[1],
                    })

    pred_df = pd.DataFrame(pos_events)
    return pred_df

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.power(x1-x2, 2)))

def softmax(x, temp):
    return np.exp(x/temp)/np.sum(np.exp(x/temp))


# TODO: should these be defined here?
def remove_less_than(pred_df, time):
    events_to_drop = pred_df.index[pred_df['Endtime'] <= time].tolist()
    return pred_df.drop(events_to_drop)

def select_events_with_value(df, value='POS'):
    return df.index[df['Q'] == value].tolist()

def select_nth_event_with_value(df, n=5, value='POS'):
    df_sorted = df.sort_values('Starttime')
    df_pos_indexes = select_events_with_value(df_sorted, value)
    nth_pos_index = df_pos_indexes[n-1]
    
    nth_event = df.iloc[nth_pos_index]
    return nth_event
