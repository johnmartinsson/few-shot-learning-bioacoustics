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

def create_embeddings(model, data_loader):
    embeddings   = []
    for x in tqdm.tqdm(data_loader):
        x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double()
        x = x.cuda()
        _, embedding = model(x)     
        embeddings.append(embedding.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    return embeddings

def create_prototype_embeddings(model, wave, sample_rate, annotations, window_size, tf_transform):
    hop_size = window_size // 16

    embeddings = []

    for (start_time, end_time) in annotations:
        start_idx = int(np.ceil(sample_rate * start_time))
        end_idx   = int(np.floor(sample_rate * end_time))
        ann_window_size = end_idx - start_idx

        # TODO: consider how to expand this in the best way
        if window_size - ann_window_size > 0:
            to_pad = window_size - ann_window_size
        else:
            to_pad = 0

        annotation_segment = wave[start_idx:end_idx]
        annotation_segment = np.pad(annotation_segment, int(np.ceil(to_pad / 2)))

        annotation_segments, _ = sed_utils.split_into_segments(annotation_segment, sample_rate, hop_size, window_size)

        for wave_segment in tqdm.tqdm(annotation_segments):
            x_array = tf_transform(wave_segment)
            x_tensor = torch.from_numpy(x_array)
            x_tensor = x_tensor.view((1, 1, x_tensor.shape[0], x_tensor.shape[1])).double()
            x_tensor = x_tensor.cuda()
            _, embedding = model(x_tensor)
            embeddings.append(embedding.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
        
    return embeddings 

def create_prototype(model, wave, sample_rate, annotations, window_size, tf_transform):
    embeddings = create_prototype_embeddings(model, wave, sample_rate, annotations, window_size, tf_transform)
    return np.mean(embeddings, axis=0)

def create_positive_prototype(model, n_shot, csv_path, window_size, tf_transform):
    p_embeddings = create_positive_embeddings(model, n_shot, csv_path, window_size, tf_transform)
    return np.mean(p_embeddings, axis=0)

def create_positive_embeddings(model, n_shot, csv_path, window_size, tf_transform):
    print("creating positive embeddings ...")
    wav_path    = csv_path.replace('.csv', '.wav')
    wave, sample_rate = sed_utils.load_wave(wav_path)

    ann_df      = pd.read_csv(csv_path)
    ann_df.sort_values(by='Starttime', axis=0, ascending=True)

    ref_pos_indexes = ann_df.index[ann_df["Q"] == 'POS'].tolist()
    ann_5shot_df = ann_df.loc[ref_pos_indexes[:n_shot]]

    starttimes = ann_5shot_df['Starttime'].tolist()
    endtimes = ann_5shot_df['Endtime'].tolist()

    annotations = list(zip(starttimes, endtimes))

    embeddings = create_prototype_embeddings(model, wave, sample_rate, annotations, window_size, tf_transform)

    return embeddings

def create_negative_prototype(model, n_shot, csv_path, window_size, tf_transform):
    n_embeddings = create_negative_embeddings(model, n_shot, csv_path, window_size, tf_transform)
    return np.mean(n_embeddings, axis=0)

def create_negative_embeddings(model, n_shot, csv_path, window_size, tf_transform):
    print("creating negative embeddings ...")
    wav_path    = csv_path.replace('.csv', '.wav')
    wave, sample_rate = sed_utils.load_wave(wav_path)

    ann_df      = pd.read_csv(csv_path)
    ann_df.sort_values(by='Starttime', axis=0, ascending=True)

    ref_pos_indexes = ann_df.index[ann_df["Q"] == 'POS'].tolist()
    ann_5shot_df = ann_df.loc[ref_pos_indexes[:n_shot]]

    endtimes   = ann_5shot_df['Starttime'][1:].tolist()
    starttimes = ann_5shot_df['Endtime'][:4].tolist()
 
    annotations = list(zip(starttimes, endtimes))

    embeddings = create_prototype_embeddings(model, wave, sample_rate, annotations, window_size, tf_transform)

    return embeddings

def create_query_embeddings(model, wav_path, sample_rate, window_size, hop_size, tf_transform, batch_size=64):
    print("creating query embeddings ...")
    wave, sample_rate = sed_utils.load_wave(wav_path)

    N = len(wave)
    M = int(np.floor((N-window_size)/hop_size))
    segment_indices = [(hop_size*i, hop_size*i + window_size) for i in range(M+1)]
    embedding_times = [(start_idx / sample_rate, end_idx / sample_rate) for (start_idx, end_idx) in segment_indices]

    # load all the tensors
    #start_idx, end_idx = segment_indices[0]
    #wave_segment = wave[start_idx:end_idx]
    #x_array = tf_transform(wave_segment)
    #(n_mels, n_bins) = x_array.shape

    #x_arrays = np.zeros((len(wave_segments), n_mels, n_bins))
    #print("x_arrays: ", x_arrays.shape)

    embeddings   = []
    for idx, (start_idx, end_idx) in enumerate(tqdm.tqdm(segment_indices)):
        wave_segment = wave[start_idx:end_idx]
        x_array = tf_transform(wave_segment)
        #x_arrays[idx,:,:] = x_array

        #for x in tqdm.tqdm(x_arrays):
        x = torch.from_numpy(x_array)
        x = x.view((1, 1, x.shape[0], x.shape[1])).double()
        x = x.cuda()
        logits, embedding = model(x)
        embeddings.append(embedding.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings, embedding_times #wave_segment_times

def evaluate(experiment_dir, conf):
    root_path = conf['root_path']
    csv_paths = conf['csv_paths']
    #csv_paths = glob.glob(os.path.join(root_path, '*/*.csv'))

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
            verbose        = False
    )
    #fmeasure  = overall_scores['f-measure']
    #precision = overall_scores['precision']
    #recall    = overall_scores['recall']

    post_pred_file_path = os.path.join(experiment_dir, 'post_processed_pred.csv')
    post_overall_scores, post_scores_per_subset = dcase_evaluation.evaluate(
            pred_file_path = post_pred_file_path,
            ref_file_path  = root_path,
            team_name      = "TeamGBG",
            dataset        = 'VAL',
            savepath       = experiment_dir,
            metadata       = [],
            verbose        = False
    )
    #post_fmeasure  = overall_scores['f-measure']
    #post_precision = overall_scores['precision']
    #post_recall    = overall_scores['recall']


    return overall_scores, scores_per_subset, post_overall_scores, post_scores_per_subset

def predict(experiment_dir, csv_paths, conf):
    # TODO: go through this and compare with notebook, something seems off....

    # data settings
    n_classes = conf['n_classes'] #48
    n_time = conf['n_time'] # 8
    sample_rate = conf['sample_rate'] #22050
    n_mels = conf['n_mels'] #40
    n_bins = None # TODO: handle this?
    tf_transform_name = conf['tf_transform']
    n_shot = conf['n_shot']
    
    window_size = conf['window_size']
    hop_size = conf['hop_size']

    # model settings
    model_name = conf['model_name']
    use_embeddings = conf['use_embeddings']

    conf_bio    = sed_utils.get_bioacoustic_pcen_conf()
    conf_speech = sed_utils.get_speech_pcen_conf()
    tf_transforms = {
            'decibel'           : lambda x: sed_utils.wav_to_mel(x - (np.sum(x)/np.size(x)), sample_rate, n_mels=n_mels),
            'pcen_biodiversity' : lambda x: sed_utils.wav_to_pcen(x - (np.sum(x)/np.size(x)), sample_rate, conf_bio, n_mels=n_mels),
            'pcen_speech'       : lambda x: sed_utils.wav_to_pcen(x - (np.sum(x)/np.size(x)), sample_rate, conf_speech, n_mels=n_mels),
    }

    tf_transform = tf_transforms[tf_transform_name]

    pos_events = []
    for csv_path in tqdm.tqdm(csv_paths):
        wav_path = csv_path.replace('.csv', '.wav')

        ###############################################################################################################
        # Compute the prototypes
        ###############################################################################################################
        
        print("n-shot: ", n_shot)
        print("window_size: ", window_size)
        print("hop_size: ", hop_size)
        valid_dataset_5_shot = dcase_dataset.BioacousticDatasetNew(
            csv_paths          = [csv_path],
            window_size        = window_size,
            hop_size           = hop_size,
            sample_rate        = sample_rate,
            n_classes          = n_classes,
            n_time             = n_time,
            n_shot             = n_shot,
            n_background       = 100000000000000000,
            transform          = tf_transform,
            cache              = False,
            is_validation_data = True,
            use_old            = False,
        )

        print("5-shot: ", len(valid_dataset_5_shot.x_sig))

        # load best model
        model = models.get_model(model_name, n_classes, n_time, n_mels, n_bins)
        model = model.double()
        model_path = os.path.join(experiment_dir, 'best_model.ckpt')
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
        model.eval()

        valid_loader_5_shot = torch.utils.data.DataLoader(valid_dataset_5_shot, batch_size=64, shuffle=False, num_workers=8)
        n_prototype, p_prototype = create_prototypes(valid_loader_5_shot, model, use_embeddings=use_embeddings)

        ###############################################################################################################
        # Classify the whole validation file
        ###############################################################################################################
        valid_dataset_all = dcase_dataset.BioacousticDatasetNew(
            csv_paths          = [csv_path],
            window_size        = window_size,
            hop_size           = hop_size,
            sample_rate        = sample_rate,
            n_classes          = n_classes,
            n_time             = n_time,
            n_shot             = 10000000000000000000,
            n_background       = 10000000000000000000,
            transform          = tf_transform,
            cache              = False,
            is_validation_data = True,
            use_old            = False,
        )
        
        valid_loader_all = torch.utils.data.DataLoader(valid_dataset_all, batch_size=64, shuffle=False, num_workers=8)

        embeddings   = []
        for (x, _) in valid_loader_all:
            x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double()
            x = x.cuda()
            logits, embedding = model(x)     
            if not use_embeddings:
                embeddings.append(torch.sigmoid(logits).detach().cpu().numpy())
            else:
                embeddings.append(embedding.detach().cpu().numpy())

        embeddings = np.concatenate(embeddings)

        y_preds = []
        for query in embeddings:
            y_pred = classify(query, n_prototype, p_prototype, n_time)
            y_preds.append(y_pred)
        y_preds = np.concatenate(y_preds)

        sorted_predicitions, sorted_intervals = zip(*sorted(list(zip(y_preds, valid_dataset_all.intervals)), key=lambda x: x[1][0]))

        # Get the 5th annotated positive event, and set the end of that
        # event as the skiptime. (Remove all predictions before.)
        ann_df = pd.read_csv(csv_path)
        ann_df = ann_df.sort_values(by='Starttime', axis=0, ascending=True)
        nth_event = select_nth_event_with_value(ann_df, 5, value='POS')
        skiptime = nth_event['Endtime']

        for y_pred, interval in zip(sorted_predicitions, sorted_intervals):
            class_name = classify_category(y_pred[0], thr=conf['classification_threshold'])
            if class_name == 'POS':
                if not interval[0] < skiptime:
                    pos_events.append({
                        'Audiofilename' : os.path.basename(csv_path).replace('.csv', '.wav'),
                        'Starttime'     : interval[0],
                        'Endtime'       : interval[1],
                    })

    pred_df = pd.DataFrame(pos_events)
    return pred_df

def is_negative(target, n_classes):
    return (np.sum(target[:,0:n_classes,:], axis=(1, 2)) == 0)

def is_positive(target, n_classes):
    return np.sum(target[:,0:n_classes,:], axis=(1, 2)) > 0

def create_prototypes(valid_loader, model, use_embeddings=True):
    y_preds      = []
    embeddings   = []
    neg_bool_idx = []
    
    for (x, y) in valid_loader:
        n_classes = y.shape[1]
        x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double()
        x = x.cuda()
        logits, embedding = model(x)
        y_pred = torch.sigmoid(logits)
        
        y_preds.append(y_pred.detach().cpu().numpy())        
        neg_bool_idx.append(is_negative(y.detach().cpu().numpy(), n_classes))
        embeddings.append(embedding.detach().cpu().numpy())
        
    embeddings = np.concatenate(embeddings)
    #print("embeddings: ", embeddings.shape)
    
    y_preds = np.concatenate(y_preds)
    #print("y_preds: ", y_preds.shape)
    
    neg_bool_idx = np.concatenate(neg_bool_idx)
    #print("negs: ", np.sum(neg_bool_idx))
    
    if use_embeddings:
        n_prototypes = embeddings[neg_bool_idx]
        p_prototypes = embeddings[~neg_bool_idx]
    else:
        n_prototypes = y_preds[neg_bool_idx]
        p_prototypes = y_preds[~neg_bool_idx]
    
    n_prototype = np.mean(n_prototypes, axis=0)
    p_prototype = np.mean(p_prototypes, axis=0)
    
    return n_prototype, p_prototype

#def create_positive_prototype(dataset):
#    targets = dataset.y
#    sg_bool_idx = np.sum(targets[:,0:n_classes,:], axis=(1, 2)) > 0
#    return p_prototype

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.power(x1-x2, 2)))

def softmax(x, temp):
    return np.exp(x/temp)/np.sum(np.exp(x/temp))

# TODO: chose the threshold based on k-fold cross-validation on the few-shot data.
# E.g., define a prototype based on 4-shots, and fine-tune threshold on 5th shot
def classify_category(pred, thr=0.5):
    if pred[0] > thr:
        return 'POS'
    else:
        return 'NEG'

def classify(query, n_prototype, p_prototype, n_time):
    
    d_n = euclidean_distance(query, n_prototype)
    d_p = euclidean_distance(query, p_prototype)
    x = np.array([d_n, d_p]) # TODO: 1/d_n ?
    
    y = softmax(x, 1)
    y = np.expand_dims(y, axis=0)
    y = np.expand_dims(y, axis=2)
    y = np.repeat(y, n_time, axis=2)
    return y

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
