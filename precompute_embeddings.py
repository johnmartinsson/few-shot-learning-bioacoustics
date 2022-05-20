import os
import sys
#module_path = os.path.abspath(os.path.join('..'))
#if module_path not in sys.path:
#    sys.path.append(module_path)

import time
import tqdm
import glob
import torch
import librosa
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dcase_dataset
import sed_utils
import models
import explore
import post_processing as pp
import evaluate_model
import dcase_evaluation
import stats_utils

n_shot = 5
csv_paths = glob.glob('/mnt/storage_1/datasets/bioacoustics_dcase2022/Development_Set_22050Hz/Validation_Set/*/*.csv')

window_sizes = np.array([1024, 1024*4, 1024*8])

for csv_path in csv_paths:
    print(csv_path)
    sample_rate = 22050

#     # 1.1. choose network model based on average average event time
#     # TODO:
#     n_shot_event_lengths = stats_utils.get_nshot_event_lengths(n_shot, csv_path)
#     average_event_length = np.mean(n_shot_event_lengths)
#     average_event_size = int(sample_rate * average_event_length)
#     window_size = window_sizes[np.argmin(np.sqrt(np.power(window_sizes-average_event_size, 2)))]
#     print("average_event_size: ", average_event_size)
#     print("chose window_size: ", window_size)
#     # 1.2. choose input segment size based on average event time
#     input_segment_size = window_size
    
    wave, sample_rate = sed_utils.load_wave(csv_path.replace('.csv', '.wav'))

    for window_size in window_sizes:
        experiment_dir = 'experiments/resnet_downpool/window_size_{}/n_background_0/n_time_16/n_mels_80/'.format(window_size)
        for idx_run in range(5, 10):
            experiment_path = os.path.join(experiment_dir, 'run_{}'.format(idx_run))
            train_conf = np.load(os.path.join(experiment_path, 'train_conf.npy'), allow_pickle=True).item()
            valid_conf = np.load(os.path.join(experiment_path, 'valid_conf.npy'), allow_pickle=True).item()

            sample_rate = valid_conf['sample_rate']
            hop_size    = valid_conf['hop_size']

            n_mels      = train_conf['n_mels']
            n_classes   = train_conf['n_classes']
            n_time      = train_conf['n_time']
            print("window_size: ", window_size)
            print("hop_size: ", hop_size)

            # get the tf-transform
            bioacoustic_conf = sed_utils.get_bioacoustic_pcen_conf()
            speech_conf      = sed_utils.get_speech_pcen_conf()
            tf_transforms = {
                'decibel'           : lambda x: sed_utils.wav_to_mel(x - (np.sum(x)/np.size(x)), sample_rate, n_mels=n_mels),
                'pcen_biodiversity' : lambda x: sed_utils.wav_to_pcen(x - (np.sum(x)/np.size(x)), sample_rate, bioacoustic_conf, n_mels=n_mels),
                'pcen_speech'       : lambda x: sed_utils.wav_to_pcen(x - (np.sum(x)/np.size(x)), sample_rate, speech_conf, n_mels=n_mels),
            }
            tf_transform = tf_transforms[train_conf['tf_transform']]

            # load the model
            model = models.get_model(train_conf['model_name'], n_classes, n_time, n_mels, None)
            model = model.double()
            model_path = os.path.join(experiment_path, 'best_model.ckpt')
            #print("model_path: ", model_path)
            model.load_state_dict(torch.load(model_path))
            model = model.cuda()
            model.eval()

            # 2.   compute positive embeddings
            # 3.   compute negative embeddings
            # 4.   compute query embeddings and times

            pos_anns = stats_utils.get_positive_annotations(csv_path, n_shot, class_name='Q')
            gap_anns = stats_utils.get_gap_annotations(csv_path, n_shot, class_name='Q')
            query_anns = stats_utils.get_query_annotations(csv_path, n_shot, class_name='Q')

            query_dataset = dcase_dataset.PrototypeDataset(wave, query_anns, window_size, hop_size, sample_rate, tf_transform)
            neg_dataset = dcase_dataset.PrototypeDataset(wave, gap_anns, window_size, window_size//16, sample_rate, tf_transform)
            pos_dataset = dcase_dataset.PrototypeDataset(wave, pos_anns, window_size, window_size//16, sample_rate, tf_transform)
            
            query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=8)
            neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=64, shuffle=False, num_workers=8)
            pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=64, shuffle=False, num_workers=8)

            q_embeddings = evaluate_model.create_embeddings(model, query_loader)
            q_embedding_times = np.array(query_dataset.times)
            
            p_embeddings = evaluate_model.create_embeddings(model, pos_loader)
            n_embeddings = evaluate_model.create_embeddings(model, neg_loader)
            
            embeddings_dir = os.path.join(experiment_path, "embeddings", os.path.basename(csv_path).split('.')[0])
            
            if not os.path.exists(embeddings_dir):
                os.makedirs(embeddings_dir)
            print("saving embeddings in: ", embeddings_dir)
            np.save(os.path.join(embeddings_dir, "positive.npy"), p_embeddings)
            np.save(os.path.join(embeddings_dir, "negative.npy"), n_embeddings)
            np.save(os.path.join(embeddings_dir, "query.npy"), q_embeddings)
            np.save(os.path.join(embeddings_dir, "query_times.npy"), q_embedding_times)
