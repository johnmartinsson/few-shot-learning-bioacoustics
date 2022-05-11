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

window_sizes = np.array([2048, 4096, 8192])

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
    
    for window_size in window_sizes:
        experiment_dir = './experiments/medium/sample_rate_{}/pcen_biodiversity/window_size_{}/n_background_0/n_time_16/n_mels_128/'.format(sample_rate, window_size)

        for idx_run in range(3):
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
            p_embeddings = evaluate_model.create_positive_embeddings(model, n_shot, csv_path, window_size, tf_transform)
            # 3.   compute negative embeddings
            n_embeddings = evaluate_model.create_negative_embeddings(model, n_shot, csv_path, window_size, tf_transform)
            # 4.   compute query embeddings and times
            valid_dataset_all = dcase_dataset.BioacousticDatasetNew(
                csv_paths          = [csv_path],
                window_size        = window_size,
                hop_size           = hop_size,
                sample_rate        = sample_rate,
                n_classes          = n_classes,
                n_time             = n_time,
                n_shot             = 10000000000000000000000,
                n_background       = 10000000000000000000000,
                transform          = tf_transform,
                cache              = False,
                is_validation_data = True,
                use_old            = False
            )

            use_embeddings = True
            valid_loader_all = torch.utils.data.DataLoader(valid_dataset_all, batch_size=16, shuffle=False, num_workers=8)

            q_embeddings   = []
            for (x, _) in valid_loader_all:
                x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double()
                x = x.cuda()
                logits, embedding = model(x)
                if not use_embeddings:
                    q_embeddings.append(torch.sigmoid(logits).detach().cpu().numpy())
                else:
                    q_embeddings.append(embedding.detach().cpu().numpy())
            q_embeddings = np.concatenate(q_embeddings)
            
            q_embedding_times = valid_dataset_all.intervals

#             q_embeddings, q_embedding_times = evaluate_model.create_query_embeddings(model, wav_path, sample_rate, window_size, hop_size, tf_transform)

            print(q_embeddings.shape)
            print(q_embedding_times.shape)
            
            embeddings_dir = os.path.join(experiment_path, "embeddings", os.path.basename(csv_path).split('.')[0])
            
            if not os.path.exists(embeddings_dir):
                os.makedirs(embeddings_dir)
            print("saving embeddings in: ", embeddings_dir)
            np.save(os.path.join(embeddings_dir, "positive.npy"), p_embeddings)
            np.save(os.path.join(embeddings_dir, "negative.npy"), n_embeddings)
            np.save(os.path.join(embeddings_dir, "query.npy"), q_embeddings)
            np.save(os.path.join(embeddings_dir, "query_times.npy"), q_embedding_times)
