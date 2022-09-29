import train_model
import glob
import os
import sys

# ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sample_rate = 22050
root_path_train = './data/Development_Set_{}Hz/Training_Set/'.format(sample_rate)
root_path_valid = './data/Development_Set_{}Hz/Validation_Set/'.format(sample_rate)

n_time = 16
n_layer = 32
embedding_dim = 1024

n_mels = 128
#window_size = 1024 #4096
tf_transform_name = sys.argv[1]
channels = 1

n_background = 0 #1000
normalize_energy = False
normalize_input = True

for window_size in [16384, 2048]:
    model_name = 'resnet_{}'.format(window_size)
    experiment_dir = 'experiments/final_ensemble_small/tf_{}/window_size_{}'.format(tf_transform_name, window_size)
    print("running experiment: {} ...".format(experiment_dir))

    train_conf = {
        'csv_paths' : glob.glob(os.path.join(root_path_train, '*/*.csv')),
        # model
        'model_name' : model_name,
        'embedding_dim' : embedding_dim,
        'n_layer' : n_layer,
        'channels' : channels,
        # training settings
        'epochs'        : 1000,
        'learning_rate' : 1e-3,
        'patience'      : 10,
        'batch_size'    : 64,
        'nb_runs'       : 5,
        'epoch_downstream_eval' : 5, # evaluate every nth epoch
        
        # data settings
        ## audio
        'sample_rate'  : sample_rate,
        'window_size'  : window_size,
        'hop_size'     : window_size // 2,
        'n_background' : n_background,
        'cache'        : False,
        
        ## target
        'n_classes' : 48,
        'n_time'    : n_time,
        
        ## input
        'n_mels'       : n_mels,
        'tf_transform' : tf_transform_name, 
        'normalize_energy' : normalize_energy,
        'normalize_input' : normalize_input,
    }

    eval_conf = {
        'root_path'      : root_path_valid,
        'csv_paths'      : glob.glob(os.path.join(root_path_valid, '*/*.csv')),
        'n_shot'         : 5,
        'n_classes'      : train_conf['n_classes'],
        'n_time'         : train_conf['n_time'],
        'sample_rate'    : train_conf['sample_rate'],
        'n_mels'         : train_conf['n_mels'],
        'window_size'    : train_conf['window_size'],
        'hop_size'       : train_conf['window_size'] // 2,
        'model_name'     : train_conf['model_name'],
        'embedding_dim'  : train_conf['embedding_dim'],
        'n_layer'        : train_conf['n_layer'],
        'channels'       : train_conf['channels'],
        'tf_transform'   : train_conf['tf_transform'],
        'classification_threshold' : 0.5,
        'padding' : 'expand',
        'adaptive_window_size' : False,
        'normalize_energy' : train_conf['normalize_energy'],
        'normalize_input' : train_conf['normalize_input'],
    }


    train_model.main(experiment_dir, train_conf, eval_conf)
