import train_model
import glob
import os

sample_rate = 22050
root_path_train = '/mnt/storage_1/datasets/bioacoustics_dcase2022/Development_Set_{}Hz/Training_Set/'.format(sample_rate)
root_path_valid = '/mnt/storage_1/datasets/bioacoustics_dcase2022/Development_Set_{}Hz/Validation_Set/'.format(sample_rate)

n_background = 0
n_time = 32
window_size = 4096
n_mels = 128
tf_transform_name = 'pcen_biodiversity'

experiment_dir = 'experiments/n_background/window_size_{}/n_background_{}/n_time_{}/n_mels_{}/'.format(window_size, n_background, n_time, n_mels)
print("running experiment: {} ...".format(experiment_dir))

if window_size == 1024:
    model_name = "resnet_1024"
elif window_size == 2048:
    model_name = "resnet_2048"
else:
    model_name = "resnet"

train_conf = {
    'csv_paths' : glob.glob(os.path.join(root_path_train, '*/*.csv')),
    # model
    'model_name' : model_name,
    # training settings
    'epochs'        : 1000,
    'learning_rate' : 1e-3,
    'patience'      : 10,
    'batch_size'    : 64,
    'nb_runs'       : 1,
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
    'tf_transform'   : train_conf['tf_transform'],
    'classification_threshold' : 0.5,
    'padding' : 'expand',
    'adaptive_window_size' : False,
}


train_model.main(experiment_dir, train_conf, eval_conf)
