import train_model

sample_rate = 22050
root_path_train = '/mnt/storage_1/datasets/bioacoustics_dcase2022/Development_Set_{}Hz/Training_Set/'.format(sample_rate)
root_path_valid = '/mnt/storage_1/datasets/bioacoustics_dcase2022/Development_Set_{}Hz/Validation_Set/'.format(sample_rate)

#n_mels = 40
n_background = 0
tf_transform_name = 'pcen_biodiversity'
for n_time in [8]: #,8]:
    for n_mels in [40]:
        for window_size in [4096]:
            hop_size = window_size // 4
            experiment_dir = 'experiments/medium/sample_rate_{}/{}/window_size_{}/n_background_{}/n_time_{}/n_mels_{}/hop_size_{}/'.format(sample_rate, tf_transform_name, window_size, n_background, n_time, n_mels, hop_size)
            print("running experiment: {} ...".format(experiment_dir))

            train_conf = {
                'root_path' : root_path_train,
                # model
                'model_name' : 'resnet',
                # training settings
                'epochs'        : 1000,
                'learning_rate' : 1e-3,
                'patience'      : 10,
                'batch_size'    : 64,
                'nb_runs'       : 2,
                'epoch_downstream_eval' : 10, # evaluate every nth epoch
                
                # data settings
                ## audio
                'sample_rate'  : sample_rate,
                'window_size'  : window_size,
                'hop_size'     : hop_size, #window_size//2,
                'n_background' : n_background,
                'cache'        : False,
                
                ## target
                'n_classes' : 48,
                'n_time'    : n_time,
                
                ## input
                'n_mels'       : n_mels,
                'tf_transform' : tf_transform_name, # 'decibel', 'pcen_biodiversity', 'pcen_speech'
            }

            eval_conf = {
                'root_path'      : root_path_valid,
                'n_shot'         : 5,
                'n_classes'      : train_conf['n_classes'],
                'n_time'         : train_conf['n_time'],
                'sample_rate'    : train_conf['sample_rate'],
                'n_mels'         : train_conf['n_mels'],
                'window_size'    : train_conf['window_size'],
                'hop_size'       : train_conf['window_size'] // 2,
                'model_name'     : train_conf['model_name'],
                'tf_transform'   : train_conf['tf_transform'],
                'cache'          : False,
                'use_embeddings' : True
            }


            train_model.main(experiment_dir, train_conf, eval_conf)
