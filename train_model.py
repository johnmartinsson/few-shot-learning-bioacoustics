import os
import tqdm

import numpy as np
import torch
import torch.utils.tensorboard

import sed_utils
import dcase_dataset
import models
import evaluate_model

def train(model, optimizer, loss_function, train_loader):
    model.train()

    running_loss = 0
    count = 0
    for (x, y) in tqdm.tqdm(train_loader):
        x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double() # add channel dimension
        x = x.cuda()
        y = y.double()
        y = y.cuda()

        optimizer.zero_grad()

        y_pred, _ = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        count += 1
    return running_loss / count

def evaluate(model, loader, loss_function):
    model.eval()

    count = 0
    running_acc = 0
    running_loss = 0

    ys = []
    ys_pred = []
    for (x, y) in loader:
        x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double() # add channel dimension
        x = x.cuda()
        y = y.double()
        y = y.cuda()

        y_pred, _ = model(x)
        loss = loss_function(y_pred, y)
        running_loss += loss.item()

        count+=1

    return running_loss / count

def main(experiment_dir, train_conf, downstream_eval_conf):
    root_path = train_conf['root_path']
    # Model settings
    model_name = train_conf['model_name']

    # Training settings
    epochs        = train_conf['epochs'] #1000
    learning_rate = train_conf['learning_rate'] #1e-3 #[1e-2, 1e-3, 3e-4, 1e-4, 1e-5]
    patience      = train_conf['patience'] #10
    batch_size    = train_conf['batch_size'] #64
    nb_runs       = train_conf['nb_runs'] #10 #5
    epoch_downstream_eval = train_conf['epoch_downstream_eval']

    # Data settings
    n_classes    = train_conf['n_classes']
    n_time       = train_conf['n_time']
    n_mels       = train_conf['n_mels']
    n_bins       = None # TODO: fix this!
    tf_transform_name = train_conf['tf_transform']

    window_size  = train_conf['window_size']
    hop_size     = train_conf['hop_size']
    n_background = train_conf['n_background']
    sample_rate  = train_conf['sample_rate']

    # choose time-frequency transform
    conf_bio    = sed_utils.get_bioacoustic_pcen_conf()
    conf_speech = sed_utils.get_speech_pcen_conf()
    tf_transforms = {
            'decibel'           : lambda x: sed_utils.wav_to_mel(x - (np.sum(x)/np.size(x)), sample_rate, n_mels=n_mels),
            'pcen_biodiversity' : lambda x: sed_utils.wav_to_pcen(x - (np.sum(x)/np.size(x)), sample_rate, conf_bio, n_mels=n_mels),
            'pcen_speech'       : lambda x: sed_utils.wav_to_pcen(x - (np.sum(x)/np.size(x)), sample_rate, conf_speech, n_mels=n_mels),
    }

    tf_transform = tf_transforms[tf_transform_name]

    # load the dataset
    base_dataset = dcase_dataset.BioacousticDataset(
	root_dir           = root_path,
	window_size        = window_size,
	hop_size           = hop_size,
	sample_rate        = sample_rate,
	n_classes          = n_classes,
	n_time             = n_time,
	n_background       = n_background,
	transform          = tf_transform,
	cache              = False,
	is_validation_data = False,
        use_old            = False
    )

    # split data
    train_size = int(0.8 * len(base_dataset))
    valid_size = len(base_dataset) - train_size
    base_train, base_valid = torch.utils.data.random_split(base_dataset, [train_size, valid_size])

    for idx_run in range(nb_runs):
        experiment_path = os.path.join(experiment_dir, 'run_{}'.format(idx_run))

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=experiment_path)

        model = models.get_model(model_name, n_classes, n_time, n_mels, n_bins).double()
        model = model.cuda()
        # just a copy of the model
        best_model = models.get_model(model_name, n_classes, n_time, n_mels, n_bins).double()
        best_model = best_model.cuda()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        loss_function = torch.nn.BCEWithLogitsLoss()

        base_train_loader = torch.utils.data.DataLoader(base_train, batch_size=batch_size, shuffle=True, num_workers=8)
        base_valid_loader = torch.utils.data.DataLoader(base_valid, batch_size=batch_size, shuffle=False, num_workers=8)

        # convergence state
        best_valid_loss = np.inf
        best_epoch = 0
        epoch = 0
        not_converged = True

        while not_converged:
            # evaluate
            valid_loss = evaluate(model, base_valid_loader, loss_function)

            # save best model
            if valid_loss < best_valid_loss or epoch == 0:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_model.load_state_dict(model.state_dict())
                torch.save(best_model.state_dict(), os.path.join(experiment_path, "best_model.ckpt"))

                # save settings files
                np.save(os.path.join(experiment_path, "train_conf.npy"), train_conf)
                np.save(os.path.join(experiment_path, "valid_conf.npy"), downstream_eval_conf)
                print("saving best model ...")

            # evaluate on downstream task
            if epoch % epoch_downstream_eval == 0:
                overall_scores, scores_per_subset, post_overall_scores, post_scores_per_subset = evaluate_model.evaluate(experiment_path, downstream_eval_conf)
                print(overall_scores)
                print(scores_per_subset)

                writer.add_scalar('downstream/overall/fmeasure', overall_scores['f-measure'], epoch)
                writer.add_scalar('downstream/overall/precision', overall_scores['precision'], epoch)
                writer.add_scalar('downstream/overall/recall', overall_scores['recall'], epoch)

                writer.add_scalar('downstream/ME/fmeasure', scores_per_subset['ME']['f-measure'], epoch)
                writer.add_scalar('downstream/PB/fmeasure', scores_per_subset['PB']['f-measure'], epoch)
                writer.add_scalar('downstream/HB/fmeasure', scores_per_subset['HB']['f-measure'], epoch)

                writer.add_scalar('downstream_post/overall/fmeasure', post_overall_scores['f-measure'], epoch)

                writer.add_scalar('downstream_post/ME/fmeasure', post_scores_per_subset['ME']['f-measure'], epoch)
                writer.add_scalar('downstream_post/PB/fmeasure', post_scores_per_subset['PB']['f-measure'], epoch)
                writer.add_scalar('downstream_post/HB/fmeasure', post_scores_per_subset['HB']['f-measure'], epoch)


            # train model for one epoch
            train_loss = train(model, optimizer, loss_function, base_train_loader)

            # print results
            print("train loss: {}".format(train_loss))
            print("valid loss: {}".format(valid_loss))
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/valid', valid_loss, epoch)

            epoch += 1
            # convergence criterion
            if epoch - best_epoch >= patience or epoch >= epochs:
                not_converged = False

            # TODO: every nth epoch do a proper few-shot evaluation

        torch.save(model.state_dict(), os.path.join(experiment_path, "model_epochs_{}.ckpt".format(epoch)))
        torch.save(best_model.state_dict(), os.path.join(experiment_path, "best_model.ckpt"))
