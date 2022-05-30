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
        #x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double() # add channel dimension
        x = x.double()
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
    for (x, y) in tqdm.tqdm(loader):
        #x = x.view((x.shape[0], 1, x.shape[1], x.shape[2])).double() # add channel dimension
        x = x.double()
        x = x.cuda()
        y = y.double()
        y = y.cuda()

        y_pred, _ = model(x)
        loss = loss_function(y_pred, y)
        running_loss += loss.item()

        count+=1

    return running_loss / count

def main(experiment_dir, train_conf, downstream_eval_conf):
    csv_paths = train_conf['csv_paths']
    # Model settings
    model_name = train_conf['model_name']
    embedding_dim = train_conf['embedding_dim']
    n_layer = train_conf['n_layer']
    channels = train_conf['channels']

    # Training settings
    epochs        = train_conf['epochs']
    learning_rate = train_conf['learning_rate']
    patience      = train_conf['patience']
    batch_size    = train_conf['batch_size']
    nb_runs       = train_conf['nb_runs']
    epoch_downstream_eval = train_conf['epoch_downstream_eval']

    # Data settings
    n_classes    = train_conf['n_classes']
    n_time       = train_conf['n_time']
    n_mels       = train_conf['n_mels']
    tf_transform_name = train_conf['tf_transform']

    window_size  = train_conf['window_size']
    hop_size     = train_conf['hop_size']
    n_background = train_conf['n_background']
    sample_rate  = train_conf['sample_rate']
    normalize_input = train_conf['normalize_input']

    # choose time-frequency transform
    normalize_energy = train_conf['normalize_energy']
    tf_transform = sed_utils.get_tf_transform(tf_transform_name, sample_rate=sample_rate, n_mels=n_mels, normalize=normalize_energy)

    # load the base dataset
    base_dataset = dcase_dataset.BioacousticDataset(
	csv_paths          = csv_paths,
	window_size        = window_size,
	hop_size           = hop_size,
	sample_rate        = sample_rate,
	n_classes          = n_classes,
	n_time             = n_time,
	n_background       = n_background,
	transform          = tf_transform,
        normalize          = normalize_input,
    )


    # split data
    train_size = int(0.8 * len(base_dataset))
    valid_size = len(base_dataset) - train_size
    base_train, base_valid = torch.utils.data.random_split(base_dataset, [train_size, valid_size])

    print("start training ...")
    for idx_run in range(nb_runs):
        experiment_path = os.path.join(experiment_dir, 'run_{}'.format(idx_run))

        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        if normalize_input:
            np.save(os.path.join(experiment_path, "mean.npy"), base_dataset.mean)
            np.save(os.path.join(experiment_path, "std.npy"), base_dataset.std)

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=experiment_path)

        print("moving model to gpu ...")
        model = models.get_model(model_name, n_classes, n_time, embedding_dim=embedding_dim, n_layer=n_layer, channels=channels).double()
        model = model.cuda()
        # just a copy of the model
        #best_model = models.get_model(model_name, n_classes, n_time).double()
        #best_model = best_model.cuda()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        loss_function = torch.nn.BCEWithLogitsLoss()

        base_train_loader = torch.utils.data.DataLoader(base_train, batch_size=batch_size, shuffle=True, num_workers=8)
        base_valid_loader = torch.utils.data.DataLoader(base_valid, batch_size=batch_size, shuffle=False, num_workers=8)

        # convergence state
        best_valid_loss = np.inf
        best_epoch = 0
        epoch = 0
        not_converged = True
        best_fmeasure = 0

        while not_converged:
            # evaluate
            print("evaluate model ...")
            valid_loss = evaluate(model, base_valid_loader, loss_function)

            # save best model
            if valid_loss < best_valid_loss or epoch == 0:
                print("saving best model ...")
                best_valid_loss = valid_loss
                best_epoch = epoch
                #best_model.load_state_dict(model.state_dict())
                torch.save(model.state_dict(), os.path.join(experiment_path, "best_model.ckpt"))

                # save settings files
                np.save(os.path.join(experiment_path, "train_conf.npy"), train_conf)
                np.save(os.path.join(experiment_path, "valid_conf.npy"), downstream_eval_conf)

            # evaluate on downstream task
            if epoch % epoch_downstream_eval == 0:
                print("evaluating best model ...")
                overall_scores, scores_per_subset, post_overall_scores, post_scores_per_subset = evaluate_model.evaluate(experiment_path, downstream_eval_conf)
                #print(overall_scores)
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

                if post_overall_scores['f-measure'] >= best_fmeasure:
                    print("saving best downstream model ...")
                    best_fmeasure = post_overall_scores['f-measure']
                    torch.save(model.state_dict(), os.path.join(experiment_path, "best_downstream_model.ckpt"))



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
        #torch.save(best_model.state_dict(), os.path.join(experiment_path, "best_model.ckpt"))
