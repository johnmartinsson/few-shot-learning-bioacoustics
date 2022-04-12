import os
import tqdm

import numpy as np
import torch
import torch.utils.tensorboard

import models

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

def main(base_train, base_valid, experiment_dir):
    # Model settings
    n_classes = 48
    n_time    = 4

    # Training settings
    epochs = 1000
    learning_rate = 1e-3 #[1e-2, 1e-3, 3e-4, 1e-4, 1e-5]
    patience = 10
    batch_size = 64
    mode = 'train'
    nb_runs = 5

    for idx_run in range(nb_runs):
        experiment_path = os.path.join(experiment_dir, 'variation/run_{}'.format(idx_run))

        if mode == "train":

            writer = torch.utils.tensorboard.SummaryWriter(log_dir=experiment_path)

            model = models.get_model(n_classes, n_time).double()
            model = model.cuda()
            # just a copy of the model
            best_model = models.get_model(n_classes, n_time).double()
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
                train_loss = train(model, optimizer, loss_function, base_train_loader)
                print("train loss: {}".format(train_loss))
                valid_loss = evaluate(model, base_valid_loader, loss_function)
                print("valid loss: {}".format(valid_loss))
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/valid', valid_loss, epoch)

                epoch += 1

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    best_model.load_state_dict(model.state_dict())
                    torch.save(best_model.state_dict(), os.path.join(experiment_path, "best_model.ckpt"))
                    print("saving best model ...")

                # convergence criterion
                if epoch - best_epoch >= patience or epoch >= epochs:
                    not_converged = False

            torch.save(model.state_dict(), os.path.join(experiment_path, "model_epochs_{}.ckpt".format(epoch)))
            torch.save(best_model.state_dict(), os.path.join(experiment_path, "best_model.ckpt"))
