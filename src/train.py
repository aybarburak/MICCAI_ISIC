from __future__ import print_function, division
import torch
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
from tqdm import tqdm
import dataloader as dl
from models import build_model
import test
import utils
from optimizer import get_optimizer
from losses import get_loss
from sklearn.metrics import accuracy_score, roc_auc_score
from matplotlib import pyplot as plt
import os
from colorama import Fore
from tensorboard_writer import TensorboardWriter
import warnings

log = utils.get_logger()
warnings.filterwarnings("ignore")


def train_model(dataloaders_folds, model, criterion, optimizer, results_dir, params, **kwargs):
    """
    Fine tunes torchvision efficientNet model to SIIM dataset.
    :param dataloaders_folds: pytorch train, val and test dataloaders fold class
    :param model: torchvision model to be finetuned (efficient in this case)
    :param criterion: loss criterion (binary cross entropy loss, BCELoss)
    :param optimizer: optimizer to use in training (SGD)
    :param results_dir: the directory where to save the checkpoints and the prediction results
    :param params: the object containing the hyperparameters and other training settings
    :return: best epoch
    """
    losses = []
    rocs = []
    since = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val = None
    es_patience = params[
        'num_epochs_early_stop']  # Early Stopping patience - for how many epochs with no improvements to wait
    num_epochs = params['num_epochs']
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True,
                                  factor=params['decay_factor'])

    dl_its = dataloaders_folds.dataloaders
    writer = TensorboardWriter(kwargs.get("logdir", "logdir"), log,
                               enabled=kwargs.get("enabled", True))

    # iterate over epochs
    patience = es_patience
    num_iters = 0

    for epoch in range(num_epochs):

        start_time = time.time()
        correct = 0
        epoch_loss = 0
        model.train()

        log.tinfo(f'Starting epoch {epoch}/{num_epochs}')

        # iterate over all datapoints in train dataloader:
        with tqdm(total=len(dl_its['train']), desc='train', ncols=100,
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)) as pbar:
            for inx, batch in enumerate(dl_its['train']):
                x, y = batch
                if dataloaders_folds.meta_features is not None:
                    x[0] = torch.tensor(x[0], device=device, dtype=torch.float32)
                    x[1] = torch.tensor(x[1], device=device, dtype=torch.float32)
                else:
                    x = torch.tensor(x, device=device, dtype=torch.float32)

                y = torch.tensor(y, device=device, dtype=torch.float32)

                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                pred = torch.round(torch.sigmoid(z))  # round off sigmoid to obtain predictions
                correct += (pred.cpu() == y.cpu().unsqueeze(
                    1)).sum().item()  # tracking number of correctly predicted samples
                epoch_loss += loss.item()

                # log per batch training loss and accuracy to tensorboard
                num_iters += 1
                writer.set_step(num_iters, "train")
                writer.add_scalar("batch_loss", loss.item())
                writer.add_scalar("batch_acc", correct)

                # tqdm progress bar update
                pbar.update()

            train_acc = correct / len(dataloaders_folds.train_idx)

            # log per epoch training loss and accuracy to tensorboard
            writer.set_step(epoch, mode='train')
            writer.add_scalar("loss", epoch_loss / len(dataloaders_folds.train_idx))
            writer.add_scalar("acc", train_acc)

        # iterate over all datapoints in val dataloader:
        # switch model to the evaluation mode
        model.eval()
        val_preds = torch.zeros((len(dataloaders_folds.val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            # Predicting on validation set
            j = 0
            for x_val, y_val in tqdm(dl_its['val'], desc='val', ncols=100,
                                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
                if dataloaders_folds.meta_features is not None:
                    x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
                    x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
                    batch_size = x_val[0].shape[0]

                else:
                    x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                    batch_size = x_val.shape[0]
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[j * batch_size:j * batch_size + batch_size] = val_pred
                j += 1
            val_acc = accuracy_score(dataloaders_folds.train_df.iloc[dataloaders_folds.val_idx]['target'].values,
                                     torch.round(val_preds.cpu()))
            val_roc = roc_auc_score(dataloaders_folds.train_df.iloc[dataloaders_folds.val_idx]['target'].values,
                                    val_preds.cpu())

            # log epoch validation loss, accuracy and auc to tensorboard
            writer.set_step(epoch, mode='val')
            writer.add_scalar("loss", epoch_loss / len(dataloaders_folds.train_idx))
            writer.add_scalar("acc", val_acc)
            writer.add_scalar("roc_auc", val_roc)

            scheduler.step(val_roc)
            log.info(
                'Epoch {:02}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
                    epoch,
                    epoch_loss,
                    train_acc,
                    val_acc,
                    val_roc,
                    str(datetime.timedelta(seconds=time.time() - start_time))[:7]))

            losses.append(epoch_loss)
            rocs.append(val_roc)
            if not best_val:
                best_val = val_roc
                utils.save_checkpoint(model, optimizer, best_val, epoch,
                                      checkpoint_file=f'{results_dir}/checkpoint.pth')
                log.info('Saved model on Epoch {:02}:'.format(epoch))
                continue

            if val_roc >= best_val:
                best_val = val_roc
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                utils.save_checkpoint(model, optimizer, best_val, epoch,
                                      checkpoint_file=f'{results_dir}/checkpoint.pth')  # Saving current best model
                log.info('Saved model on Epoch {:02}:'.format(epoch))
            else:
                patience -= 1
                if patience == 0:
                    log.info('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                    break

    plt.subplot(121)
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.subplot(122)
    plt.plot(rocs)
    plt.xlabel("Iteration")
    plt.ylabel("Validation ROC")
    plt.savefig(os.path.join(results_dir, 'log.png'))

    elapsed = time.time() - since
    log.tinfo(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")


def experiment_run(im_dir, metadata_dir, results_dir, params, **kwargs):
    """
    Train torchvision model to SIIM data given high level hyperparameters.
    Then it evaluates it on the test data and write the submission results.
    :param im_dir: path to main path for dataset
    :param metadata_dir: path to main path for csv
    :param results_dir: the directory where to save the checkpoints and the prediction results
    :param params: containing model's hyperaparameters and other training settings
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) prepare the training validation data
    dataloaders_folds = dl.folds(im_dir, metadata_dir, ['train', 'test'], params.dataloader_settings)

    # 2) build model
    model = build_model(dataloaders_folds.meta_features, params.model_settings)
    if params.train_resume:
        utils.load_checkpoint(model, checkpoint_file=f'{results_dir}/checkpoint.pth')

    model = model.cuda()

    # 3) loss function
    criterion = get_loss(params.loss_settings, dataloaders_folds.label_freq)

    # 4) prepare the optimizer
    optimizer = get_optimizer(model, params.optimizer_settings)

    _lengths = ', '.join([f'#{_k} = {len(dataloaders_folds.dataloaders[_k].dataset)}' for _k in ['train', 'val']])
    log.info(f"Datasets sizes: {_lengths}")

    # train model
    log.info("Starting training")

    train_model(dataloaders_folds,
                model,
                criterion,
                optimizer,
                results_dir,
                params.training_settings, logdir=kwargs.get("logdir"))

    # load back the best weights (and set the dropout to 0)
    model = build_model(dataloaders_folds.meta_features, params.model_settings)
    utils.load_checkpoint(model, checkpoint_file=f'{results_dir}/checkpoint.pth')
    model.cuda()
    # switch model to the evaluation mode
    model.eval()
    # oof
    oof = np.zeros((len(dataloaders_folds.train_df), 1))  # Out Of Fold predictions
    val_preds = torch.zeros((len(dataloaders_folds.val_idx), 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        j = 0
        for x_val, y_val in tqdm(dataloaders_folds.dataloaders['val'], desc='val',
                                 ncols=10, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            if dataloaders_folds.meta_features is not None:
                x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
                x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
                batch_size = x_val[0].shape[0]

            else:
                x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                batch_size = x_val.shape[0]

            z_val = model(x_val)
            val_pred = torch.sigmoid(z_val)
            val_preds[j * batch_size:j * batch_size + batch_size] = val_pred
            j += 1
        oof[dataloaders_folds.val_idx] = val_preds.cpu().numpy()
    log.info('OOF: {:.3f}'.format(roc_auc_score(dataloaders_folds.train_df['target'], oof)))

    # get predictions and AUC scores on the test fold
    log.info("Starting testing")
    preds = test.test_model(dataloaders_folds.dataloaders['test'], model,
                            results_dir, params.testing_settings,
                            dataloaders_folds.meta_features)
    log.info("End testing")

    return preds
