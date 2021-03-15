import optuna
import dataloader as dl
import utils
from models import build_model
import numpy as np
import os
from zipfile import ZipFile
from tqdm import tqdm
from colorama import Fore
from shutil import copyfile
import logging
import torch
import train
from losses import get_loss
from optimizer import get_optimizer
import time
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from tensorboard_writer import TensorboardWriter
import pickle as pkl
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, average_precision_score


class Experiment(object):

    def __init__(self, args, **kwargs):
        super(Experiment, self).__init__()
        self.args = args
        self.rocs = []
        self.losses = []
        self.params = None
        self.log = None
        self.use_gpu = None
        self.dataloaders_folds = None
        self.criterion = None
        self.lengths = None
        self.optimizer = None
        self.device = torch.device('cpu')
        self.study = None
        self.op_logger = None
        self.save_study = False

        self.setup_experiment()

        self.log_dir = kwargs.get("logdir", "logdir")
        self.tensorboard_enabled = kwargs.get("enabled", True)

        run_id = datetime.datetime.now().strftime(r'%m%d_%H%M%S')

        self.writer = TensorboardWriter(self.log_dir + f'/{run_id}', self.log,
                                        enabled=self.tensorboard_enabled)
        warnings.filterwarnings("ignore")

    def check_requirements(self):
        # Check if the dataset directory: train.zip - test.zip
        assert os.path.exists(os.path.join(self.args['im_dir'], 'train.zip')), \
            f"Directory {os.path.join(self.args['im_dir'], 'train.zip')} doesn't exist."
        assert os.path.exists(os.path.join(self.args['im_dir'], 'test.zip')), \
            f"Directory {os.path.join(self.args['im_dir'], 'test.zip')} doesn't exist."
        # Check if the labels directory: train.csv - test.csv - sample_submission.csv
        assert os.path.exists(os.path.join(self.args['metadata_dir'], 'train.csv')), \
            f"Directory {os.path.join(self.args['metadata_dir'], 'train.csv')} doesn't exist."
        assert os.path.exists(os.path.join(self.args['metadata_dir'], 'test.csv')), \
            f"Directory {os.path.join(self.args['metadata_dir'], 'test.csv')} doesn't exist."

        # Check if the params directory: *.json
        assert os.path.exists(self.args['params_file']), f'File {self.args["params_file"]} does not exist.'

    def setup_experiment(self):

        # move the current directory to the project root
        abspath = os.path.abspath(__file__)
        project_root_dir = os.path.join(os.path.dirname(abspath), '..')
        os.chdir(project_root_dir)

        # Parse the command line arguments
        self.check_requirements()

        # Extract dataset
        train_path = os.path.join(self.args['im_dir'], 'train')
        if not os.path.exists(train_path):
            with ZipFile(file=train_path + '.zip') as zip_file:
                for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()),
                                 desc='Extract training data',
                                 ncols=100, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)):
                    zip_file.extract(member=file, path=self.args['im_dir'])

        test_path = os.path.join(self.args['im_dir'], 'test')
        if not os.path.exists(test_path):
            with ZipFile(file=test_path + '.zip') as zip_file:
                for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()),
                                 desc='Extract testing data',
                                 ncols=100, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)):
                    zip_file.extract(member=file, path=self.args['im_dir'])

        # Create a directory for the current experiment
        results_dir_current_job = os.path.join(self.args['results_dir'], utils.now_as_str_f())
        while os.path.isdir(results_dir_current_job):  # generate a new timestamp if the current one already exists
            results_dir_current_job = os.path.join(self.args['results_dir'], utils.now_as_str_f())
        os.makedirs(results_dir_current_job)

        self.args['results_dir'] = results_dir_current_job

        # Copy the settings file into the results directory
        copyfile(self.args['params_file'],
                 os.path.join(results_dir_current_job, os.path.basename(self.args['params_file'])))

        submission_path = os.path.join(self.args['metadata_dir'], 'sample_submission.csv')
        copyfile(submission_path, os.path.join(results_dir_current_job, os.path.basename(submission_path)))

        # Load training settings (e.g. hyperparameters)
        self.params = utils.Params(self.args['params_file'])

        # Get the logger
        log_path = os.path.join(results_dir_current_job, 'training.log')
        log_level = self.params.log_level if hasattr(self.params, 'log_level') else logging.DEBUG
        self.log = utils.get_logger(log_path, log_level)
        self.log.info(f"Results directory: {results_dir_current_job}")
        self.log.info(f"Hyperparameters: {self.params.dict}")

        # set optuna logger
        self.op_logger = logging.getLogger()
        self.op_logger.setLevel(logging.INFO)  # Setup the root logger.
        self.op_logger.addHandler(logging.FileHandler("optuna.log", mode="w"))
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.

        # Check if any GPU is available
        self.use_gpu = torch.cuda.is_available()
        self.log.info(f"GPU is {'not ' if not self.use_gpu else ''}available")
        if self.use_gpu:
            gpu_count = torch.cuda.device_count()
            self.log.info(f"Available GPU count: {gpu_count}")
            device_id = int(self.args['gpu_device_id'])
            torch.cuda.set_device(device_id)
            self.log.info(f"Choose CUDA device {device_id} ({torch.cuda.get_device_properties(device_id)})")

        # Set the random seed for reproducible experiments
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
            torch.manual_seed(self.params.seed)
            if self.use_gpu:
                torch.cuda.manual_seed(self.params.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            self.log.info(f"Set seed to {self.params.seed}")
        else:
            self.log.info(f"Running with random seed initialization")

    def setup_checkpoint(self):
        results_dir_current_job = os.path.join('./results', utils.now_as_str_f())
        while os.path.isdir(results_dir_current_job):  # generate a new timestamp if the current one already exists
            results_dir_current_job = os.path.join('./results', utils.now_as_str_f())
        os.makedirs(results_dir_current_job)

        self.args['results_dir'] = results_dir_current_job

        # Copy the settings file into the results directory
        copyfile(self.args['params_file'],
                 os.path.join(results_dir_current_job, os.path.basename(self.args['params_file'])))

        submission_path = os.path.join(self.args['metadata_dir'], 'sample_submission.csv')
        copyfile(submission_path, os.path.join(results_dir_current_job, os.path.basename(submission_path)))

        log_path = os.path.join(results_dir_current_job, 'training.log')
        log_level = self.params.log_level if hasattr(self.params, 'log_level') else logging.DEBUG
        self.log = utils.get_logger(log_path, log_level)
        self.log.info(f"Results directory: {results_dir_current_job}")

    def setup_data(self):
        self.dataloaders_folds = dl.folds(self.args['im_dir'], self.args['metadata_dir'],
                                          ['train', 'test'], self.params.dataloader_settings)

        self.lengths = ', '.join(
            [f'#{_k} = {len(self.dataloaders_folds.dataloaders[_k].dataset)}' for _k in ['train', 'val']]
        )

    def build_model(self):
        model = build_model(self.dataloaders_folds.meta_features, self.params.model_settings)
        if self.params.train_resume:
            path = self.params.checkpoint_path
            if path is None:
                path = f'{self.args["results_dir"]}/checkpoint.pth'
            utils.load_checkpoint(model, checkpoint_file=path)

        if self.use_gpu:
            model = model.cuda()

        return model

    def create_loss(self):
        self.criterion = get_loss(self.params.loss_settings, self.dataloaders_folds.label_freq)

    def create_optimizer(self, model):
        self.optimizer = get_optimizer(model, self.params.optimizer_settings)

    def train(self, model):
        self.setup_checkpoint()
        since = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")
        best_val = None
        rocs, average_precision = [], []

        # Early Stopping patience - for how many epochs with no improvements to wait
        es_patience = self.params.training_settings['num_epochs_early_stop']
        num_epochs = self.params.training_settings['num_epochs']
        scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='max', patience=1, verbose=True,
                                      factor=self.params.training_settings['decay_factor'])

        dl_its = self.dataloaders_folds.dataloaders
        augments = self.dataloaders_folds.augments

        self.log.info(f"Augmentation List: {augments}")

        # iterate over epochs
        patience = es_patience
        num_iters = 0
        for epoch in range(num_epochs):

            start_time = time.time()
            correct = 0
            epoch_loss = 0
            model.train()
            train_preds = []
            train_labels, val_labels = [], []

            self.log.tinfo(f'Starting epoch {epoch}/{num_epochs}')

            # iterate over all datapoints in train dataloader:

            with tqdm(total=len(dl_its['train']), desc='train', ncols=100,
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)) as pbar:
                for inx, batch in enumerate(dl_its['train']):
                    x, y = batch
                    if self.params.dataloader_settings[
                        'use_metafeatures'] and self.dataloaders_folds.meta_features is not None:
                        x[0] = torch.tensor(x[0], device=self.device, dtype=torch.float32)
                        x[1] = torch.tensor(x[1], device=self.device, dtype=torch.float32)
                    else:
                        x = torch.tensor(x, device=self.device, dtype=torch.float32)
                    y = torch.tensor(y, device=self.device, dtype=torch.float32)

                    self.optimizer.zero_grad()
                    z = model(x)
                    loss = self.criterion(z, y.unsqueeze(1))
                    loss.backward()
                    self.optimizer.step()
                    pred = torch.round(torch.sigmoid(z))  # round off sigmoid to obtain predictions
                    # tracking predictions and labels
                    train_preds += pred.squeeze(1).detach().cpu().numpy().tolist()
                    train_labels += y.detach().cpu().numpy().tolist()
                    # tracking number of correctly predicted samples
                    batch_correct = np.array(pred.cpu() == y.cpu().unsqueeze(1), dtype=np.int32).mean().item()
                    correct += np.array(pred.cpu() == y.cpu().unsqueeze(1), dtype=np.int32).sum().item()
                    epoch_loss += loss.item()

                    # log training loss and accuracy to tensorboard
                    num_iters += 1
                    self.writer.set_step(num_iters, "train")
                    self.writer.add_scalar("batch_loss", loss.item())
                    self.writer.add_scalar("batch_acc", batch_correct)

                    # tqdm progress bar update
                    pbar.update()

            train_acc = correct / len(self.dataloaders_folds.train_idx)
            train_loss = epoch_loss / len(self.dataloaders_folds.train_idx)

            # iterate over all datapoints in val dataloader:
            # switch model to the evaluation mode

            model.eval()
            val_preds = torch.zeros((len(self.dataloaders_folds.val_idx), 1), dtype=torch.float32, device=self.device)

            # Do not calculate gradient since we are only predicting
            with torch.no_grad():
                j = 0
                epoch_loss = 0

                for x_val, y_val in tqdm(dl_its['val'], desc='val', ncols=100,
                                         bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
                    if self.params.dataloader_settings['use_metafeatures'] \
                            and self.dataloaders_folds.meta_features is not None:
                        x_val[0] = torch.tensor(x_val[0], device=self.device, dtype=torch.float32)
                        x_val[1] = torch.tensor(x_val[1], device=self.device, dtype=torch.float32)
                        batch_size = x_val[0].shape[0]

                    else:
                        x_val = torch.tensor(x_val, device=self.device, dtype=torch.float32)
                        batch_size = x_val.shape[0]

                    y_val = y_val.type(torch.FloatTensor).to(self.device)

                    z_val = model(x_val)

                    # added loss for transparency
                    loss = self.criterion(z_val, y_val.unsqueeze(1))
                    epoch_loss += loss.item()

                    # tracking validation labels
                    val_labels += y_val.detach().cpu().numpy().tolist()

                    val_pred = torch.sigmoid(z_val)
                    val_preds[j * batch_size:j * batch_size + batch_size] = val_pred
                    j += 1

                val_acc = accuracy_score(
                    self.dataloaders_folds.train_df.iloc[self.dataloaders_folds.val_idx]['target'].values,
                    torch.round(val_preds.cpu()))

                val_roc = roc_auc_score(
                    self.dataloaders_folds.train_df.iloc[self.dataloaders_folds.val_idx]['target'].values,
                    val_preds.cpu())

                val_ap = average_precision_score(
                    self.dataloaders_folds.train_df.iloc[self.dataloaders_folds.val_idx]['target'].values,
                    val_preds.cpu())

                conf_mat = confusion_matrix(
                    self.dataloaders_folds.train_df.iloc[self.dataloaders_folds.val_idx]['target'].values,
                    torch.round(val_preds.cpu()))

                # log epoch validation loss, accuracy and auc to tensorboard

                val_loss = epoch_loss / len(self.dataloaders_folds.val_idx)

                self.writer.set_step(epoch, mode='')
                self.writer.add_scalars("loss", {
                    "val": val_loss,
                    "train": train_loss
                })
                self.writer.add_scalars("acc", {
                    "val": val_acc,
                    "train": train_acc
                })
                self.writer.add_scalars("roc_auc", {
                    "val": val_roc
                })
                self.writer.add_scalars("average_precision", {
                    "val": val_ap
                })

                scheduler.step(val_roc)
                self.log.info("Distribution of training labels:")
                self.log.info(Counter(train_labels))
                self.log.info("Distribution of training predictions:")
                self.log.info(Counter(train_preds))

                ######## extra logging for debugging #########
                # self.log.info("Distribution of validation labels:")
                # self.log.info(Counter(val_labels))
                # self.log.info("Distribution of validation predictions:")
                # self.log.info(Counter(torch.round(val_preds).squeeze(1).cpu().numpy().tolist()))
                ######## extra logging for debugging #########

                self.log.info("Confusion matrix:")
                self.log.info("# Actual 0 and Predicted 0: %d" % conf_mat[0][0])
                self.log.info("# Actual 0 and Predicted 1: %d" % conf_mat[0][1])
                self.log.info("# Actual 1 and Predicted 0: %d" % conf_mat[1][0])
                self.log.info("# Actual 1 and Predicted 1: %d" % conf_mat[1][1])

                self.log.info(
                    'Epoch {:02}: | Train Loss: {:.3f} | Val Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} \
                    | Val roc_auc: {:.3f} | Val AP: {:.3f} | Training time: {}'.format(
                        epoch,
                        train_loss,
                        val_loss,
                        train_acc,
                        val_acc,
                        val_roc,
                        val_ap,
                        str(datetime.timedelta(seconds=time.time() - start_time))[:7]))

                self.losses.append(epoch_loss)
                rocs.append(val_roc)
                average_precision.append(val_ap)

                if not best_val:
                    best_val = val_roc
                    utils.save_checkpoint(model, self.optimizer, best_val, epoch,
                                          checkpoint_file=f'{self.args["results_dir"]}/checkpoint.pth')
                    self.log.info('Saved model on Epoch {:02}:'.format(epoch))
                    continue

                if val_roc >= best_val:
                    best_val = val_roc
                    patience = es_patience  # Resetting patience since we have new best validation accuracy
                    utils.save_checkpoint(model, self.optimizer, best_val, epoch,
                                          checkpoint_file=f'{self.args["results_dir"]}/checkpoint.pth')  # Saving current best model
                    self.log.info('Saved model on Epoch {:02}:'.format(epoch))
                else:
                    patience -= 1
                    if patience == 0:
                        self.log.info('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                        break

                # saving the optuna study after every epoch
                if self.save_study:
                    with open(os.path.join(self.log_dir, "study.pkl"), "wb") as fp:
                        pkl.dump(self.study, fp)

        return np.max(average_precision)

    def objective(self, trial):
        """
        Hyperparameters are declared in the following manner
        n_estimators = trial.suggest_int('n_estimators', 2, 20)
        max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))

        Define hyperparameters and runs model. Returns the validation metric
        Returns: validation auc score

        """
        run_id = datetime.datetime.now().strftime(r'%m%d_%H%M%S')

        self.writer = TensorboardWriter(self.log_dir + f'/{run_id}', self.log,
                                        enabled=self.tensorboard_enabled)

        self.params.dataloader_settings['rand_aug']['N'] = trial.suggest_int('N', 2, 12)
        self.params.dataloader_settings['rand_aug']['M'] = trial.suggest_int('M', 0, 10)

        self.setup_data()
        model = self.build_model()
        self.create_loss()
        self.create_optimizer(model)

        N = self.params.dataloader_settings['rand_aug']['N']
        M = self.params.dataloader_settings['rand_aug']['M']

        self.log.info("Parameters: N: %s - M: %s" % (N, M))
        self.log.info("Starting training for: %s" % run_id)
        cv_average_precision = self.train(model)

        return cv_average_precision

    def create_study(self):
        self.study = optuna.create_study(direction='maximize')
        self.op_logger.info("Start optimization.")

    def run_study(self, save=True):
        self.study.optimize(self.objective, n_trials=self.params.grid_search_settings['num_trials'])

        trial = self.study.best_trial

        self.log.info('AUC ROC Score: {}'.format(trial.value))
        self.log.info("Best hyperparameters: {}".format(trial.params))

        # saving the study
        self.save_study = save

    def load_study(self, path):
        with open(path, "rb") as fp:
            self.study = pkl.load(fp)
            print("Loaded study from", path)
            print('AUC ROC Score: {}'.format(self.study.best_trial.value))
            self.log.info("Hyperparameters: {}".format(self.study.best_trial.params))
            self.op_logger.info("Resuming optimisation.. ")


if __name__ == "__main__":
    args = {
        "im_dir": "./dataset",
        "metadata_dir": "./labels",
        "results_dir": "./results",
        "params_file": "./params/Melanoma_params.json",
        "gpu_device_id": 0,
        "save_study": True,
    }

    experiment = Experiment(args=args, logdir="full-data-random-feat-agg-0001")

    if args.get("study_path", None) is not None:
        experiment.load_study(args["study_path"])
    else:
        experiment.create_study()
    experiment.run_study(save=args.get("save_study", False))
