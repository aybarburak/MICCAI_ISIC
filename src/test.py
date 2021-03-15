import argparse
import glob
import logging
import os
import numpy as np
from efficientnet_pytorch import EfficientNet
from models import build_model
import pandas as pd
import sklearn.metrics as sklm
from tqdm import tqdm
import torch
from torch.autograd import Variable
from dataloader import MelanomaDataset
import seaborn as sns
import utils
import dataloader as dl
from zipfile import ZipFile
from colorama import Fore
import warnings

warnings.filterwarnings("ignore")
log = utils.get_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_dir', default='./dataset', help="Directory containing the SIIM-ISIC Melanoma images")
    parser.add_argument('--metadata_dir', default='./labels', help="Directory containing metadata files")
    parser.add_argument('--results_dir', default='./results/2020_06_21---15_44_035248',
                        help="Root directory where to save the model checkpoints and results")
    parser.add_argument('--params_file', default='Melanoma_params.json',
                        help="Path to the file  containing the training settings")
    parser.add_argument('--gpu_device_id', default=0, type=int,
                        help='GPU device number to run on (e.g. when multiple GPUs are available')
    return parser.parse_args()


def test_model(dataloader, model, result_dir, params, meta_features = None):
    """
    Gives predictions for test fold and save csv submission file

    :param dataloader: the DataLoader needed for the testing fold.
    :param model: the trained model
    :param results_dir: the directory where to save the prediction results
    :param params: containing testing hyperaparameters
    """
    # Predicting on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = params.get('batch_size', 32)
    TTA = params['TTA']  # Test Time Augmentation rounds
    preds = torch.zeros((len(dataloader.dataset), 1), dtype=torch.float32, device=device)  # Predictions for test test
    with torch.no_grad():
        for _ in range(TTA):
            i=0
            for x_test in tqdm(dataloader, desc='test', ncols=100,
                               bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
                if meta_features is not None:
                    x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                    x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                    batch_size = x_test[0].shape[0]
                else:
                    x_test = x_test.to(device)

                z_test = model(x_test)
                z_test = torch.sigmoid(z_test)
                preds[i * batch_size:i * batch_size + batch_size] += z_test
                i += 1
        preds /= TTA
    sns_plot = sns.kdeplot(pd.Series(preds.cpu().numpy().reshape(-1, )))
    sns_plot.get_figure().savefig(os.path.join(result_dir,'preds.png'))
    sub = pd.read_csv(os.path.join(result_dir,'sample_submission.csv'))
    sub['target'] = preds.cpu().numpy().reshape(-1,)
    sub.to_csv(os.path.join(result_dir,'submission.csv'), index=False)

def check_requirements(args):
    # Check if the dataset directory: test
    assert os.path.exists(os.path.join(args.im_dir,'test.zip')), \
        f"Directory {os.path.join(args.im_dir,'test.zip')} doesn't exist."
    # Check if the labels directory: test.csv - sample_submission.csv
    assert os.path.exists(os.path.join(args.metadata_dir,'train.csv')), \
        f"Directory {os.path.join(args.metadata_dir,'train.csv')} doesn't exist."
    assert os.path.exists(os.path.join(args.metadata_dir,'test.csv')), \
        f"Directory {os.path.join(args.metadata_dir,'test.csv')} doesn't exist."
    assert os.path.exists(os.path.join(args.results_dir,'sample_submission.csv')), \
        f"Directory {os.path.join(args.results_dir,'sample_submission.csv')} doesn't exist."

    # Check if the params directory: *.json
    assert os.path.exists(os.path.join(args.results_dir,args.params_file)), \
        f"File {os.path.join(args.results_dir,args.params_file)} doesn't exist."

if __name__ == '__main__':
    # move the current directory to the project root
    abspath = os.path.abspath(__file__)
    project_root_dir = os.path.join(os.path.dirname(abspath), '..')
    os.chdir(project_root_dir)

    # Parse the command line arguments
    args = parse_args()
    check_requirements(args)

    test_path = os.path.join(args.im_dir, 'test')
    if not os.path.exists(test_path):
        with ZipFile(file=test_path+'.zip') as zip_file:
            for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()),
                             desc='Extract testing data', ncols=100,
                             bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)):
                zip_file.extract(member=file, path=args.im_dir)

    results_dir_current_job = args.results_dir
    params_file = os.path.join(args.results_dir,args.params_file)
    submission_path = os.path.join(args.results_dir, 'sample_submission.csv')

    # Load training settings (e.g. hyperparameters)
    params = utils.Params(params_file)

    # Get the logger
    log_path = os.path.join(results_dir_current_job, 'testing.log')
    log_level = params.log_level if hasattr(params, 'log_level') else logging.DEBUG
    log = utils.get_logger(log_path, log_level)
    log.info(f"Results directory: {results_dir_current_job}")
    log.info(f"Hyperparameters: {params.dict}")
    # Check if any GPU is available
    use_gpu = torch.cuda.is_available()
    log.info(f"GPU is {'not ' if not use_gpu else ''}available")
    if use_gpu:
        gpu_count = torch.cuda.device_count()
        log.info(f"Available GPU count: {gpu_count}")
        device_id = int(args.gpu_device_id)
        torch.cuda.set_device(device_id)
        log.info(f"Chose CUDA device {device_id} ({torch.cuda.get_device_properties(device_id)})")

    # Set the random seed for reproducible experiments
    if params.seed is not None:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if use_gpu:
            torch.cuda.manual_seed(params.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        log.info(f"Set seed to {params.seed}")
    else:
        log.info(f"Running with random seed initialization")

    dataloaders_folds = dl.folds(args.im_dir,args.metadata_dir, ['test'], params.dataloader_settings)
    # load back the best weights (and set the dropout to 0)
    model = build_model(dataloaders_folds.meta_features, params.model_settings)
    utils.load_checkpoint(model, checkpoint_file=f'{results_dir_current_job}/checkpoint.pth')
    # move model to GPU
    model.cuda()
    model.eval()  # switch model to the evaluation mode
    # get predictions and AUC scores on the test fold
    log.info("Starting testing")
    preds = test_model(dataloaders_folds.dataloaders['test'], model,
                       results_dir_current_job,params.testing_settings,
                       dataloaders_folds.meta_features)
    log.info("Testing end")
