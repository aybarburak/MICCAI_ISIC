import argparse
import logging
import os
from shutil import copyfile
from zipfile import ZipFile
from tqdm import tqdm
from colorama import Fore


import numpy as np
import torch

import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_dir', default='./dataset', help="Directory containing the SIIM-ISIC Melanoma images")
    parser.add_argument('--metadata_dir', default='./labels', help="Directory containing metadata files")
    parser.add_argument('--results_dir', default='./results',
                        help="Root directory where to save the model checkpoints and results")
    parser.add_argument('--params_file', default='./params/Melanoma_params.json',
                        help="Path to the file  containing the training settings")
    parser.add_argument('--gpu_device_id', default=0, type=int,
                        help='GPU device number to run on (e.g. when multiple GPUs are available')
    parser.add_argument("--logdir", default="logdir", type=str, help="tensorboard log directory")
    return parser.parse_args()


def check_requirements(args):
    # Check if the dataset directory: train.zip - test.zip
    assert os.path.exists(os.path.join(args.im_dir,'train.zip')), \
        f"Directory {os.path.join(args.im_dir,'train.zip')} doesn't exist."
    assert os.path.exists(os.path.join(args.im_dir,'test.zip')),\
        f"Directory {os.path.join(args.im_dir,'test.zip')} doesn't exist."
    # Check if the labels directory: train.csv - test.csv - sample_submission.csv
    assert os.path.exists(os.path.join(args.metadata_dir,'train.csv')), \
        f"Directory {os.path.join(args.metadata_dir,'train.csv')} doesn't exist."
    assert os.path.exists(os.path.join(args.metadata_dir,'test.csv')), \
        f"Directory {os.path.join(args.metadata_dir,'test.csv')} doesn't exist."
    assert os.path.exists(os.path.join(args.metadata_dir,'sample_submission.csv')), \
        f"Directory {os.path.join(args.metadata_dir,'sample_submission.csv')} doesn't exist."

    # Check if the params directory: *.json
    assert os.path.exists(args.params_file), f"File {args.params_file} doesn't exist."


if __name__ == '__main__':
    # move the current directory to the project root
    abspath = os.path.abspath(__file__)
    project_root_dir = os.path.join(os.path.dirname(abspath), '..')
    os.chdir(project_root_dir)

    # Parse the command line arguments
    args = parse_args()
    check_requirements(args)

    # Extract dataset
    train_path = os.path.join(args.im_dir, 'train')
    if not os.path.exists(train_path):
        with ZipFile(file=train_path+'.zip') as zip_file:
            for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()),
                             desc='Extract training data', ncols=100,
                             bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)):
                zip_file.extract(member=file, path=args.im_dir)

    test_path = os.path.join(args.im_dir, 'test')
    if not os.path.exists(test_path):
        with ZipFile(file=test_path+'.zip') as zip_file:
            for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()),
                             desc='Extract testing data', ncols=100,
                             bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)):
                zip_file.extract(member=file, path=args.im_dir)

    # Create a directory for the current experiment
    results_dir_current_job = os.path.join(args.results_dir, utils.now_as_str_f())
    while os.path.isdir(results_dir_current_job):  # generate a new timestamp if the current one already exists
        results_dir_current_job = os.path.join(args.results_dir, utils.now_as_str_f())
    os.makedirs(results_dir_current_job)

    # Copy the settings file into the results directory
    copyfile(args.params_file, os.path.join(results_dir_current_job, os.path.basename(args.params_file)))

    submission_path = os.path.join(args.metadata_dir, 'sample_submission.csv')
    copyfile(submission_path, os.path.join(results_dir_current_job, os.path.basename(submission_path)))

    # Load training settings (e.g. hyperparameters)
    params = utils.Params(args.params_file)

    # Get the logger
    log_path = os.path.join(results_dir_current_job, 'training.log')
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
        log.info(f"Choose CUDA device {device_id} ({torch.cuda.get_device_properties(device_id)})")

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

    train.experiment_run(args.im_dir, args.metadata_dir, results_dir_current_job, params, logdir=args.logdir)
    log.info("Experiment end")
