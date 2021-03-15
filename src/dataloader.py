import os
import numpy as np
import pandas as pd
import torch
import cv2
import torchtoolbox.transform as transforms
import utils
from torchvision.transforms import transforms as tfms

from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from rand_augment import RandAugment

log = utils.get_logger()

# use imagenet mean, std for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def data_augmentation(augmentation_strategy):
    """
       Train torchvision model to SIIM data given high level hyperparameters.
       Then it evaluates it on the test data and write the submission results.
       :param augmentation_strategy: name of data transform used
       :return dict for train and val data augmentation sequence
       """

    DATA_TRANSFORMS = {}

    if augmentation_strategy is None:
        DATA_TRANSFORMS = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                transforms.Cutout(scale=(0.05, 0.007), value=(0, 0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
        }
    return DATA_TRANSFORMS


class MelanomaDataset(Dataset):
    """
    Class used to model the SIIM ISIC Dataset, together with the transformations applied on the data as needed
    by the EfficientNet architecture.
    """

    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None, meta_features=None):
        """
         :param   df (pd.DataFrame): DataFrame with data description
         :param   imfolder (str): folder with images
         :param   train (bool): flag of whether a training dataset is being initialized or testing one
         :param   transforms: image transformation method to be applied
         :param   meta_features (list): list of features with meta information, such as sex and age
        """
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        x = cv2.imread(im_path)
        if self.transforms:
            x = self.transforms(x)

        input = tuple()
        if self.meta_features is not None:
            meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)
            input = (x, meta)
        else:
            input = x

        if self.train:
            y = self.df.loc[index]['target']
            return input, y
        else:
            return input

    def __len__(self):
        return len(self.df)


class folds:
    def __init__(self, im_dir, metadata_dir, mode, params):
        """
        Returns the dataloaders for the train, validation, and test.

        :param im_dir: the directory of the images
        :param metadata_dir: the directory of the csv files
        :param mode: train, test string list
        :param params: containing hyperaparameters of dataloader settings
        :return: returns the DataLoader objects for training, validation and testing and metafeatures length
        """
        # set the transformation for each fold
        self.transforms_dict = {}
        self.df = {}
        self.img_path = {}
        self.dataloaders = {}
        self.meta_features = None
        self.augments = {}

        DATA_TRANSFORMS = data_augmentation(params['augmentation_strategy'])
        dataloader_list = []

        self.train_df = pd.read_csv(os.path.join(metadata_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(metadata_dir, 'test.csv'))

        if params['use_metafeatures']:
            concat = pd.concat(
                [self.train_df['anatom_site_general_challenge'], self.test_df['anatom_site_general_challenge']],
                ignore_index=True)
            dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')

            self.train_df = pd.concat([self.train_df, dummies.iloc[:self.train_df.shape[0]]], axis=1)
            self.test_df = pd.concat([self.test_df, dummies.iloc[self.train_df.shape[0]:].reset_index(drop=True)],
                                     axis=1)
            self.meta_features = ['sex', 'age_approx'] + [col for col in self.train_df.columns if 'site_' in col]
            self.meta_features.remove('anatom_site_general_challenge')

        # Training/Validation data
        if 'train' in mode:
            if params['use_rand_aug']:
                rand_augment = RandAugment(params['rand_aug'])
                self.augments = rand_augment.augments
                self.transforms_dict['train'] = tfms.Compose([
                    tfms.ToPILImage(),
                    rand_augment,
                    tfms.ToTensor(),
                    tfms.Normalize(MEAN, STD)
                ])
            else:
                self.transforms_dict['train'] = DATA_TRANSFORMS['train']

            self.transforms_dict['val'] = DATA_TRANSFORMS['val']

            if params['use_metafeatures']:
                # Sex features
                self.train_df['sex'] = self.train_df['sex'].map({'male': 1, 'female': 0})
                self.train_df['sex'] = self.train_df['sex'].fillna(-1)
                # Age features
                self.train_df['age_approx'] /= self.train_df['age_approx'].max()
                self.train_df['age_approx'] = self.train_df['age_approx'].fillna(0)
                # patient id
                self.train_df['patient_id'] = self.train_df['patient_id'].fillna(0)

            skf = GroupKFold(n_splits=params['num_splits'])
            splits = list(skf.split(X=np.zeros(len(self.train_df)), y=self.train_df['target'],
                                    groups=self.train_df['patient_id'].tolist()))

            self.train_idx, self.val_idx = splits[params['fold_idx']]
            log.info(f"Run on Fold: {params['fold_idx']}")

            self.df['train'] = self.train_df.iloc[self.train_idx].reset_index(drop=True)
            self.df['val'] = self.train_df.iloc[self.val_idx].reset_index(drop=True)

            self.img_path['train'] = os.path.join(im_dir, 'train')
            self.img_path['val'] = os.path.join(im_dir, 'train')

            dataloader_list.append('train')
            dataloader_list.append('val')
            # compute label frequency
            target_dict = self.train_df['target'].value_counts().to_dict()
            self.label_freq = np.zeros(1)
            self.label_freq[0] = target_dict[1]
            # for i, cl in target_dict.items():
            #    self.label_freq[i] = cl

            self.label_freq /= len(self.train_df)

        # Testing data
        if 'test' in mode:
            if params['use_test_augmentation']:
                if params['use_rand_aug']:
                    rand_augment = RandAugment(params['rand_aug'])
                    self.augments = rand_augment.augments
                    self.transforms_dict['test'] = tfms.Compose([
                        tfms.ToPILImage(),
                        rand_augment,
                        tfms.ToTensor(),
                        tfms.Normalize(MEAN, STD)
                    ])
                else:
                    self.transforms_dict['test'] = DATA_TRANSFORMS['train']
            else:
                self.transforms_dict['test'] = DATA_TRANSFORMS['val']

            if params['use_metafeatures']:
                # Sex features
                self.test_df['sex'] = self.test_df['sex'].map({'male': 1, 'female': 0})
                self.test_df['sex'] = self.test_df['sex'].fillna(-1)
                # Age features
                self.test_df['age_approx'] /= self.test_df['age_approx'].max()
                self.test_df['age_approx'] = self.test_df['age_approx'].fillna(0)

            self.df['test'] = self.test_df
            self.img_path['test'] = os.path.join(im_dir, 'test')
            dataloader_list.append('test')

        # prepare the dataloader for each fold
        for k in dataloader_list:
            self.dataloaders[k] = generate_dataloader(data_frame=self.df[k],
                                                      im_dir=self.img_path[k],
                                                      transformation=self.transforms_dict[k],
                                                      meta_features=self.meta_features,
                                                      train_ind=False if k == 'test' else True,
                                                      batch_size=params['batch_size'],
                                                      num_workers=params['num_workers'],
                                                      shuffle=True if k == 'train' else False)


def generate_dataloader(data_frame, im_dir, transformation, meta_features, train_ind,
                        batch_size=16, num_workers=8, shuffle=False, dataset_class=MelanomaDataset):
    """
    Function that generates dataloader for train,val or test
    :param data_frame: data frame includes csv data
    :param im_dir: the images root directory
    :param transformation: one transformation from the given dataset class
    :param meta_features: include meta features names which are inside csv file
    :param train_ind: 'train', 'val' is True and 'test' is False
    :param batch_size: the batch size used for the DataLoader
    :param num_workers: how many workers to use for loading the data
    :param shuffle: if data has to be shuffled or not
    :param dataset_class: dataset class
    :return: the DataLoader object
    """

    transformed_dataset = dataset_class(df=data_frame,
                                        imfolder=im_dir,
                                        train=train_ind,
                                        transforms=transformation,
                                        meta_features=meta_features)

    dataloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle,
                                             num_workers=num_workers)

    return dataloader
