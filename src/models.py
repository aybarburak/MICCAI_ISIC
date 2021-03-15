import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from efficientnet_pytorch import EfficientNet

log = utils.get_logger()


class MetaFeaturesEfficientNet(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(MetaFeaturesEfficientNet, self).__init__()
        self.arch = arch

        self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 50 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.output = nn.Linear(500 + 250, 1)

    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.output(features)
        return output


class RawEfficientNet(nn.Module):
    def __init__(self, arch):
        super(RawEfficientNet, self).__init__()
        self.arch = arch
        self.arch._fc = nn.Linear(in_features=1280, out_features=256, bias=True)
        self.feature_agg = nn.Linear(in_features=256, out_features=128, bias=True)

        self.output = nn.Linear(128, 1)

    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x = inputs
        features = self.arch(x)
        features = F.relu(self.feature_agg(features))
        output = self.output(features)
        return output


def build_model(meta_features,params):
    """
       return efficient net model used
       :param dataloaders_folds: folds of data
       :param params: containing hyperaparameters of model settings
       """
    arch = EfficientNet.from_pretrained(params["efficientnet_model"])

    if meta_features is not None:
        return MetaFeaturesEfficientNet(arch=arch, n_meta_features=len(meta_features))
    else:
        return RawEfficientNet(arch=arch)