import torch
import torch.nn as nn
import numpy as np


def get_loss(params, label_freq=0):
    """
    :return: the loss criterion
    """
    if params['label_weighting_strategy'] is None:
        return nn.BCEWithLogitsLoss()
    elif params['label_weighting_strategy'] == "inverse_frequency":
        weights = 1.0 / label_freq
        weights = torch.from_numpy(np.array(weights)).type(torch.FloatTensor)

        if torch.cuda.is_available():
            weights = weights.cuda()

        return nn.BCEWithLogitsLoss(weight=weights)
    elif params['label_weighting_strategy'] == 'inverse_sqrt_frequency':
        weights = 1.0 / np.sqrt(label_freq)
        weights = torch.from_numpy(np.array(weights)).type(torch.FloatTensor)

        if torch.cuda.is_available():
            weights = weights.cuda()

        return nn.BCEWithLogitsLoss(weight=weights)
    else:
        assert False
