import torch.optim as optim


def get_optimizer(model, params):
    """
    Creates an optimizer for the model based on the settings given in params.

    :param model: the model which has to be trained
    :type model: torch.nn.Module
    :param params: the object containing the hyperparameters and other training settings
    :type params: utils.Params
    :return: the optimizer
    """
    optimizer = None
    optimizer_name = params['optimizer']

    if optimizer_name == "SGD":
        lr = params['learning_rate']
        mom = params['momentum']
        wd = params['weight_decay']
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom,
                              weight_decay=wd)
    elif optimizer_name == "Adam":
        lr = params['learning_rate']
        var1 = params['beta1']
        var2 = params['beta2']
        wd = params['weight_decay']
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=[var1, var2],
                               weight_decay=wd)

    return optimizer