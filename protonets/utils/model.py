from tqdm import tqdm

from protonets.utils import filter_opt
from protonets.models import get_model

def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def evaluate(model, data_loader, meters, desc=None):

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        import copy
        import torch
        tmpModel = copy.deepcopy(model)
        tmpModel.eval()
        tmpModel.corase_classifier.train()
        corase_optimizer = torch.optim.Adam(tmpModel.corase_classifier.parameters(), lr = 0.001)
        corase_optimizer.zero_grad()
        loss, _ = tmpModel.corase_loss(sample)
        loss.backward()
        _, output = tmpModel.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
