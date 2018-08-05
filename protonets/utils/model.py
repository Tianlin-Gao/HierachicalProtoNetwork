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
        corase_optimizer = torch.optim.Adam(tmpModel.corase_classifier.parameters(), lr = 0.01)
        corase_optimizer.zero_grad()
        loss0, _ = tmpModel.corase_loss(sample, 0, [1, 2, 3, 4])
        loss1, _ = tmpModel.corase_loss(sample, 1, [0, 2, 3, 4])
        loss2, _ = tmpModel.corase_loss(sample, 2, [0, 1, 3, 4])
        loss3, _ = tmpModel.corase_loss(sample, 3, [0, 1, 2, 4])
        loss4, _ = tmpModel.corase_loss(sample, 4, [0, 1, 2, 3])
        loss = (loss0 + loss1 + loss2 + loss3 + loss4) / 5
        loss.backward()
        _, output = tmpModel.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
