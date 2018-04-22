import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist
from visdom import Visdom
import copy
viz = Visdom()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, shared_layers, corase_classifier, n_corase, fine_encoders):
        super(Protonet, self).__init__()
        self.register_buffer('shared_layers', shared_layers)
        # self.shared_layers = shared_layers
        self.corase_classifier = corase_classifier
        self.n_corase = n_corase
        self.fine_encoders = fine_encoders

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)


        # share layers part
        z_share = self.shared_layers.forward(x)

        # corase classifier part
        z_corase = self.corase_classifier.forward(z_share)

        log_p_y_corase = F.log_softmax(z_corase)
       
        # fine feature part
        z = self.fine_encoders[0].forward(z_share)
        z_dim = z.size(-1)

        for i in range(1, self.n_corase):
           z += self.fine_encoders[i].forward(z_share)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.data[0],
            'acc': acc_val.data[0]
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    # shared_layers = nn.Sequential(
    #     conv_block(x_dim[0], hid_dim),
    #     conv_block(hid_dim, hid_dim),
    #     conv_block(hid_dim, hid_dim),
    # )

    model = torch.load('proto_results/m30_5way5shot/best_model.t7')

    # load pretrained layers 
    shared_layers = nn.Sequential(
        copy.deepcopy(model.encoder[0]),
        copy.deepcopy(model.encoder[1]),
        copy.deepcopy(model.encoder[2])
    )

    for param in shared_layers.parameters():
        param.requires_grad = False

    # TODO: make n_corase a commandline parameter
    n_corase = 1

    def gap_block(in_channels, out_channels, pre_size):
        return nn.Sequential(
            nn.Conv2d(hid_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(pre_size),
        )

    corase_classifier = nn.Sequential(
        gap_block(hid_dim, n_corase, x_dim[1] // 8),
        Flatten()
    )

    fine_encoders = []
    for i in range(n_corase):
        fine_encoders.append(
            nn.Sequential(
                conv_block(hid_dim, z_dim),
                Flatten()
        ))
    # encoder = nn.Sequential(
    #     conv_block(x_dim[0], hid_dim),
    #     conv_block(hid_dim, hid_dim),
    #     conv_block(hid_dim, hid_dim),
    #     conv_block(hid_dim, z_dim),
    #     Flatten()
    # )

    return Protonet(shared_layers, corase_classifier, n_corase, fine_encoders)
