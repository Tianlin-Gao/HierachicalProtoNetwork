import torch
from torch.autograd import Variable

class HistogramLoss(torch.nn.Module):
    def __init__(self, num_steps, cuda=True):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.cuda = cuda
        self.t = torch.range(-1, 1, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        if self.cuda:
            self.t = self.t.cuda()
        
    def forward(self, features, classes):
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            print(delta_repeat)
            print(self.t - self.step)
            indsa = (delta_repeat == (self.t - self.step)) & inds

            indsb = (delta_repeat == self.t) & inds
            s_repeat_[~(indsb|indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - Variable(self.t) + self.step)[indsa] / self.step
            s_repeat_[indsb] =  (-s_repeat_ + Variable(self.t) + self.step)[indsb] / self.step
            
            return s_repeat_.sum(1) / size
        
        classes_size = classes.size()[0]
        # print(classes.repeat(classes_size, 1))
        # print(classes.view(-1, 1).repeat(1, classes_size))
        classes_eq = (classes.repeat(classes_size, 1)  == classes.view(-1, 1).repeat(1, classes_size)).data
        dists = torch.mm(features, features.transpose(0, 1))
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
        # if self.cuda:
        #     s_inds= s_inds.cuda()
        pos_inds = classes_eq.cpu()[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq.cpu()[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq.cpu()[s_inds].sum()
        neg_size = (~classes_eq.cpu()[s_inds]).sum()
        s = dists.cpu()[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        delta_repeat = (torch.floor((s_repeat.data + 1) / self.step) * self.step - 1).float()
        
        histogram_pos = histogram(pos_inds, pos_size)
        histogram_neg = histogram(neg_inds, neg_size)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
        # if self.cuda:
        #     histogram_pos_inds = histogram_pos_inds.cuda()
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        if self.cuda:
            loss = torch.sum((histogram_neg * histogram_pos_cdf).cuda())
        else:
            loss = torch.sum(histogram_neg * histogram_pos_cdf)
        return loss
    
