import os
import json
import math
from tqdm import tqdm
import pickle

import torch
import torchnet as tnt
from torch.autograd import Variable

from visdom import Visdom
from PIL import Image

viz = Visdom()

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils

from functools import reduce

def get_cache_path(split):                                             
    cache_path = ("../data/mini-imagenet/mini-imagenet-cache-" + split + ".pkl")
    return cache_path  

def nextBatchOfClass(nClass, labelDecoders):
    def toByte(s):
        return s.encode()
    if nClass < 64:
        split = 'train-handout'
    elif nClass >= 64 and nClass < 80:
        split = 'val-handout'
    else:
        split = 'test-handout'    
    cache_path = get_cache_path(split)
    with open(cache_path, "rb") as f:                                   
        try:                                                       
            data = pickle.load(f, encoding='bytes')                   
            img_data = data[b'image_data']                         
            class_dict = data[b'class_dict']                       
        except:                                                    
            data = pickle.load(f)                                     
            img_data = data['image_data']                          
            class_dict = data['class_dict']
    data = img_data[class_dict[toByte(labelDecoders[nClass])]].transpose(0, 3, 1, 2) / 255.0
    return Variable(torch.Tensor(data))

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def nearestMean(model, data):
    n = data['num']
    z = model.encoder.forward(data['data'])
    z_dim = z.size(-1)
    return z.view(n, z_dim).mean(0).view(1, z_dim).data
    
def getProtoDistMatrix(opt):
    model = torch.load(opt['model_path'])
    model.eval()
    
    if opt['cuda']:
        model.cuda()
    
    protos = []
    
    for i in range(opt['nClass']):
        print(i, opt['labelDecoders'][i])
        dataLoader = dict(data = nextBatchOfClass(i, opt['labelDecoders']), num = 200)
        if opt['cuda']:
            dataLoader['data'] = dataLoader['data'].cuda()
        protos.append(nearestMean(model, dataLoader))
    
    return euclidean_dist(torch.cat(protos), torch.cat(protos)), torch.cat(protos)

def savePickle(obj, file):
    with open(file, 'wb') as fh:
        pickle.dump(obj, fh)

def labelEncoder(split, beginner):
    cache_path = get_cache_path(split)                                  
    def toStr(s):
        return str(s, encoding='utf-8')
    
    with open(cache_path, "rb") as f:                                   
        try:                                                       
            data = pickle.load(f, encoding='bytes')                   
            img_data = data[b'image_data']                         
            class_dict = data[b'class_dict']                       
        except:                                                    
            data = pickle.load(f)                                     
            img_data = data['image_data']                          
            class_dict = data['class_dict']
    n = len(class_dict.keys())
    return dict(zip(map(toStr, list(class_dict.keys())), range(beginner, beginner+n))), beginner + n

if __name__ == '__main__':
    model_path = ['../proto_results/m30_5way5shot/best_model.t7']
    dm_path = ['../proto_results/m30_5way5shot/dm.pkl']
    proto_path = ['../proto_results/m30_5way5shot/protos.pkl']
    
    splits = ['train', 'val', 'test']
    labelDecoders = [None]*100
    try:
        with open('../labelEncoders.pkl', 'rb') as fp:
            labelEncoders = pickle.load(fp)
        with open('../labelDecoders.pkl', 'rb') as fp:
            labelDecoders = pickle.load(fp)
    except:
        labelEncoders = []
        counter = 0
        for split in splits:
            encoder, counter = labelEncoder(split, counter)
            labelEncoders.append(encoder)
        labelEncoders = reduce(lambda x,y: {**x, **y}, labelEncoders)
        for k,v in labelEncoders.items():
            labelDecoders[v] = k
        savePickle(labelEncoders, '../labelEncoders.pkl')
        savePickle(labelDecoders, '../labelDecoders.pkl')

    for path, dmPath, protoPath in zip(model_path, dm_path, proto_path):
        distMatrix, protos = getProtoDistMatrix(dict(model_path = path, nClass = 100, cuda=True, labelDecoders=labelDecoders))
        im = Image.fromarray(distMatrix.cpu().numpy())
        savePickle(protos.cpu().numpy(), protoPath) 
        savePickle(distMatrix.cpu().numpy(), dmPath)
#     im.show()
        viz.image(distMatrix.cpu().numpy())