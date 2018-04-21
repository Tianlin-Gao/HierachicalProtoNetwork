import os
import json
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import torchvision
import torchnet as tnt

import sys
import os.path
sys.path.append('/home/cx/lifelong_learning/prototypical-networks')
print(sys.path)
from protonets.engine import Engine

import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils
import protonets.utils.visual as visual_utils

from visdom import Visdom
viz = Visdom()
#设置了state的回调方法?hook钩子，有点前端的感觉

def main(opt):
    # 新建日志目录
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])

    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)
    
    #??? trainval是什么???
    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']


    model = model_utils.load(opt)
    #model = torch.load("results/m5_5way5shot/pre.t7")

    if opt['data.cuda']:
        model.cuda()

    engine = Engine()

    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }

    if val_loader is not None:
        meters['val'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }
    
    # 看名字知道功能的start函数，配置优化器
    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)
    engine.hooks['on_start'] = on_start
    
    # 第一个epoch需要解决的事
    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()
    engine.hooks['on_start_epoch'] = on_start_epoch
    
    # 更新那个算平均的类
    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])
    engine.hooks['on_update'] = on_update
    
    #一个epoch结束时判断训练效果，以及是否结束训练(patience?为什么不用loss的改变?看了实际训练貌似loss变化挺大的)
    title = '%s, %s: %i_%iw_%is'%(opt['model.exp_name'], opt['data.dataset'], 
        opt['data.way'], opt['data.test_way'], opt['data.test_shot'])
    lossPic = visual_utils.train_val_loss(title)
    accPic = visual_utils.train_val_acc(title)
    def on_end_epoch(hook_state, state):
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            model_utils.evaluate(state['model'],
                                 val_loader,
                                 meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))
 
        meter_vals = log_utils.extract_meter_values(meters)
        lossPic(state['epoch'], meter_vals['train']['loss'], meter_vals['val']['loss'])
        accPic(state['epoch'], meter_vals['train']['acc'], meter_vals['val']['acc'])
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['model'].cpu()
                torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.t7'))
                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.t7'))
            if opt['data.cuda']:
                state['model'].cuda()

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, { })

    engine.train(
        model = model,
        loader = train_loader,
        optim_method = getattr(optim, opt['train.optim_method']),
        optim_config = { 'lr': opt['train.learning_rate'],
                         'weight_decay': opt['train.weight_decay'] },
        max_epoch = opt['train.epochs']
    )
