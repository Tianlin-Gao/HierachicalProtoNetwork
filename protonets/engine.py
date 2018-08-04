from tqdm import tqdm

class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = { }
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False
        }

        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])
        import torch
        corase_optimizer = torch.optim.Adam(state['model'].corase_classifier.parameters(), lr = 0.001)

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                corase_optimizer.zero_grad()
                loss0, state['output'] = state['model'].corase_loss(state['sample'], 0, [1, 2, 3, 4])
                loss1, state['output'] = state['model'].corase_loss(state['sample'], 1, [0, 2, 3, 4])
                loss2, state['output'] = state['model'].corase_loss(state['sample'], 2, [0, 1, 3, 4])
                loss3, state['output'] = state['model'].corase_loss(state['sample'], 3, [0, 1, 2, 4])
                loss4, state['output'] = state['model'].corase_loss(state['sample'], 4, [0, 1, 2, 3])
                loss = (loss0 + loss1 + loss2 + loss3 + loss4) / 5
                loss.backward()

                state['optimizer'].zero_grad()
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
