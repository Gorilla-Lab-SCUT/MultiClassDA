import torch
import torch.nn as nn
import os
from utils.utils import to_cuda, AverageMeter
from config.config import cfg

class BaseSolver:
    def __init__(self, net, dataloaders, **kwargs):
        self.opt = cfg
        self.net = net
        self.dataloaders = dataloaders
        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda()
        self.epoch = 0
        self.iters = 0
        self.best_prec1 = 0
        self.iters_per_epoch = None
        self.build_optimizer()
        self.init_data(self.dataloaders)

    def init_data(self, dataloaders):
        self.train_data = {key: dict() for key in dataloaders if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloaders:
                continue
            cur_dataloader = dataloaders[key]
            self.train_data[key]['loader'] = cur_dataloader
            self.train_data[key]['iterator'] = None

        if 'test' in dataloaders:
            self.test_data = dict()
            self.test_data['loader'] = dataloaders['test']

    def build_optimizer(self):
        print('Optimizer built')


    def complete_training(self):
        if self.epoch > self.opt.TRAIN.MAX_EPOCH:
            return True

    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample


    def update_network(self, **kwargs):
        pass