import torch
import numpy as np

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_cpu(x):
    return x.cpu()

def to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    identity = torch.eye(num_classes).to(label.device)
    onehot = torch.index_select(identity, 0, label)
    return onehot

def accuracy(output, target):
    """Computes the precision"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = correct[:1].view(-1).float().sum(0, keepdim=True)
    res = correct.mul_(100.0 / batch_size)
    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[target[i]] += 1
        correct_vector[torch.LongTensor([target[i]])] += correct[i]

    return total_vector, correct_vector

def process_one_values(tensor):
    if (tensor == 1).sum() != 0:
        eps = torch.FloatTensor(tensor.size()).fill_(0)
        eps[tensor.data.cpu() == 1] = 1e-6
        tensor = tensor - eps.cuda()
    return tensor

def process_zero_values(tensor):
    if (tensor == 0).sum() != 0:
        eps = torch.FloatTensor(tensor.size()).fill_(0)
        eps[tensor.data.cpu() == 0] = 1e-6
        tensor = tensor + eps.cuda()
    return tensor


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count