import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import process_zero_values
import ipdb


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class CrossEntropyClassWeighted(_Loss):

    def __init__(self, size_average=True, ignore_index=-100, reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyClassWeighted, self).__init__(size_average)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        return F.cross_entropy(input, target, weight, ignore_index=self.ignore_index, reduction=self.reduction)


### clone this function from: https://github.com/krumo/swd_pytorch/blob/master/swd_pytorch.py. [Unofficial]
def discrepancy_slice_wasserstein(p1, p2):
    s = p1.shape
    if s[1] > 1:
        proj = torch.randn(s[1], 128).cuda()
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))

    return wdist


class McDalNetLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=True):
        super(McDalNetLoss, self).__init__(weight, size_average)

    def forward(self, input1, input2, dis_type='L1'):

        if dis_type == 'L1':
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = torch.mean(torch.abs(prob_s - prob_t))  ### element-wise
        elif dis_type == 'CE':  ## Cross entropy
            loss = - ((F.log_softmax(input2, dim=1)).mul(F.softmax(input1, dim=1))).mean() - (
                (F.log_softmax(input1, dim=1)).mul(F.softmax(input2, dim=1))).mean()
            loss = loss * 0.5
        elif dis_type == 'KL':  ##### averaged over elements, not the real KL div (summed over elements of instance, and averaged over instance)
            ############# nn.KLDivLoss(size_average=False) Vs F.kl_div()
            loss = (F.kl_div(F.log_softmax(input1), F.softmax(input2))) + (
                F.kl_div(F.log_softmax(input2), F.softmax(input1)))
            loss = loss * 0.5
        ############# the following two distances are not evaluated in our paper, and need further investigation
        elif dis_type == 'L2':
            nClass = input1.size()[1]
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = torch.norm(prob_s - prob_t, p=2, dim=1).mean() / nClass  ### element-wise
        elif dis_type == 'Wasse':  ## distance proposed in Sliced wasserstein discrepancy for unsupervised domain adaptation,
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = discrepancy_slice_wasserstein(prob_s, prob_t)

        return loss


class TargetDiscrimLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(TargetDiscrimLoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)

        if (prob.data[:, self.num_classes:].sum(1) == 0).sum() != 0:  ########### in case of log(0)
            soft_weight = torch.FloatTensor(batch_size).fill_(0)
            soft_weight[prob[:, self.num_classes:].sum(1).data.cpu() == 0] = 1e-6
            soft_weight_var = soft_weight.cuda()
            loss = -((prob[:, self.num_classes:].sum(1) + soft_weight_var).log().mean())
        else:
            loss = -(prob[:, self.num_classes:].sum(1).log().mean())
        return loss

class SourceDiscrimLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(SourceDiscrimLoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)

        if (prob.data[:, :self.num_classes].sum(1) == 0).sum() != 0:  ########### in case of log(0)
            soft_weight = torch.FloatTensor(batch_size).fill_(0)
            soft_weight[prob[:, :self.num_classes].sum(1).data.cpu() == 0] = 1e-6
            soft_weight_var = soft_weight.cuda()
            loss = -((prob[:, :self.num_classes].sum(1) + soft_weight_var).log().mean())
        else:
            loss = -(prob[:, :self.num_classes].sum(1).log().mean())
        return loss


class ConcatenatedCELoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(ConcatenatedCELoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        prob = F.softmax(input, dim=1)
        prob_s = prob[:, :self.num_classes]
        prob_t = prob[:, self.num_classes:]

        prob_s = process_zero_values(prob_s)
        prob_t = process_zero_values(prob_t)
        loss = - (prob_s.log().mul(prob_t)).sum(1).mean() - (prob_t.log().mul(prob_s)).sum(1).mean()
        loss = loss * 0.5
        return loss



class ConcatenatedEMLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(ConcatenatedEMLoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        prob = F.softmax(input, dim=1)
        prob_s = prob[:, :self.num_classes]
        prob_t = prob[:, self.num_classes:]
        prob_sum = prob_s + prob_t
        prob_sum = process_zero_values(prob_sum)
        loss = - prob_sum.log().mul(prob_sum).sum(1).mean()

        return loss

class MinEntropyConsensusLoss(nn.Module):
    def __init__(self, num_classes):
        super(MinEntropyConsensusLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, y):
        i = torch.eye(self.num_classes).unsqueeze(0).cuda()
        x = F.log_softmax(x, dim=1)
        y = F.log_softmax(y, dim=1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        ce_x = (- 1.0 * i * x).sum(1)
        ce_y = (- 1.0 * i * y).sum(1)

        ce = 0.5 * (ce_x + ce_y).min(1)[0].mean()

        return ce