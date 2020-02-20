import torch
import torch.nn as nn
import os
import math
import time
from utils.utils import to_cuda, accuracy_for_each_class, accuracy, AverageMeter, process_one_values
from config.config import cfg
import torch.nn.functional as F
from models.loss_utils import McDalNetLoss
from .base_solver import BaseSolver
import ipdb

class McDalNetSolver(BaseSolver):
    def __init__(self, net, dataloaders, **kwargs):
        super(McDalNetSolver, self).__init__(net, dataloaders, **kwargs)
        self.BCELoss = nn.BCEWithLogitsLoss().cuda()
        self.McDalNetLoss = McDalNetLoss().cuda()
        if cfg.RESUME != '':
            resume_dict = torch.load(cfg.RESUME)
            model_state_dict = resume_dict['model_state_dict']
            self.net.load_state_dict(model_state_dict)
            self.best_prec1 = resume_dict['best_prec1']
            self.epoch = resume_dict['epoch']

    def solve(self):
        stop = False
        while not stop:
            stop = self.complete_training()
            self.update_network()
            acc = self.test()
            if acc > self.best_prec1:
                self.best_prec1 = acc
                self.save_ckpt()
            self.epoch += 1


    def update_network(self, **kwargs):
        stop = False
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        self.iters_per_epoch = len(self.train_data['target']['loader'])
        iters_counter_within_epoch = 0
        data_time = AverageMeter()
        batch_time = AverageMeter()
        total_loss = AverageMeter()
        ce_loss = AverageMeter()
        da_loss = AverageMeter()
        prec1_task = AverageMeter()
        prec1_aux1 = AverageMeter()
        prec1_aux2 = AverageMeter()
        self.net.train()
        end = time.time()
        if self.opt.TRAIN.PROCESS_COUNTER == 'epoch':
            lam = 2 / (1 + math.exp(-1 * 10 * self.epoch / self.opt.TRAIN.MAX_EPOCH)) - 1
            self.update_lr()
            print('value of lam is: %3f' % (lam))
        while not stop:
            if self.opt.TRAIN.PROCESS_COUNTER == 'iteration':
                lam = 2 / (1 + math.exp(-1 * 10 * self.iters / (self.opt.TRAIN.MAX_EPOCH * self.iters_per_epoch))) - 1
                print('value of lam is: %3f' % (lam))
                self.update_lr()
            source_data, source_gt = self.get_samples('source')
            target_data, _ = self.get_samples('target')
            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            target_data = to_cuda(target_data)
            data_time.update(time.time() - end)

            feature_source, output_source, output_source1, output_source2, output_source_dc, output_source1_trunc, output_source2_trunc = self.net(source_data, lam)
            loss_task_auxiliary_1 = self.CELoss(output_source1_trunc, source_gt)
            loss_task_auxiliary_2 = self.CELoss(output_source2_trunc, source_gt)
            loss_task = self.CELoss(output_source, source_gt)
            if self.opt.MCDALNET.DISTANCE_TYPE != 'SourceOnly':
                feature_target, output_target, output_target1, output_target2, output_target_dc, output_target1_trunc, output_target2_trunc = self.net(target_data, lam)
                if self.opt.MCDALNET.DISTANCE_TYPE == 'DANN':
                    num_source = source_data.size()[0]
                    num_target = target_data.size()[0]
                    dlabel_source = to_cuda(torch.zeros(num_source, 1))
                    dlabel_target = to_cuda(torch.ones(num_target, 1))
                    loss_domain_all = self.BCELoss(output_source_dc, dlabel_source) + self.BCELoss(output_target_dc, dlabel_target)
                    loss_all = loss_task + loss_domain_all
                elif self.opt.MCDALNET.DISTANCE_TYPE == 'MDD':
                    prob_target1 = F.softmax(output_target1, dim=1)
                    _, target_pseudo_label = torch.topk(output_target2, 1)
                    batch_index = torch.arange(output_target.size()[0]).long()
                    pred_gt_prob = prob_target1[batch_index, target_pseudo_label]  ## the prob values of the predicted gt
                    pred_gt_prob = process_one_values(pred_gt_prob)
                    loss_domain_target = (1 - pred_gt_prob).log().mean()

                    _, source_pseudo_label = torch.topk(output_source2, 1)
                    loss_domain_source = self.CELoss(output_source1, source_pseudo_label[:, 0])
                    loss_domain_all = loss_domain_source - loss_domain_target
                    loss_all = loss_task + loss_domain_all + loss_task_auxiliary_1 + loss_task_auxiliary_2
                else:
                    loss_domain_source = self.McDalNetLoss(output_source1, output_source2, self.opt.MCDALNET.DISTANCE_TYPE)
                    loss_domain_target = self.McDalNetLoss(output_target1, output_target2, self.opt.MCDALNET.DISTANCE_TYPE)
                    loss_domain_all = loss_domain_source - loss_domain_target
                    loss_all = loss_task + loss_domain_all + loss_task_auxiliary_1 + loss_task_auxiliary_2
                da_loss.update(loss_domain_all, source_data.size()[0])
            else:
                loss_all = loss_task
            ce_loss.update(loss_task, source_data.size()[0])
            total_loss.update(loss_all, source_data.size()[0])
            prec1_task.update(accuracy(output_source, source_gt), source_data.size()[0])
            prec1_aux1.update(accuracy(output_source1, source_gt), source_data.size()[0])
            prec1_aux2.update(accuracy(output_source2, source_gt), source_data.size()[0])

            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()

            print("  Train:epoch: %d:[%d/%d], LossCE: %3f, LossDA: %3f, LossAll: %3f, Auxi1: %3f, Auxi2: %3f, Task: %3f" % \
                  (self.epoch, iters_counter_within_epoch, self.iters_per_epoch, ce_loss.avg, da_loss.avg, total_loss.avg, prec1_aux1.avg, prec1_aux2.avg, prec1_task.avg))

            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            iters_counter_within_epoch += 1
            if iters_counter_within_epoch >= self.iters_per_epoch:
                log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
                log.write("\n")
                log.write("  Train:epoch: %d:[%d/%d], LossCE: %3f, LossDA: %3f, LossAll: %3f, Auxi1: %3f, Auxi2: %3f, Task: %3f" % \
                    (self.epoch, iters_counter_within_epoch, self.iters_per_epoch, ce_loss.avg, da_loss.avg, total_loss.avg, prec1_aux1.avg, prec1_aux2.avg, prec1_task.avg))
                log.close()
                stop = True


    def test(self):
        self.net.eval()
        prec1_task = AverageMeter()
        prec1_auxi1 = AverageMeter()
        prec1_auxi2 = AverageMeter()
        counter_all = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_all_auxi1 = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_all_auxi2 = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_acc = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_acc_auxi1 = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_acc_auxi2 = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)

        for i, (input, target) in enumerate(self.test_data['loader']):
            input, target = to_cuda(input), to_cuda(target)
            with torch.no_grad():
                _, output_test, output_test1, output_test2, _, _, _ = self.net(input, 1)  ## the value of lam do not affect the test process

            if self.opt.EVAL_METRIC == 'accu':
                prec1_task_iter = accuracy(output_test, target)
                prec1_auxi1_iter = accuracy(output_test1, target)
                prec1_auxi2_iter = accuracy(output_test2, target)
                prec1_task.update(prec1_task_iter, input.size(0))
                prec1_auxi1.update(prec1_auxi1_iter, input.size(0))
                prec1_auxi2.update(prec1_auxi2_iter, input.size(0))
                if i % self.opt.PRINT_STEP == 0:
                    print("  Test:epoch: %d:[%d/%d], Auxi1: %3f, Auxi2: %3f, Task: %3f" % \
                          (self.epoch, i, len(self.test_data['loader']), prec1_auxi1.avg, prec1_auxi2.avg, prec1_task.avg))
            elif self.opt.EVAL_METRIC == 'accu_mean':
                prec1_task_iter = accuracy(output_test, target)
                prec1_task.update(prec1_task_iter, input.size(0))
                counter_all, counter_acc = accuracy_for_each_class(output_test, target, counter_all, counter_acc)
                counter_all_auxi1, counter_acc_auxi1 = accuracy_for_each_class(output_test1, target, counter_all_auxi1, counter_acc_auxi1)
                counter_all_auxi2, counter_acc_auxi2 = accuracy_for_each_class(output_test2, target, counter_all_auxi2, counter_acc_auxi2)
                if i % self.opt.PRINT_STEP == 0:
                    print("  Test:epoch: %d:[%d/%d], Task: %3f" % \
                          (self.epoch, i, len(self.test_data['loader']), prec1_task.avg))
            else:
                raise NotImplementedError
        acc_for_each_class = counter_acc / counter_all
        acc_for_each_class_auxi1 = counter_acc_auxi1 / counter_all_auxi1
        acc_for_each_class_auxi2 = counter_acc_auxi2 / counter_all_auxi2
        log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
        log.write("\n")
        if self.opt.EVAL_METRIC == 'accu':
            log.write(
                "                                    Test:epoch: %d, Top1_auxi1: %3f, Top1_auxi2: %3f, Top1: %3f" % \
                (self.epoch, prec1_auxi1.avg, prec1_auxi2.avg, prec1_task.avg))
            log.close()
            return max(prec1_auxi1.avg, prec1_auxi2.avg, prec1_task.avg)
        elif self.opt.EVAL_METRIC == 'accu_mean':
            log.write(
                "                                    Test:epoch: %d, Top1_auxi1: %3f, Top1_auxi2: %3f, Top1: %3f" % \
                (self.epoch, acc_for_each_class_auxi1.mean(), acc_for_each_class_auxi2.mean(), acc_for_each_class.mean()))
            log.write("\nClass-wise Acc:")  ## based on the task classifier.
            for i in range(self.opt.DATASET.NUM_CLASSES):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class[i]))
            log.close()
            return max(acc_for_each_class_auxi1.mean(), acc_for_each_class_auxi2.mean(), acc_for_each_class.mean())

    def build_optimizer(self):
        if self.opt.TRAIN.OPTIMIZER == 'SGD':  ## some params may not contribute the loss_all, thus they are not updated in the training process.
            self.optimizer = torch.optim.SGD([
                {'params': self.net.module.conv1.parameters(), 'name': 'pre-trained'},
                {'params': self.net.module.bn1.parameters(), 'name': 'pre-trained'},
                {'params': self.net.module.layer1.parameters(), 'name': 'pre-trained'},
                {'params': self.net.module.layer2.parameters(), 'name': 'pre-trained'},
                {'params': self.net.module.layer3.parameters(), 'name': 'pre-trained'},
                {'params': self.net.module.layer4.parameters(), 'name': 'pre-trained'},
                {'params': self.net.module.fc.parameters(), 'name': 'new-added'},
                {'params': self.net.module.fc_aux1.parameters(), 'name': 'new-added'},
                {'params': self.net.module.fc_aux2.parameters(), 'name': 'new-added'},
                {'params': self.net.module.fcdc.parameters(), 'name': 'new-added'}
            ],
                lr=self.opt.TRAIN.BASE_LR,
                momentum=self.opt.TRAIN.MOMENTUM,
                weight_decay=self.opt.TRAIN.WEIGHT_DECAY,
                nesterov=True)
        else:
            raise NotImplementedError
        print('Optimizer built')

    def update_lr(self):
        if self.opt.TRAIN.LR_SCHEDULE == 'inv':
            if self.opt.TRAIN.PROCESS_COUNTER == 'epoch':
                lr = self.opt.TRAIN.BASE_LR / pow((1 + self.opt.INV.ALPHA * self.epoch / self.opt.TRAIN.MAX_EPOCH), self.opt.INV.BETA)
            elif self.opt.TRAIN.PROCESS_COUNTER == 'iteration':
                lr = self.opt.TRAIN.BASE_LR / pow((1 + self.opt.INV.ALPHA * self.iters / (self.opt.TRAIN.MAX_EPOCH * self.iters_per_epoch)), self.opt.INV.BETA)
            else:
                raise NotImplementedError
        elif self.opt.TRAIN.LR_SCHEDULE == 'fix':
            lr = self.opt.TRAIN.BASE_LR
        else:
            raise NotImplementedError
        lr_pretrain = lr * 0.1
        print('the lr is: %3f' % (lr))
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'pre-trained':
                param_group['lr'] = lr_pretrain
            elif param_group['name'] == 'new-added':
                param_group['lr'] = lr
            elif param_group['name'] == 'fixed': ## Fix the lr as 0 can not fix the runing mean/var of the BN layer
                param_group['lr'] = 0

    def save_ckpt(self):
        log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
        log.write("      Best Acc so far: %3f" % (self.best_prec1))
        log.close()
        if self.opt.TRAIN.SAVING:
            save_path = self.opt.SAVE_DIR
            ckpt_resume = os.path.join(save_path, 'ckpt_%d.resume' % (self.loop))
            torch.save({'epoch': self.epoch,
                        'best_prec1': self.best_prec1,
                        'model_state_dict': self.net.state_dict()
                        }, ckpt_resume)