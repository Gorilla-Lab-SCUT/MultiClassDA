import torch
import torch.nn as nn
import os
import math
import time
from utils.utils import to_cuda, accuracy_for_each_class, accuracy, AverageMeter, process_one_values
from config.config import cfg
import torch.nn.functional as F
from models.loss_utils import TargetDiscrimLoss, ConcatenatedCELoss, MinEntropyConsensusLoss
from .base_solver import BaseSolver
import ipdb
from data.prepare_data import UniformBatchSampler
import numpy as np
from spherecluster import SphericalKMeans

class SymmNetsV2SolverSC(BaseSolver):
    def __init__(self, net, dataloaders, **kwargs):
        super(SymmNetsV2SolverSC, self).__init__(net, dataloaders, **kwargs)
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.TargetDiscrimLoss = TargetDiscrimLoss(num_classes=self.num_classes).cuda()
        self.ConcatenatedCELoss = ConcatenatedCELoss(num_classes=self.num_classes).cuda()
        self.feature_extractor = self.net['feature_extractor']
        self.classifier = self.net['classifier']
        self.target_train_dataset = dataloaders['target_train_dataset']
        self.init_data(dataloaders)
        self.init_imgs = self.target_train_dataset.imgs[:] #### ??
        self.MECLoss = MinEntropyConsensusLoss(num_classes=self.num_classes)

        self.moving_feature_centeriod_s = torch.zeros(self.num_classes, 2048).cuda()  # .normal_(0, 1).cuda()
        self.moving_feature_centeriod_t = torch.zeros(self.num_classes, 2048).cuda()  # .normal_(0, 1).cuda()
        self.moving_feature_centeriod_st = torch.zeros(self.num_classes, 2048).cuda()

        if cfg.STRENGTHEN.DATALOAD == 'normal':
            self.target_train_dataset.imgs = self.init_imgs[:]
            self.train_data['target']['loader'] = torch.utils.data.DataLoader(
            self.target_train_dataset, batch_size=cfg.STRENGTHEN.PERCATE * self.num_classes, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=None)
        else:
            self.clustering_lables_with_path = self.download_feature_and_clustering(self.train_data['source_cluster']['loader'],
                                            self.train_data['target_cluster']['loader'],self.feature_extractor)
            category_index_list, imgs = self.generate_category_index_list_imgs(self.clustering_lables_with_path, self.target_train_dataset)
            min_num_cate = cfg.STRENGTHEN.PERCATE  ## just a large number
            for i in range(len(category_index_list)):
                list_len = len(category_index_list[i])
                if min_num_cate > list_len:
                    min_num_cate = list_len
            if min_num_cate < cfg.STRENGTHEN.PERCATE:  ### in case of some target category has few samples, we return to the normal dataloader
                self.target_train_dataset.imgs = self.init_imgs[:]
                self.train_data['target']['loader'] = torch.utils.data.DataLoader(
                    self.target_train_dataset, batch_size=cfg.STRENGTHEN.PERCATE * self.num_classes, shuffle=True,
                    num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=None)
            else:
                if cfg.STRENGTHEN.DATALOAD == 'hard':
                    self.target_train_dataset.imgs = self.init_imgs[:]
                    uniformbatchsampler = UniformBatchSampler(cfg.STRENGTHEN.PERCATE, category_index_list, imgs)
                    self.train_data['target']['loader'] = torch.utils.data.DataLoader(self.target_train_dataset,
                                                                      num_workers=cfg.NUM_WORKERS, pin_memory=True,
                                                                      batch_sampler=uniformbatchsampler)
                elif cfg.STRENGTHEN.DATALOAD == 'soft':
                    self.target_train_dataset.imgs = imgs  ################ udpate the image lists
                    weights = self.make_weights_for_balanced_classes(self.target_train_dataset.imgs, self.num_classes)
                    weights = torch.DoubleTensor(weights)
                    sampler_t = torch.utils.data.sampler.WeightedRandomSampler(weights, len(
                        weights))  #### sample instance uniformly for each category
                    self.train_data['target']['loader'] = torch.utils.data.DataLoader(
                        self.target_train_dataset, batch_size=cfg.STRENGTHEN.PERCATE * self.num_classes, shuffle=False,
                        drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=sampler_t
                    )
                else:
                    raise NotImplementedError

        if cfg.RESUME != '':
            resume_dict = torch.load(cfg.RESUME)
            self.net['feature_extractor'].load_state_dict(resume_dict['feature_extractor_state_dict'])
            self.net['classifier'].load_state_dict(resume_dict['classifier_state_dict'])
            self.best_prec1 = resume_dict['best_prec1']
            self.epoch = resume_dict['epoch']

    def generate_category_index_list_imgs(self, clusering_labels_for_path, train_t_dataset):
        images = []
        for i in range(len(train_t_dataset.imgs)):
            path = train_t_dataset.imgs[i][0]
            target = clusering_labels_for_path[path]
            item = (path, target)
            images.append(item)
        category_index_list = []
        for i in range(self.num_classes):
            list_temp = []
            for j in range(len(images)):
                if i == images[j][1]:
                    list_temp.append(j)
            category_index_list.append(list_temp)

        return category_index_list, images

    def make_weights_for_balanced_classes(self, images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        # weight_per_class[-1] = weight_per_class[-1]  ########### adjust the cate-weight for unknown category.
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight

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


    def solve(self):
        stop = False
        while not stop:
            stop = self.complete_training()
            self.update_network()
            acc = self.test()
            if self.epoch % cfg.STRENGTHEN.CLUSTER_FREQ == 0 and not self.epoch == 0:
                if not cfg.STRENGTHEN.DATALOAD == 'normal':
                    self.clustering_lables_with_path = self.download_feature_and_clustering(
                        self.train_data['source_cluster']['loader'],
                        self.train_data['target_cluster']['loader'], self.feature_extractor)
                    category_index_list, imgs = self.generate_category_index_list_imgs(self.clustering_lables_with_path,
                                                                                       self.target_train_dataset)
                    min_num_cate = cfg.STRENGTHEN.PERCATE  ## just a large number
                    for i in range(len(category_index_list)):
                        list_len = len(category_index_list[i])
                        if min_num_cate > list_len:
                            min_num_cate = list_len
                    if min_num_cate < cfg.STRENGTHEN.PERCATE:  ### in case of some target category has few samples, we return to the normal dataloader
                        self.target_train_dataset.imgs = self.init_imgs[:]
                        self.train_data['target']['loader'] = torch.utils.data.DataLoader(
                            self.target_train_dataset, batch_size=cfg.STRENGTHEN.PERCATE * self.num_classes, shuffle=True,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=None)
                    else:
                        if cfg.STRENGTHEN.DATALOAD == 'hard':
                            self.target_train_dataset.imgs = self.init_imgs[:]
                            uniformbatchsampler = UniformBatchSampler(cfg.STRENGTHEN.PERCATE, category_index_list, imgs)
                            self.train_data['target']['loader'] = torch.utils.data.DataLoader(self.target_train_dataset,
                                                                                              num_workers=cfg.NUM_WORKERS,
                                                                                              pin_memory=True,
                                                                                              batch_sampler=uniformbatchsampler)
                        elif cfg.STRENGTHEN.DATALOAD == 'soft':
                            self.target_train_dataset.imgs = imgs  ################ udpate the image lists
                            weights = self.make_weights_for_balanced_classes(self.target_train_dataset.imgs, self.num_classes)
                            weights = torch.DoubleTensor(weights)
                            sampler_t = torch.utils.data.sampler.WeightedRandomSampler(weights, len(
                                weights))  #### sample instance uniformly for each category
                            self.train_data['target']['loader'] = torch.utils.data.DataLoader(
                                self.target_train_dataset, batch_size=cfg.STRENGTHEN.PERCATE * self.num_classes, shuffle=False,
                                drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=sampler_t
                            )
                        else:
                            raise NotImplementedError
            if acc > self.best_prec1:
                self.best_prec1 = acc
                self.save_ckpt()
            self.epoch += 1


    def update_network(self, **kwargs):
        stop = False
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        self.iters_per_epoch = max(len(self.train_data['target']['loader']), len(self.train_data['source']['loader']))
        iters_counter_within_epoch = 0
        data_time = AverageMeter()
        batch_time = AverageMeter()
        classifier_loss = AverageMeter()
        feature_extractor_loss = AverageMeter()
        prec1_fs = AverageMeter()
        prec1_ft = AverageMeter()
        self.feature_extractor.train()
        self.classifier.train()
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
            target_data, target_data_mec, target_gt_NOTUSE, target_path = self.get_samples('target')
            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            target_data = to_cuda(target_data)
            target_data_mec = to_cuda(target_data_mec)
            cluster_gt = torch.zeros(target_gt_NOTUSE.size(), dtype=torch.int64)
            for i in range(len(target_path)):
                cluster_gt[i] = torch.as_tensor(self.clustering_lables_with_path[target_path[i]].astype(np.float64))
            cluster_gt = to_cuda(cluster_gt)
            data_time.update(time.time() - end)

            feature_source = self.feature_extractor(source_data)
            output_source = self.classifier(feature_source)
            feature_target = self.feature_extractor(target_data)
            output_target = self.classifier(feature_target)
            feature_target_mec = self.feature_extractor(target_data_mec)
            output_target_mec = self.classifier(feature_target_mec)
            loss_mec = 0.5 * self.MECLoss(output_target[:, :self.num_classes], output_target_mec[:, :self.num_classes]) + \
                       0.5 * self.MECLoss(output_target[:, self.num_classes:], output_target_mec[:, self.num_classes:])
            fea_dim = feature_source.size(1)
            ######################################################################## loss of the prototypical net
            category_center_st = torch.zeros(self.num_classes, fea_dim).cuda()
            category_count_st = torch.zeros(self.num_classes, 1).cuda()
            category_center_s = torch.zeros(self.num_classes, fea_dim).cuda()
            category_count_s = torch.zeros(self.num_classes, 1).cuda()
            for j in range(source_data.size(0)):
                category_center_s[source_gt[j].item()] = category_center_s[source_gt[j].item()] + feature_source[j]
                category_count_s[source_gt[j].item()] = category_count_s[source_gt[j].item()] + 1
                category_center_st[source_gt[j].item()] = category_center_st[source_gt[j].item()] + feature_source[j]
                category_count_st[source_gt[j].item()] = category_count_st[source_gt[j].item()] + 1
            self.moving_feature_centeriod_s = self.moving_feature_centeriod_s.data.clone().cuda()
            mask_s = ((self.moving_feature_centeriod_s.data != 0).sum(1, keepdim=True) != 0).float() * 0.7
            mask_s[category_count_s == 0] = 1.0
            category_count_s[category_count_s == 0] = 1.0
            self.moving_feature_centeriod_s = mask_s * self.moving_feature_centeriod_s + (1-mask_s) * (category_center_s / category_count_s)

            category_center_t = torch.zeros(self.num_classes, fea_dim).cuda()
            category_count_t = torch.zeros(self.num_classes, 1).cuda()
            for j in range(target_data.size(0)):
                category_center_t[cluster_gt[j].item()] = category_center_t[cluster_gt[j].item()] + feature_target[j]
                category_count_t[cluster_gt[j].item()] = category_count_t[cluster_gt[j].item()] + 1
                category_center_st[cluster_gt[j].item()] = category_center_st[cluster_gt[j].item()] + feature_target[j]
                category_count_st[cluster_gt[j].item()] = category_count_st[cluster_gt[j].item()] + 1
            self.moving_feature_centeriod_t = self.moving_feature_centeriod_t.data.clone().cuda()
            self.moving_feature_centeriod_st = self.moving_feature_centeriod_st.data.clone().cuda()
            mask_t = ((self.moving_feature_centeriod_t.data != 0).sum(1, keepdim=True) != 0).float() * 0.7
            mask_t[category_count_t == 0] = 1.0
            category_count_t[category_count_t == 0] = 1.0
            self.moving_feature_centeriod_t = mask_t * self.moving_feature_centeriod_t + (1-mask_t) * (category_center_t / category_count_t)
            mask_st = ((self.moving_feature_centeriod_st.data != 0).sum(1, keepdim=True) != 0).float() * 0.7
            mask_st[category_count_st == 0] = 1.0
            category_count_st[category_count_st == 0] = 1.0
            self.moving_feature_centeriod_st = mask_st * self.moving_feature_centeriod_st + (1-mask_st) * (category_center_st / category_count_st)

            # dis_matrix_s_t = torch.norm(moving_feature_centeriod_s.unsqueeze(0) - moving_feature_centeriod_t.unsqueeze(1), p=2, dim=2)
            # dis_matrix_s_st = torch.norm(moving_feature_centeriod_s.unsqueeze(0) - moving_feature_centeriod_st.unsqueeze(1), p=2, dim=2)
            # dis_matrix_st_t = torch.norm(moving_feature_centeriod_st.unsqueeze(0) - moving_feature_centeriod_t.unsqueeze(1), p=2, dim=2)
            # same_category_mask = torch.eye(args.num_classes, dtype=torch.uint8).cuda()
            # loss_center_pair_intra = torch.mean(torch.masked_select(dis_matrix_s_t, same_category_mask)) + \
            #                          torch.mean(torch.masked_select(dis_matrix_s_st, same_category_mask)) + \
            #                          torch.mean(torch.masked_select(dis_matrix_st_t, same_category_mask))

            dis_matrix_s = - torch.norm(feature_source.unsqueeze(1) - self.moving_feature_centeriod_s.unsqueeze(0), p=2, dim=2)
            dis_matrix_t = - torch.norm(feature_target.unsqueeze(1) - self.moving_feature_centeriod_t.unsqueeze(0), p=2, dim=2)
            loss_proto_s = self.CELoss(dis_matrix_s, source_gt)
            loss_proto_t = self.CELoss(dis_matrix_t, cluster_gt)



            loss_task_fs = self.CELoss(output_source[:,:self.num_classes], source_gt)
            loss_task_ft = self.CELoss(output_source[:,self.num_classes:], source_gt)
            loss_discrim_source = self.CELoss(output_source, source_gt)
            loss_discrim_target = self.TargetDiscrimLoss(output_target)
            loss_summary_classifier = loss_task_fs + loss_task_ft + loss_discrim_source + loss_discrim_target

            source_gt_for_ft_in_fst = source_gt + self.num_classes
            loss_confusion_source = 0.5 * self.CELoss(output_source, source_gt) + 0.5 * self.CELoss(output_source, source_gt_for_ft_in_fst)
            loss_confusion_target = self.ConcatenatedCELoss(output_target)
            loss_summary_feature_extractor = loss_confusion_source + lam * (loss_confusion_target + loss_mec + loss_proto_t)


            self.optimizer_classifier.zero_grad()
            loss_summary_classifier.backward(retain_graph=True)
            self.optimizer_classifier.step()

            self.optimizer_feature_extractor.zero_grad()
            loss_summary_feature_extractor.backward()
            self.optimizer_feature_extractor.step()

            classifier_loss.update(loss_summary_classifier, source_data.size()[0])
            feature_extractor_loss.update(loss_summary_feature_extractor, source_data.size()[0])
            prec1_fs.update(accuracy(output_source[:, :self.num_classes], source_gt), source_data.size()[0])
            prec1_ft.update(accuracy(output_source[:, self.num_classes:], source_gt), source_data.size()[0])

            print("  Train:epoch: %d:[%d/%d], LossCla: %3f, LossFeat: %3f, AccFs: %3f, AccFt: %3f" % \
                  (self.epoch, iters_counter_within_epoch, self.iters_per_epoch, classifier_loss.avg, feature_extractor_loss.avg, prec1_fs.avg, prec1_ft.avg))

            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            iters_counter_within_epoch += 1
            if iters_counter_within_epoch >= self.iters_per_epoch:
                log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
                log.write("\n")
                log.write("  Train:epoch: %d:[%d/%d], LossCla: %3f, LossFeat: %3f, AccFs: %3f, AccFt: %3f" % \
                  (self.epoch, iters_counter_within_epoch, self.iters_per_epoch, classifier_loss.avg, feature_extractor_loss.avg, prec1_fs.avg, prec1_ft.avg))
                log.close()
                stop = True


    def test(self):
        self.feature_extractor.eval()
        self.classifier.eval()
        prec1_fs = AverageMeter()
        prec1_ft = AverageMeter()
        counter_all_fs = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_all_ft = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_acc_fs = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_acc_ft = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)

        for i, (input, target) in enumerate(self.test_data['loader']):
            input, target = to_cuda(input), to_cuda(target)
            with torch.no_grad():
                feature_test = self.feature_extractor(input)
                output_test = self.classifier(feature_test)


            if self.opt.EVAL_METRIC == 'accu':
                prec1_fs_iter = accuracy(output_test[:, :self.num_classes], target)
                prec1_ft_iter = accuracy(output_test[:, self.num_classes:], target)
                prec1_fs.update(prec1_fs_iter, input.size(0))
                prec1_ft.update(prec1_ft_iter, input.size(0))
                if i % self.opt.PRINT_STEP == 0:
                    print("  Test:epoch: %d:[%d/%d], AccFs: %3f, AccFt: %3f" % \
                          (self.epoch, i, len(self.test_data['loader']), prec1_fs.avg, prec1_ft.avg))
            elif self.opt.EVAL_METRIC == 'accu_mean':
                prec1_ft_iter = accuracy(output_test[:, self.num_classes:], target)
                prec1_ft.update(prec1_ft_iter, input.size(0))
                counter_all_fs, counter_acc_fs = accuracy_for_each_class(output_test[:, :self.num_classes], target, counter_all_fs, counter_acc_fs)
                counter_all_ft, counter_acc_ft = accuracy_for_each_class(output_test[:, self.num_classes:], target, counter_all_ft, counter_acc_ft)
                if i % self.opt.PRINT_STEP == 0:
                    print("  Test:epoch: %d:[%d/%d], Task: %3f" % \
                          (self.epoch, i, len(self.test_data['loader']), prec1_ft.avg))
            else:
                raise NotImplementedError
        acc_for_each_class_fs = counter_acc_fs / counter_all_fs
        acc_for_each_class_ft = counter_acc_ft / counter_all_ft
        log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
        log.write("\n")
        if self.opt.EVAL_METRIC == 'accu':
            log.write(
                "                                                          Test:epoch: %d, AccFs: %3f, AccFt: %3f" % \
                (self.epoch, prec1_fs.avg, prec1_ft.avg))
            log.close()
            return max(prec1_fs.avg, prec1_ft.avg)
        elif self.opt.EVAL_METRIC == 'accu_mean':
            log.write(
                "                                            Test:epoch: %d, AccFs: %3f, AccFt: %3f" % \
                (self.epoch,acc_for_each_class_fs.mean(), acc_for_each_class_ft.mean()))
            log.write("\nClass-wise Acc of Ft:")  ## based on the task classifier.
            for i in range(self.opt.DATASET.NUM_CLASSES):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class_ft[i]))
            log.close()
            return max(acc_for_each_class_ft.mean(), acc_for_each_class_fs.mean())

    def download_feature_and_clustering(self, train_loader, val_loader, model):
        model.eval()
        image_paths = []
        GT_labels = []
        source_feature_list = []
        for i in range(self.num_classes):
            source_feature_list.append([])  ######### each for one categoty
        for i, (input, target, img_path) in enumerate(train_loader):
            print('soruce center calculation', i)
            with torch.no_grad():
                feature_source = model(input)
            feature_source = feature_source.cpu()
            batchsize = feature_source.size(0)
            for j in range(batchsize):
                img_label = target[j]
                source_feature_list[img_label].append(feature_source[j].view(1, feature_source.size(1)))

        target_feature_list = []
        for i, (input, target, img_path) in enumerate(val_loader):
            print('target feature calculation', i)
            with torch.no_grad():
                feature_target = model(input)
            batchsize = feature_target.size(0)
            feature_target = feature_target.cpu()
            for j in range(batchsize):
                GT_labels.append(target[j].item())
                image_paths.append(img_path[j])
                target_feature_list.append(feature_target[j].view(1, feature_target.size(1)))


        feature_matrix = torch.cat(target_feature_list, dim=0)
        feature_matrix = F.normalize(feature_matrix, dim=1, p=2)
        feature_matrix = feature_matrix.numpy()
        ########################################### calculte source category center
        for i in range(self.num_classes):
            source_feature_list[i] = torch.cat(source_feature_list[i], dim=0)  ########## K * [num * dim]
            source_feature_list[i] = F.normalize(source_feature_list[i].mean(0), dim=0, p=2)
            source_feature_list[i] = source_feature_list[i].numpy()
        source_feature_array = np.array(source_feature_list)
        print('use the original cnn features to play cluster')

        kmeans = SphericalKMeans(n_clusters=self.num_classes, random_state=0, init=source_feature_array,
                                 max_iter=500).fit(feature_matrix)


        Ind = kmeans.labels_
        print(Ind)
        print(GT_labels)
        gt_label_array = np.array(GT_labels)
        acc_count = torch.zeros(self.num_classes)
        all_count = torch.zeros(self.num_classes)
        for i in range(len(gt_label_array)):
            all_count[gt_label_array[i]] += 1
            if gt_label_array[i] == Ind[i]:
                acc_count[gt_label_array[i]] += 1

        acc_for_each_class1 = acc_count / all_count
        acc_cluster_label = sum(gt_label_array == Ind) / gt_label_array.shape[0]
        corresponding_labels = []
        for i in range(len(Ind)):
            corresponding_labels.append(Ind[i])

        clustering_label_for_path = {image_paths[i]: corresponding_labels[i] for i in range(len(corresponding_labels))}
        # NMI_value = NMI_calculation(GT_labels, corresponding_labels)
        log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
        if self.opt.EVAL_METRIC == 'accu_mean':
            log.write("\nAcc for each class1: ")
            for i in range(self.num_classes):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class1[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class1[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class1[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class1[i]))
            log.write("Avg. over all classes: %3f" % acc_for_each_class1.mean())
        log.write("   Avg. over all sample: %3f" % acc_cluster_label)
        log.close()

        return clustering_label_for_path

    def build_optimizer(self):
        if self.opt.TRAIN.OPTIMIZER == 'SGD':  ## some params may not contribute the loss_all, thus they are not updated in the training process.
            self.optimizer_feature_extractor = torch.optim.SGD([
                {'params': self.net['feature_extractor'].module.conv1.parameters(), 'name': 'fixed'},
                {'params': self.net['feature_extractor'].module.bn1.parameters(), 'name': 'fixed'},
                {'params': self.net['feature_extractor'].module.layer1.parameters(), 'name': 'fixed'},
                {'params': self.net['feature_extractor'].module.layer2.parameters(), 'name': 'pre-trained'},
                {'params': self.net['feature_extractor'].module.layer3.parameters(), 'name': 'pre-trained'},
                {'params': self.net['feature_extractor'].module.layer4.parameters(), 'name': 'pre-trained'},
            ],
                lr=self.opt.TRAIN.BASE_LR,
                momentum=self.opt.TRAIN.MOMENTUM,
                weight_decay=self.opt.TRAIN.WEIGHT_DECAY,
                nesterov=True)

            self.optimizer_classifier = torch.optim.SGD([
                {'params': self.net['classifier'].parameters(), 'name': 'new-added'},
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
        for param_group in self.optimizer_feature_extractor.param_groups:
            if param_group['name'] == 'pre-trained':
                param_group['lr'] = lr_pretrain
            elif param_group['name'] == 'new-added':
                param_group['lr'] = lr
            elif param_group['name'] == 'fixed': ## Fix the lr as 0 can not fix the runing mean/var of the BN layer
                param_group['lr'] = 0

        for param_group in self.optimizer_classifier.param_groups:
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
                        'feature_extractor_state_dict': self.net['feature_extractor'].state_dict(),
                        'classifier_state_dict': self.net['classifier'].state_dict()
                        }, ckpt_resume)
