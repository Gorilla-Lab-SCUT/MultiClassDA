import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config.config import cfg
import numpy as np
import cv2
import random
from data.folder_new import ImageFolder_Withpath
from data.folder_mec import ImageFolder_MEC

class Sampler(object):
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class UniformBatchSampler(Sampler):
    def __init__(self, per_category, category_index_list, imgs):

        self.per_category = per_category
        self.category_index_list = category_index_list
        self.imgs = imgs
        self.batch_size = per_category * len(category_index_list)
        self.batch_num = len(self.imgs) // self.batch_size
    def __iter__(self):
        for bat in range(self.batch_num):
            batch = []
            for i in range(len(self.category_index_list)):
                batch = batch + random.sample(self.category_index_list[i], self.per_category)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batch_num

########################################## function for MEC
def _random_affine_augmentation(x):
    M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0],
        [np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
    rows, cols = x.shape[1:3]
    dst = cv2.warpAffine(np.transpose(x.numpy(), [1, 2, 0]), M, (cols,rows))
    dst = np.transpose(dst, [2, 0, 1])
    return torch.from_numpy(dst)


def _gaussian_blur(x, sigma=0.1):
    ksize = int(sigma + 0.5) * 8 + 1
    dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
    return torch.from_numpy(dst)




############# To control the categorical weight of each batch.
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    # weight_per_class[-1] = weight_per_class[-1]  ########### adjust the cate-weight for unknown category.
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def _select_image_process(DATA_TRANSFORM_TYPE):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if DATA_TRANSFORM_TYPE == 'ours':
        transforms_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    elif DATA_TRANSFORM_TYPE == 'longs':
        transforms_train = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_test = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    elif DATA_TRANSFORM_TYPE == 'simple':
        transforms_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        raise NotImplementedError

    return transforms_train, transforms_test

def _select_image_process_mec(DATA_TRANSFORM_TYPE):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if DATA_TRANSFORM_TYPE == 'ours':
        transforms_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: _random_affine_augmentation(x)),
                transforms.Lambda(lambda x: _gaussian_blur(x)),
                normalize,
            ])
    elif DATA_TRANSFORM_TYPE == 'longs':
        transforms_train = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: _random_affine_augmentation(x)),
                transforms.Lambda(lambda x: _gaussian_blur(x)),
                normalize,
            ])
    elif DATA_TRANSFORM_TYPE == 'simple':
        transforms_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: _random_affine_augmentation(x)),
                transforms.Lambda(lambda x: _gaussian_blur(x)),
                normalize,
            ])
    else:
        raise NotImplementedError

    return transforms_train

def generate_dataloader():
    dataloaders = {}
    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    val = cfg.DATASET.VAL_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    dataroot_V = os.path.join(cfg.DATASET.DATAROOT, val)

    if not os.path.isdir(dataroot_S):
        raise ValueError('Invalid path of source data!!!')

    transforms_train, transforms_test = _select_image_process(cfg.DATA_TRANSFORM.TYPE)

    ############ dataloader #############################
    source_train_dataset = datasets.ImageFolder(
        dataroot_S,
        transforms_train
    )
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=cfg.TRAIN.SOURCE_BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=False
    )

    target_train_dataset = datasets.ImageFolder(
        dataroot_T,
        transforms_train
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=cfg.TRAIN.TARGET_BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=False
    )

    target_test_dataset = datasets.ImageFolder(
        dataroot_V,
        transforms_test
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=False
    )

    dataloaders['source'] = source_train_loader
    dataloaders['target'] = target_train_loader
    dataloaders['test'] = target_test_loader

    return dataloaders


def generate_dataloader_sc():
    dataloaders = {}
    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    val = cfg.DATASET.VAL_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    dataroot_V = os.path.join(cfg.DATASET.DATAROOT, val)

    if not os.path.isdir(dataroot_S):
        raise ValueError('Invalid path of source data!!!')

    transforms_train, transforms_test = _select_image_process(cfg.DATA_TRANSFORM.TYPE)
    transforms_mec = _select_image_process_mec(cfg.DATA_TRANSFORM.TYPE)
    ############ dataloader #############################
    source_train_dataset = datasets.ImageFolder(
        dataroot_S,
        transforms_train
    )
    if cfg.STRENGTHEN.DATALOAD == 'normal':
        source_train_loader = torch.utils.data.DataLoader(
            source_train_dataset, batch_size=cfg.STRENGTHEN.PERCATE * cfg.DATASET.NUM_CLASSES, shuffle=True,
            drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=False
        )
    elif cfg.STRENGTHEN.DATALOAD == 'hard':
        uniformbatchsampler = UniformBatchSampler(cfg.STRENGTHEN.PERCATE, source_train_dataset.category_index_list, source_train_dataset.imgs)
        source_train_loader = torch.utils.data.DataLoader(
            source_train_dataset, num_workers=cfg.NUM_WORKERS, pin_memory=True, batch_sampler=uniformbatchsampler
        )
    elif cfg.STRENGTHEN.DATALOAD == 'soft':
        weights = make_weights_for_balanced_classes(source_train_dataset.imgs, len(source_train_dataset.classes))
        weights = torch.DoubleTensor(weights)
        sampler_s = torch.utils.data.sampler.WeightedRandomSampler(weights, len(
            weights))  #### sample instance uniformly for each category
        source_train_loader = torch.utils.data.DataLoader(
            source_train_dataset, batch_size=cfg.STRENGTHEN.PERCATE * cfg.DATASET.NUM_CLASSES, shuffle=False,
            drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=sampler_s
        )
    else:
        raise NotImplementedError

    target_train_dataset = ImageFolder_MEC(
        dataroot_T,
        transforms_train,
        transforms_mec
    )
    # target_train_loader = torch.utils.data.DataLoader(
    #     target_train_dataset, batch_size=cfg.TRAIN.TARGET_BATCH_SIZE, shuffle=True,
    #     drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=False
    # )

    target_test_dataset = datasets.ImageFolder(
        dataroot_V,
        transforms_test
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=False
    )

    source_cluster_dataset = ImageFolder_Withpath(
        dataroot_S,
        transforms_test
    )
    target_cluster_dataset = ImageFolder_Withpath(
        dataroot_T,
        transforms_test
    )
    source_cluster_loader = torch.utils.data.DataLoader(
        source_cluster_dataset,
        batch_size=1000, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=False
    )
    target_cluster_loader = torch.utils.data.DataLoader(
        target_cluster_dataset,
        batch_size=1000, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=False
    )

    dataloaders['source'] = source_train_loader
    dataloaders['target'] = 0
    dataloaders['target_train_dataset'] = target_train_dataset
    dataloaders['test'] = target_test_loader
    dataloaders['source_cluster'] = source_cluster_loader
    dataloaders['target_cluster'] = target_cluster_loader

    return dataloaders

def generate_dataloader_open():
    dataloaders = {}
    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    val = cfg.DATASET.VAL_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    dataroot_V = os.path.join(cfg.DATASET.DATAROOT, val)

    if not os.path.isdir(dataroot_S):
        raise ValueError('Invalid path of source data!!!')

    transforms_train, transforms_test = _select_image_process(cfg.DATA_TRANSFORM.TYPE)

    ############ dataloader #############################
    source_train_dataset = datasets.ImageFolder(
        dataroot_S,
        transforms_train
    )
    weights = make_weights_for_balanced_classes(source_train_dataset.imgs, len(source_train_dataset.classes),
                                                cfg.OPEN.WEIGHT_UNK)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(
        weights))  #### sample instance uniformly for each category

    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=cfg.TRAIN.SOURCE_BATCH_SIZE, shuffle=False,
        drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=False, sampler=sampler
    )

    target_train_dataset = datasets.ImageFolder(
        dataroot_T,
        transforms_train
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=cfg.TRAIN.TARGET_BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=False
    )

    target_test_dataset = datasets.ImageFolder(
        dataroot_V,
        transforms_test
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=False
    )

    dataloaders['source'] = source_train_loader
    dataloaders['target'] = target_train_loader
    dataloaders['test'] = target_test_loader

    return dataloaders

    ########## other dataloader options to be added ##########
    # if args.uniform_type_s == 'hard':
    #     uniformbatchsampler = UniformBatchSampler(args.per_category, source_train_dataset.category_index_list, source_train_dataset.imgs)
    #     source_train_loader = torch.utils.data.DataLoader(
    #         source_train_dataset, num_workers=args.workers, pin_memory=True, batch_sampler=uniformbatchsampler
    #     )
    # elif args.uniform_type_s == 'soft':
    #     weights = make_weights_for_balanced_classes(source_train_dataset.imgs, len(source_train_dataset.classes))
    #     weights = torch.DoubleTensor(weights)
    #     sampler_s = torch.utils.data.sampler.WeightedRandomSampler(weights, len(
    #         weights))  #### sample instance uniformly for each category
    #     source_train_loader = torch.utils.data.DataLoader(
    #         source_train_dataset, batch_size=args.per_category * args.num_classes, shuffle=False,
    #         drop_last=True, num_workers=args.workers, pin_memory=True, sampler=sampler_s
    #     )
    # else:
    #     source_train_loader = torch.utils.data.DataLoader(
    #         source_train_dataset, batch_size=args.batch_size, shuffle=True,
    #         drop_last=True, num_workers=args.workers, pin_memory=True
    #     )