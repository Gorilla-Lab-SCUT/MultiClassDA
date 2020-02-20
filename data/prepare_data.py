import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config.config import cfg
import numpy as np
import cv2
# from data.folder_new import ImageFolder_new
# from data.Uniform_folder import ImageFolder_uniform
# from data.Uniform_sampler import UniformBatchSampler

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