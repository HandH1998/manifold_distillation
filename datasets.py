# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# 2022.10.14-Changed for building manifold kd
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#

import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import Dataset
import torch
import numpy as np
import PIL.Image


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    if label_file_path is None:
        label_file_path = 'voc12/cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list
    # return [cls_labels_dict[img_name] for img_name in img_name_list ]


class VOC12Dataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, label_file_path, train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12DatasetMS(Dataset):
    def __init__(self, img_name_list_path, voc12_root, label_file_path, scales, train=True, transform=None, gen_attn=False, unit=1):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)

class COCOClsDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.coco_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_name_list)

class COCOClsDatasetMS(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, scales, train=True, transform=None, gen_attn=False, unit=1):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.unit = unit
        self.scales = scales
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.coco_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])

            # msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


def build_dataset(is_train, args, gen_attn=False):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'VOC12':
        dataset = VOC12Dataset(img_name_list_path=args.img_list, voc12_root=args.data_path, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif args.data_set == 'VOC12MS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, label_file_path=args.label_file_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif args.data_set == 'COCO':
        dataset = COCOClsDataset(img_name_list_path=args.img_list, coco_root=args.data_path, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80
    elif args.data_set == 'COCOMS':
        dataset = COCOClsDatasetMS(img_name_list_path=args.img_list, coco_root=args.data_path, scales=tuple(args.scales), label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
