import os.path
import numpy as np
from collections import namedtuple
import pdb
from PIL import Image
import torch.utils.data as data
import math
import torch


class Cityscapes(data.Dataset):
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]


    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]
    
    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

class SUNRGBD(data.Dataset):
    SUNRGBDClass = namedtuple('SUNRGBDClass', ['name', 'id', 'train_id', 'color'])
    classes = [
        SUNRGBDClass('void', 0, 255, (0, 0, 0)),
        SUNRGBDClass('wall', 1, 0, (119, 119, 119)),
        SUNRGBDClass('floor', 2, 1, (244, 243, 131)),
        SUNRGBDClass('cabinet', 3, 2, (137, 28, 157)),
        SUNRGBDClass('bed', 4, 3, (150, 255, 255)),
        SUNRGBDClass('chair', 5, 4, (54, 114, 113)),
        SUNRGBDClass('sofa', 6, 5, (0, 0, 176)),
        SUNRGBDClass('table', 7, 6, (255, 69, 0)),
        SUNRGBDClass('door', 8, 7, (87, 112, 255)),
        SUNRGBDClass('window', 9, 8, (0, 163, 33)),
        SUNRGBDClass('bookshelf', 10, 9, (255, 150, 255)),
        SUNRGBDClass('picture', 11, 10, (255, 180, 10)),
        SUNRGBDClass('counter', 12, 11, (101, 70, 86)),
        SUNRGBDClass('blinds', 13, 12, (38, 230, 0)),
        SUNRGBDClass('desk', 14, 13, (255, 120, 70)),
        SUNRGBDClass('shelves', 15, 14, (117, 41, 121)),
        SUNRGBDClass('curtain', 16, 15, (150, 255, 0)),
        SUNRGBDClass('dresser', 17, 16, (132, 0, 255)),
        SUNRGBDClass('pillow', 18, 17, (24, 209, 255)),
        SUNRGBDClass('mirror', 19, 18, (191, 130, 35)),
        SUNRGBDClass('floor mat', 20, 19, (219, 200, 109)),
        SUNRGBDClass('clothes', 21, 20, (154, 62, 86)),
        SUNRGBDClass('ceiling', 22, 21, (255, 190, 190)),
        SUNRGBDClass('books', 23, 22, (255, 0, 255)),
        SUNRGBDClass('fridge', 24, 23, (192, 79, 212)),
        SUNRGBDClass('tv', 25, 24, (152, 163, 55)),
        SUNRGBDClass('paper', 26, 25, (230, 230, 230)),
        SUNRGBDClass('towel', 27, 26, (53, 130, 64)),
        SUNRGBDClass('shower curtain', 28, 27, (155, 249, 152)),
        SUNRGBDClass('box', 29, 28, (87, 64, 34)),
        SUNRGBDClass('whiteboard', 30, 29, (214, 209, 175)),
        SUNRGBDClass('person', 31, 30, (170, 0, 59)),
        SUNRGBDClass('night stand', 32, 31, (255, 0, 0)),
        SUNRGBDClass('toilet', 33, 32, (193, 195, 234)),
        SUNRGBDClass('sink', 34, 33, (70, 72, 115)),
        SUNRGBDClass('lamp', 35, 34, (255, 255, 0)),
        SUNRGBDClass('bathtub', 36, 35, (52, 57, 131)),
        SUNRGBDClass('bag', 37, 36, (12, 83, 45)),
    ]
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 37
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

class ScanNet(data.Dataset):
    ScanNetClass = namedtuple('ScanNetClass', ['name', 'id', 'train_id', 'color'])
    classes = [
        ScanNetClass('void', 0, 255, (0, 0, 0)),
        ScanNetClass('wall', 1, 1, (174, 199, 232)),
        ScanNetClass('floor', 2, 2, (152, 223, 138)),
        ScanNetClass('cabinet', 3, 3, (31, 119, 180)),
        ScanNetClass('bed', 4, 4, (255, 187, 120)),
        ScanNetClass('chair', 5, 5, (188, 189, 34)),
        ScanNetClass('sofa', 6, 6, (140, 86, 75)),
        ScanNetClass('table', 7, 7, (255, 152, 150)),
        ScanNetClass('door', 8, 8, (214, 39, 40)),
        ScanNetClass('window', 9, 9, (197, 176, 213)),
        ScanNetClass('bookshelf', 10, 10, (148, 103, 189)),
        ScanNetClass('picture', 11, 11, (196, 156, 148)),
        ScanNetClass('counter', 12, 12, (23, 190, 207)),

        ScanNetClass('blinds', 13, 255, (178, 76, 76)),
        
        ScanNetClass('desk', 14, 13, (247, 182, 210)),
        ScanNetClass('shelves', 15, 255, (66, 188, 102)),
        ScanNetClass('curtain', 16, 14, (219, 219, 141)),
        ScanNetClass('dresser', 17, 255, (140, 57, 197)),
        ScanNetClass('pillow', 18, 255, (202, 185, 52)),
        ScanNetClass('mirror', 19, 255, (51, 176, 203)),
        ScanNetClass('floormat', 20, 255, (200, 54, 131)),
        ScanNetClass('clothes', 21, 255, (92, 193, 61)),
        ScanNetClass('ceiling', 22, 255, (78, 71, 183)),
        ScanNetClass('books', 23, 255, (172, 114, 82)),

        ScanNetClass('refrigerator', 24, 15, (255, 127, 14)),
        ScanNetClass('television', 25, 255, (91, 163, 138)),
        ScanNetClass('paper', 26, 255, (153, 98, 156)),
        ScanNetClass('towel', 27, 255, (140, 153, 101)),

        ScanNetClass('shower curtain', 28, 16, (158, 218, 229)),
        ScanNetClass('box', 29, 255, (100, 125, 154)),
        ScanNetClass('white board', 30, 255, (178, 127, 135)),
        ScanNetClass('person', 31, 255, (120, 185, 128)),
        ScanNetClass('night stand', 32, 255, (146, 111, 194)),

        ScanNetClass('toilet', 33, 17, (44, 160, 44)),
        ScanNetClass('sink', 34, 18, (112, 128, 144)),
        ScanNetClass('lamp', 35, 255, (96, 207, 209)),

        
        ScanNetClass('bathtub', 36, 19, (227, 119, 194)),
        ScanNetClass('bag', 37, 255, (213, 92, 176)),
        ScanNetClass('otherstructure', 38, 255, (94, 106, 211)),

        ScanNetClass('otherfurniture', 39, 20, (82, 84, 163)),
        ScanNetClass('otherprop', 40, 255, (100, 85, 144)),
    ]


    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]
    
    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 21
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

class NYUDv2(data.Dataset):
    NYUDv2Class = namedtuple('NYUDv2Class', ['name', 'id', 'train_id', 'color'])
    classes = [
        NYUDv2Class('void', 0, 255, (0, 0, 0)),
        NYUDv2Class('wall', 1, 0, (174, 199, 232)),
        NYUDv2Class('floor', 2, 1, (152, 223, 138)),
        NYUDv2Class('cabinet', 3, 2, (31, 119, 180)),
        NYUDv2Class('bed', 4, 3, (255, 187, 120)),
        NYUDv2Class('chair', 5, 4, (188, 189, 34)),
        NYUDv2Class('sofa', 6, 5, (140, 86, 75)),
        NYUDv2Class('table', 7, 6, (255, 152, 150)),
        NYUDv2Class('door', 8, 7, (214, 39, 40)),
        NYUDv2Class('window', 9, 8, (197, 176, 213)),
        NYUDv2Class('bookshelf', 10, 9, (148, 103, 189)),
        NYUDv2Class('picture', 11, 10, (196, 156, 148)),
        NYUDv2Class('counter', 12, 11, (23, 190, 207)),

        NYUDv2Class('blinds', 13, 12, (178, 76, 76)),
        
        NYUDv2Class('desk', 14, 13, (247, 182, 210)),
        NYUDv2Class('shelves', 15, 14, (66, 188, 102)),
        NYUDv2Class('curtain', 16, 15, (219, 219, 141)),
        NYUDv2Class('dresser', 17, 16, (140, 57, 197)),
        NYUDv2Class('pillow', 18, 17, (202, 185, 52)),
        NYUDv2Class('mirror', 19, 18, (51, 176, 203)),
        NYUDv2Class('floormat', 20, 19, (200, 54, 131)),
        NYUDv2Class('clothes', 21, 20, (92, 193, 61)),
        NYUDv2Class('ceiling', 22, 21, (78, 71, 183)),
        NYUDv2Class('books', 23, 22, (172, 114, 82)),

        NYUDv2Class('refrigerator', 24, 23, (255, 127, 14)),
        NYUDv2Class('television', 25, 24, (91, 163, 138)),
        NYUDv2Class('paper', 26, 25, (153, 98, 156)),
        NYUDv2Class('towel', 27, 26, (140, 153, 101)),

        NYUDv2Class('shower curtain', 28, 27, (158, 218, 229)),
        NYUDv2Class('box', 29, 28, (100, 125, 154)),
        NYUDv2Class('white board', 30, 29, (178, 127, 135)),
        NYUDv2Class('person', 31, 30, (120, 185, 128)),
        NYUDv2Class('night stand', 32, 31, (146, 111, 194)),

        NYUDv2Class('toilet', 33, 32, (44, 160, 44)),
        NYUDv2Class('sink', 34, 33, (112, 128, 144)),
        NYUDv2Class('lamp', 35, 34, (96, 207, 209)),

        
        NYUDv2Class('bathtub', 36, 35, (227, 119, 194)),
        NYUDv2Class('bag', 37, 36, (213, 92, 176)),
        NYUDv2Class('otherstructure', 38, 37, (94, 106, 211)),

        NYUDv2Class('otherfurniture', 39, 38, (82, 84, 163)),
        NYUDv2Class('otherprop', 40, 39, (100, 85, 144)),
    ]


    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]
    
    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 40
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

class Datasets(data.Dataset):

    def __init__(self, root, dataset, image_set='train', transform=None, missing_rate=0, count=1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set

        split_f = os.path.join(root, image_set+'.txt')
        if not os.path.exists(split_f):
            raise ValueError('Wrong image_set entered! Please use image_set="train"'
                             'or image_set="val" or image_set="test"')
        with open(os.path.join(split_f), "r") as f:
            modalities_path = [x.strip() for x in f.readlines()]

    
        ##########################################################
        
        if self.image_set == 'test' or self.image_set == 'val':
        #     missing_rate = 1.0
        #     masks = torch.ones(len(modalities_path), 2)
        #     sample_random = np.random.uniform(low=0, high=1.0, size=(len(modalities_path)))
        #     index = np.where(sample_random < missing_rate)
        #     modality_random = np.random.uniform(low=0, high=1.0, size=len(index[0]))
        #     modality1_index = np.where(modality_random <= 0.5)
        #     modality2_index = np.where(modality_random > 0.5)
        #     masks[:, 0][index[0][modality1_index]] = 0
        #     masks[:, 1][index[0][modality2_index]] = 0
        #     save_path = './testing_missing/%s/testing_missing_rate_0.5+0.5_5.csv' % (dataset)
        #     with open(save_path, 'w') as file:
        #         np.savetxt(save_path, masks.numpy(), fmt='%d', delimiter=',')

            if missing_rate == 0:
                self.masks = np.loadtxt('./testing_missing/%s/testing_missing_rate_0.csv' % (dataset), delimiter=',')
            else:
                self.masks = np.loadtxt('./testing_missing/%s/testing_missing_rate_%.1f+%.1f_%d.csv' % (dataset, missing_rate/2, missing_rate/2, count), delimiter=',')

        #############################################################
        

        self.rgbs = [x.split(' ')[0] for x in modalities_path]
        self.depths = [x.split(' ')[1] for x in modalities_path]
        self.labels = [x.split(' ')[2] for x in modalities_path]

    def __getitem__(self, index):
        rgb = Image.open(self.rgbs[index]).convert('RGB')
        depth = Image.open(self.depths[index])
        label = Image.open(self.labels[index])
        # img_name = os.path.basename(self.rgbs[index])

        if self.transform is not None:
            rgb, depth, label = self.transform(rgb, depth, label)
        if 'cityscapes' in self.root:
            cityscapes = Cityscapes()
            label = cityscapes.encode_target(label)
        elif 'sunrgbd' in self.root:
            sunrgbd = SUNRGBD()
            label = sunrgbd.encode_target(label)
        elif 'scannet' in self.root:
            scannet = ScanNet()
            label = scannet.encode_target(label)
        elif 'nyudv2' in self.root:
            nyudv2 = NYUDv2()
            label = nyudv2.encode_target(label)

        ###############################################################
        
        if self.image_set == 'test' or self.image_set == 'val':
            mask = self.masks[index]
            if mask[0] == 0:
                rgb = torch.zeros(3, rgb.shape[1], rgb.shape[2])
                # rgb = Image.fromarray(np.uint8(rgb))
            elif mask[1] == 0:
                depth = torch.zeros(3, depth.shape[1], depth.shape[2])
        else:
            mask = torch.ones(2)
        # mask = torch.ones(2)

        ###############################################################
        return rgb, depth, label, mask

    def __len__(self):
        return len(self.labels)
