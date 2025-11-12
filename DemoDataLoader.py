# coding = utf-8
import random

import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
import os
from torch.utils import data
import numpy as np
import sys
import time

from transform import MedicalTransformCompose
from utils import get_absolute_project_dir




class DemoDataLoader(data.Dataset):
    def __init__(self, type="train", transform=None, train_file = "", valid_file = ""):
        """
        :param type: 数据类别：包括train和valid
        :param data_aug: 是否采用数据增强方法
        """
        if sys.platform.startswith('linux'):
            # linux 代码
            self.root_dir = None
        else:
            self.root_dir = ""
        self.project_dir = os.path.join(get_absolute_project_dir(), "")
        self.valid_file = valid_file
        self.train_file = train_file

        (train_ids, valid_ids) = self.fetch_train_valid_ids()

        if type == "train":
             (self.data_map, self.idx_map, self.sclc_map, self.nsclc_map) = self.fetch_data_map(ids=train_ids)
        else:
            (self.data_map, self.idx_map, self.sclc_map, self.nsclc_map) = self.fetch_data_map(ids=valid_ids)


        self.transform = transform



    def __getitem__(self, item):
        start = time.time()
        (case_id,file_index) = self.calcuate_file_from_index(index=item)
        ct_file_name = os.path.join(self.root_dir, "{}_CT.npy".format(case_id))
        ct_file = np.load(ct_file_name)
        ct_data = ct_file[file_index]
        #ct_data = (ct_data - np.min(ct_data)) / (np.max(ct_data) - np.min(ct_data))

        label_file_name = os.path.join(self.root_dir, "{}_Label.py.npy".format(case_id))
        label_file = np.load(label_file_name)
        label_data = label_file[file_index]


        image = ct_data.astype("float32")

        data = {"image":image, "label":label_data}
        if self.transform is not None:
            data = self.transform(data)

        image = data["image"]
        image = image.astype(np.float32)

        ct_data = image
        label_data = data["label"]

        ct_data = ct_data.reshape(1, ct_data.shape[0], ct_data.shape[1])

        if str(case_id).startswith("B"):
            cls_label = 1
        else:
            cls_label = 0

        # 选择用于对比学习的同样样本的不同切片，不同样本的不同切片
        if str(case_id).startswith('B'):
                slice_cnt = self.sclc_map[case_id]
                if slice_cnt == 1:
                    aug_same_slice_index = 0
                else:
                  aug_same_slice_index = random.randint(0, slice_cnt - 1)
                  while (aug_same_slice_index == file_index):
                    aug_same_slice_index = random.randint(0, slice_cnt - 1)
                aug_differ_id = list(self.nsclc_map.keys())[random.randint(0, len(list(self.nsclc_map.keys())) - 1)]
                aug_diff_slice_index = random.randint(0, self.nsclc_map[aug_differ_id] - 1)
        else:
                slice_cnt = self.nsclc_map[case_id]
                if slice_cnt == 1:
                    aug_same_slice_index = 0
                else:
                  aug_same_slice_index = random.randint(0, slice_cnt - 1)
                  while (aug_same_slice_index == file_index):
                    aug_same_slice_index = random.randint(0, slice_cnt - 1)
                aug_differ_id = list(self.sclc_map.keys())[random.randint(0, len(list(self.sclc_map.keys())) - 1)]
                aug_diff_slice_index = random.randint(0, self.sclc_map[aug_differ_id] - 1)

        aug_same_id_ct_data = ct_file[aug_same_slice_index]
        aug_diff_id_ct_data = np.load(os.path.join(self.root_dir, "{}_CT.npy".format(aug_differ_id)))[aug_diff_slice_index]
        aug_same_id_ct_data = aug_same_id_ct_data.reshape(1, aug_same_id_ct_data.shape[0], aug_same_id_ct_data.shape[1])
        aug_diff_id_ct_data = aug_diff_id_ct_data.reshape(1, aug_diff_id_ct_data.shape[0], aug_diff_id_ct_data.shape[1])
        aug_same_id_ct_data = aug_same_id_ct_data.astype(np.float32)
        aug_diff_id_ct_data = aug_diff_id_ct_data.astype(np.float32)

        #对当前切片进行额外数据增强
        data = {"image": image, "label":label_data}
        if self.transform is not None:
            data = self.transform(data)
        image = data["image"]
        image = image.astype(np.float32)
        aug_ct_data = image
        aug_ct_data = aug_ct_data.reshape(1, aug_ct_data.shape[0], aug_ct_data.shape[1])

        #获取肿瘤边缘作为边缘的标签
        edge_label = label_data * 255
        edge_label = edge_label.astype(np.uint8)
        contours, _ = cv2.findContours(edge_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_label = cv2.drawContours(np.zeros(edge_label.shape).astype(np.uint8), contours, -1, 255, 1)
        edge_label[edge_label > 0] = 1
        end = time.time()


        return (case_id,  ct_data, aug_ct_data, aug_same_id_ct_data, aug_diff_id_ct_data, label_data, cls_label, edge_label)



    def __len__(self):
        max_length = 0
        for key in self.idx_map.keys():
            max_length = max(max_length, self.idx_map[key][1])
        return max_length


    def fetch_train_valid_ids(self):
        train_ids = []
        valid_ids = []
        with open(os.path.join(self.project_dir, self.train_file), "r") as file:
            for line in file:
                train_ids.append(line.strip())
        with open(os.path.join(self.project_dir, self.valid_file), "r") as file:
            for line in file:
                valid_ids.append(line.strip())
        return (train_ids, valid_ids)

    def fetch_data_map(self, ids):
        """
        获取两个map,data_map{key:case_id,valud:切片},idx_map{key:case_id, value:[起止切片]}
        :param ids:
        :return:
        """
        data_map = {}
        idx_map = {}
        sclc_map = {}
        nsclc_map = {}
        begin_init = 0
        for id in ids:
            file_name = os.path.join(self.root_dir, "{}_Label.py.npy".format(str(id)))
            label = np.load(file_name)
            data_map[id] = []
            if str(id).startswith('B'):
                sclc_map[id] = label.shape[0]
            else:
                nsclc_map[id] = label.shape[0]
            for i in range(label.shape[0]):
                if label[i].sum() <= 0:
                    continue
                data_map[id].append(i)
            idx_map[id] = [begin_init, begin_init+len(data_map[id])]
            begin_init += len(data_map[id])
        return (data_map, idx_map, sclc_map, nsclc_map)



    def calcuate_file_from_index(self, index):
        case_id = -1
        for key in self.idx_map.keys():
            if self.idx_map[key][0] <= index and self.idx_map[key][1] > index:
                case_id = key
                break
        file_index = self.data_map[case_id][index - self.idx_map[case_id][0]]
        return (case_id, file_index)


    def get_data_map(self):
        return self.data_map

    def get_idx_map(self):
        return  self.idx_map


