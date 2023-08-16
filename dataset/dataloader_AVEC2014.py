import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import torch


class AvecDataset(Dataset):
    def __init__(self, root_dir, label_file):
        self.root_dir = root_dir
        self.label_file = label_file
        self.id_label = self.load_labels()
        folder_names = os.listdir(root_dir)
        result = []
        for floder in folder_names:
            result = result + os.listdir(os.path.join(root_dir, floder))
        self.folder_names = result

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        folder_name = self.folder_names[idx]
        n_name = folder_name.split('_')[0] + '_' + folder_name.split('_')[1]
        # print(folder_name.split('_')[0])
        label = self.id_label[folder_name.split('_')[0]]
        n_name = n_name + '_' + folder_name.split('_')[2]
        folder_path = os.path.join(self.root_dir, n_name, folder_name)
        # 在此处根据需要加载和预处理文件夹中的数据
        # 首先读取相关的原始面部图像
        face_list = []
        fece_list_folder = os.listdir(folder_path)
        fece_list_folder.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # print(fece_list_folder)
        for face in fece_list_folder:
            # 读取面部图像
            image = cv2.imread(os.path.join(folder_path, face), 1)
            image = cv2.resize(image,(224,224)).transpose([2, 0, 1])
            face_list.append(torch.from_numpy(image))
        faces_tensor = torch.stack(face_list, dim=0)  # l,3,224,224
        return faces_tensor.to(torch.float32), torch.FloatTensor([label])

    def load_labels(self):
        label_df = pd.read_csv(self.label_file)
        labels = {}
        num = {}
        labels_final = {}
        for _, row in label_df.iterrows():
            folder_name = str(row['id']).split('_')[0]
            label = row['score']
            if folder_name in labels.keys():
                labels[folder_name] = labels[folder_name] + label
                num[folder_name] = num[folder_name] + 1
            else:
                labels[folder_name] = label
                num[folder_name] = 1
        for key in labels:
            labels_final[key] = labels[key] / num[key]
        print(labels_final)
        return labels_final

