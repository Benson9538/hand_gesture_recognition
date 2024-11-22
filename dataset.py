# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self, data_folder, label_mapping, window_size=30, stride=5):
        """
        自定義資料集類，用於加載特徵和標籤，並進行滑動窗口切分。

        Args:
            data_folder (str): 特徵和標籤保存的資料夾（train、val、test 子資料夾）。
            label_mapping (dict): 標籤到索引的映射。
            window_size (int): 滑動窗口大小。
            stride (int): 窗口移動的步長。
        """
        self.data_files = []
        self.label_mapping = label_mapping
        self.window_size = window_size
        self.stride = stride

        # 讀取所有的 .npz 文件
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.npz'):
                    self.data_files.append(os.path.join(root, file))

        if len(self.data_files) == 0:
            raise ValueError(f"在 {data_folder} 中未找到任何 .npz 文件")

        self.samples = []
        for file in self.data_files:
            data = np.load(file)
            features = data['features']
            labels = data['labels']

            # 將標籤轉換為索引
            labels = np.array([self.label_mapping[label] for label in labels])

            # 滑動窗口切分
            for i in range(0, len(features) - window_size + 1, stride):
                window_features = features[i:i+window_size]
                window_labels = labels[i:i+window_size]

                # 確定窗口標籤
                label_counts = np.bincount(window_labels)
                window_label_idx = label_counts.argmax()

                self.samples.append((window_features, window_label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_features, window_label_idx = self.samples[idx]
        # 轉換為 Tensor
        window_features = torch.tensor(window_features, dtype=torch.float32)
        window_label_idx = torch.tensor(window_label_idx, dtype=torch.long)
        return window_features, window_label_idx
