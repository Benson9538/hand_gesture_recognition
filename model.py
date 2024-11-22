# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5  # 縮放因子

    def forward(self, x):
        # x: (batch_size, seq_length, dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # 計算注意力權重
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, seq_length, seq_length)
        attn = torch.softmax(scores, dim=-1)
        # 加權求和
        out = torch.matmul(attn, V)  # (batch_size, seq_length, dim)
        return out, attn

class GestureRecognitionModel(nn.Module):
    def __init__(self, num_classes, input_size, num_channels, kernel_size=3, dropout=0.5):
        """
        基於 TCN 的手勢識別模型，支持多任務學習。

        Args:
            num_classes (int): 手勢類別數量。
            input_size (int): 輸入特徵的維度。
            num_channels (list): 每個 TCN 層的通道數列表。
            kernel_size (int): 卷積核大小。
            dropout (float): Dropout 機率。
        """
        super(GestureRecognitionModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.attention = SelfAttention(num_channels[-1])        
        self.fc = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向傳播。

        Args:
            x (Tensor): 輸入張量，形狀為 (batch_size, seq_length, input_size)。

        Returns:
            Tensor: 輸出張量，形狀為 (batch_size, num_classes)。
        """
        # TCN 期望的輸入形狀為 (batch_size, input_size, seq_length)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # Self-Attention 層
        y = y.transpose(1, 2)  # 轉換回 (batch_size, seq_length, num_channels)
        y, _ = self.attention(y)
        y = torch.mean(y, dim=1)  # 平均池化
        y = self.dropout(y)
        output = self.fc(y)
        return output

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.batch_norm1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.batch_norm2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
