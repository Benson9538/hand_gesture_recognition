import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import GestureDataset
from model import GestureRecognitionModel
from data_preprocessing import create_label_mapping
from post_processing import post_process_predictions
import Levenshtein as lev

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

def calculate_levenshtein_precision(true_segments, pred_segments):
    true_labels = ''.join([label for label, _, _ in true_segments])
    pred_labels = ''.join([label for label, _, _ in pred_segments])
    levenshtein_distance = lev.distance(true_labels, pred_labels)
    max_len = max(len(true_labels), len(pred_labels))
    levenshtein_precision = 1 - (levenshtein_distance / max_len)
    return levenshtein_precision

def calculate_iou(true_segments, pred_segments):
    """
    計算真實手勢段與預測手勢段之間的平均 IoU。

    Args:
        true_segments (list): 真實手勢段列表，每個元素為 (手勢標籤, 開始幀, 結束幀)。
        pred_segments (list): 預測手勢段列表，每個元素為 (手勢標籤, 開始幀, 結束幀)。

    Returns:
        float: 平均 IoU 值。
    """
    total_iou = 0.0
    count = 0

    for true_label, true_start, true_end in true_segments:
        for pred_label, pred_start, pred_end in pred_segments:
            if true_label == pred_label:  # 只有相同標籤才計算 IoU
                intersection_start = max(true_start, pred_start)
                intersection_end = min(true_end, pred_end)
                intersection = max(0, intersection_end - intersection_start)

                union_start = min(true_start, pred_start)
                union_end = max(true_end, pred_end)
                union = union_end - union_start

                if union > 0:  # 確保有重疊部分
                    iou = intersection / union
                    total_iou += iou
                    count += 1

    # 如果沒有相交部分，返回 0
    return total_iou / count if count > 0 else 0.0



def plot_gesture_timeline(true_segments, pred_segments, label_mapping, title='Gesture Timeline', filename=None):
    try:
        # 獲取所有唯一標籤
        unique_labels = list(label_mapping.keys())
        colors = plt.cm.get_cmap('tab20', len(unique_labels))
        label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
        
        fig, ax = plt.subplots(figsize=(15, 3))  # 調整圖表大小

        # 繪製真實標籤的長條圖
        for label, start, end in true_segments:
            if label not in label_to_color:
                raise ValueError(f"Label '{label}' not found in label_mapping")
            ax.barh(1, end - start, left=start, color=label_to_color[label], edgecolor='none', height=0.4) #, label=f'True: {label}'
        
        # 繪製預測標籤的長條圖
        for label, start, end in pred_segments:
            if label not in label_to_color:
                raise ValueError(f"Label '{label}' not found in label_mapping")
            ax.barh(0, end - start, left=start, color=label_to_color[label], edgecolor='none', height=0.4)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Predicted', 'True'])
        ax.set_title(title)
        ax.set_xlim(0, 5000)  # 調整 x 軸範圍

        # 顯示圖例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # 顯示圖表
        plt.show()

        # 關閉圖表以釋放記憶體
        plt.close(fig)
    except Exception as e:
        print(f"生成圖表時發生錯誤: {e}")


def main():
    all_labels = ['D0X', 'B0A', 'B0B', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11']
    label_mapping = create_label_mapping(all_labels)
    index_to_label = {v: k for k, v in label_mapping.items()}

    data_root = 'D:/hand/hand/one_stage/output'
    batch_size = 64
    num_epochs = 15

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = {x: GestureDataset(os.path.join(data_root, x), label_mapping) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

    num_classes = len(label_mapping)
    input_size = 63*2
    num_channels = [64, 128, 256 ,512]

    # 初始化權重列表
    class_weights = []

    for label in label_mapping.keys():
        if label in ['D0X', 'B0A', 'B0B']:
            weight = 0.9  # 為頻繁出現的類別賦予較低的權重
        else:
            weight = 1.2  # 為其他類別賦予較高的權重
        class_weights.append(weight)

    # 將權重轉換為 Tensor 並移動到設備上
    class_weights = torch.FloatTensor(class_weights).to(device)

    model = GestureRecognitionModel(num_classes, input_size, num_channels)
    model.load_state_dict(torch.load('gesture_recognition_model.pth'))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # model = train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=num_epochs)
    # torch.save(model.state_dict(), 'gesture_recognition_model.pth')
    # print("模型已保存。")

    # 測試模型並進行後處理
    test_dataset = GestureDataset(os.path.join(data_root, 'test'), label_mapping)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())


    # 隨機選取一個 5000 幀的片段2
    start_frame = np.random.randint(0, len(all_predictions) - 5000)
    end_frame = start_frame + 5000
    pred_segment = all_predictions[start_frame:end_frame]
    true_segment = all_true_labels[start_frame:end_frame]

    pred_segments = post_process_predictions(pred_segment, window_size=30, stride=5, index_to_label=index_to_label)
    true_segments = post_process_predictions(true_segment, window_size=30, stride=5, index_to_label=index_to_label)

    levenshtein_precision = calculate_levenshtein_precision(true_segments, pred_segments)
    print(f"Levenshtein Precision: {levenshtein_precision:.4f}")

    plot_gesture_timeline(true_segments, pred_segments, label_mapping, title='Gesture Timeline')

if __name__ == '__main__':
    main()