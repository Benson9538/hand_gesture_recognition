# data_preprocessing.py

import numpy as np
import mediapipe as mp
import cv2
import os
import random
import json
import pandas as pd
from collections import defaultdict

def read_annotations(annotation_file):
    """
    讀取標註資料，返回一個包含影片資訊的字典。

    Args:
        annotation_file (str): 標註文件的路徑。

    Returns:
        dict: 以影片名稱為鍵，包含手勢資訊的字典。
    """
    annotations = pd.read_csv(annotation_file)
    video_dict = defaultdict(list)

    for _, row in annotations.iterrows():
        video = row['video']
        gesture_info = {
            'label': row['label'],
            't_start': row['t_start'],
            't_end': row['t_end'],
            'frames': row['frames']
        }
        video_dict[video].append(gesture_info)

    return video_dict

def split_dataset(video_dict, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    將影片資料集劃分為訓練集、驗證集和測試集，考慮類別平衡。

    Args:
        video_dict (dict): 包含影片資訊的字典。
        train_ratio (float): 訓練集比例。
        val_ratio (float): 驗證集比例。
        test_ratio (float): 測試集比例。

    Returns:
        dict: 包含訓練集、驗證集和測試集的影片清單。
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例總和必須為 1"

    # 按影片名稱排序，確保一致性
    videos = sorted(video_dict.keys())
    random.shuffle(videos)  # 隨機打亂

    total_videos = len(videos)
    train_count = int(total_videos * train_ratio)
    val_count = int(total_videos * val_ratio)

    # 劃分資料集
    train_videos = videos[:train_count]
    val_videos = videos[train_count:train_count + val_count]
    test_videos = videos[train_count + val_count:]

    dataset_splits = {
        'train': train_videos,
        'val': val_videos,
        'test': test_videos
    }

    return dataset_splits

def save_json(data, filename):
    """
    將資料保存為 JSON 格式。

    Args:
        data (dict): 要保存的數據。
        filename (str): 保存的文件名。
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    """
    從 JSON 文件中加載資料。

    Args:
        filename (str): 要加載的文件名。

    Returns:
        dict: 加載的數據。
    """
    with open(filename, 'r') as f:
        return json.load(f)

def generate_frame_labels(video_length, gestures):
    """
    生成逐幀的手勢標籤序列。

    Args:
        video_length (int): 影片的總幀數。
        gestures (list): 影片中的手勢資訊列表。

    Returns:
        list: 長度為 video_length 的標籤列表。
    """
    labels = ['D0X'] * video_length
    for gesture in gestures:
        start = max(0, gesture['t_start'] - 1)  # 假設索引從 0 開始
        end = gesture['t_end']
        label = gesture['label']
        labels[start:end] = [label] * (end - start)
    return labels

def normalize_features(features):
    """
    對特徵進行歸一化。

    Args:
        features (np.array): 特徵數組。

    Returns:
        np.array: 歸一化後的特徵數組。
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # 避免除以零
    normalized_features = (features - mean) / std
    return normalized_features

def extract_features_and_labels(video_path, gestures, output_path):
    """
    使用 Mediapipe 從影片中提取手部關鍵點特徵，並生成逐幀標籤，同時計算速度變化和曼哈頓距離作為額外特徵。

    Args:
        video_path (str): 輸入影片的路徑。
        gestures (list): 影片中的手勢資訊列表。
        output_path (str): 特徵和標籤輸出的檔案路徑（例如 .npz）。
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法打開影片：{video_path}")
            return

        features = []
        success_flags = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        labels = generate_frame_labels(frame_count, gestures)
        idx = 0  # 幀索引

        # 初始化 Mediapipe
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            print(f"正在處理影片：{video_path}")
            prev_landmarks = None  # 用於計算速度

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    # 提取關鍵點坐標
                    landmark_coords = []
                    for landmark in hand_landmarks.landmark:
                        landmark_coords.extend([landmark.x, landmark.y, landmark.z])

                    # 計算速度變化
                    if prev_landmarks is not None:
                        # 計算當前幀和前一幀的差值
                        velocity = [curr - prev for curr, prev in zip(landmark_coords, prev_landmarks)]
                    else:
                        velocity = [0] * len(landmark_coords)  # 第一幀速度設為零

                    # 將速度加入特徵向量
                    feature_vector = landmark_coords + velocity

                    prev_landmarks = landmark_coords  # 更新前一幀的關鍵點
                    success_flags.append(True)
                else:
                    # 若未能檢測到手部，填充零向量
                    num_coords = 21 * 3  # 手部關鍵點數量 * 每個關鍵點的坐標數
                    landmark_coords = [0] * num_coords
                    velocity = [0] * num_coords
                    feature_vector = landmark_coords + velocity

                    prev_landmarks = None  # 無法計算速度
                    success_flags.append(False)

                features.append(feature_vector)
                idx += 1

        # 將特徵和標籤保存為 numpy 數組
        features = np.array(features)
        features = normalize_features(features)  # 歸一化特徵
        labels = np.array(labels[:len(features)])  # 確保標籤與特徵數量一致

        # 保存為 .npz 格式
        np.savez_compressed(output_path, features=features, labels=labels, success_flags=success_flags)

    except Exception as e:
        print(f"處理影片 {video_path} 時發生錯誤：{e}")
        raise  # 在開發過程中保留，便於調試

    finally:
        # 確保在任何情況下都釋放資源
        cap.release()
        print(f"已釋放影片資源：{video_path}")

def process_dataset(video_dict, dataset_splits, video_folder, output_folder):
    """
    對資料集中的所有影片進行特徵提取和標籤生成。

    Args:
        video_dict (dict): 包含影片資訊的字典。
        dataset_splits (dict): 包含訓練集、驗證集和測試集的影片清單。
        video_folder (str): 原始影片資料夾路徑。
        output_folder (str): 特徵和標籤保存的根目錄。
    """
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        videos = dataset_splits[subset]
        output_dir = os.path.join(output_folder, subset)
        os.makedirs(output_dir, exist_ok=True)

        for video in videos:
            video_path = os.path.join(video_folder, f"{video}.avi")
            gestures = video_dict[video]
            output_path = os.path.join(output_dir, f"{video}.npz")
            extract_features_and_labels(video_path, gestures, output_path)

    print("所有特徵和標籤已提取完成。")

def create_label_mapping(labels):
    """
    創建標籤到索引的映射。

    Args:
        labels (list): 所有出現過的標籤列表。

    Returns:
        dict: 標籤到索引的映射。
    """
    label_set = set(labels)
    label_to_index = {label: idx for idx, label in enumerate(sorted(label_set))}
    return label_to_index

def main():
    annotation_file = "D:/hand/hand/one_stage/Annot_List.txt"
    video_dict_file = "D:/hand/hand/one_stage/video_dict.json"
    dataset_splits_file = "D:/hand/hand/one_stage/dataset_splits.json"

    # 如果已經存在記錄文件，則加載它們，否則從頭生成
    if os.path.exists(video_dict_file) and os.path.exists(dataset_splits_file):
        print("加載已保存的 video_dict 和 dataset_splits...")
        video_dict = load_json(video_dict_file)
        dataset_splits = load_json(dataset_splits_file)
    else:
        if not os.path.exists(annotation_file):
            print(f"標註文件不存在：{annotation_file}")
            return
        video_dict = read_annotations(annotation_file)
        dataset_splits = split_dataset(video_dict)

        # 保存 video_dict 和 dataset_splits
        save_json(video_dict, video_dict_file)
        save_json(dataset_splits, dataset_splits_file)
        print(f"已保存 video_dict 至 {video_dict_file}")
        print(f"已保存 dataset_splits 至 {dataset_splits_file}")

    video_folder = "D:/hand/hand/video"
    if not os.path.exists(video_folder):
        print(f"影片資料夾不存在：{video_folder}")
        return
    output_folder = "D:/hand/hand/one_stage/output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_dataset(video_dict, dataset_splits, video_folder, output_folder)

    all_labels = set()
    for gestures in video_dict.values():
        for gesture in gestures:
            all_labels.add(gesture['label'])

    label_mapping = create_label_mapping(list(all_labels))

    # 保存標籤映射
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    print("標籤映射已保存至 label_mapping.json")

if __name__ == '__main__':
    main()
