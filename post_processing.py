# post_processing.py

def post_process_predictions(predictions, window_size, stride, index_to_label):
    """
    將模型的預測結果轉換為手勢的起始和結束時間。

    Args:
        predictions (list or np.array): 模型對每個窗口的預測標籤索引。
        window_size (int): 滑動窗口大小。
        stride (int): 窗口移動的步長。
        index_to_label (dict): 索引到標籤名稱的映射。

    Returns:
        list: 包含手勢段的列表，每個元素為 (手勢標籤, 開始幀, 結束幀)。
    """
    gestures = []
    prev_label = None
    gesture_start = None

    for i, label_idx in enumerate(predictions):
        label = index_to_label[label_idx]
        frame_start = i * stride
        frame_end = frame_start + window_size

        if label != prev_label:
            if prev_label is not None and prev_label != 'D0X':
                gestures.append((prev_label, gesture_start, frame_start))
            if label != 'D0X':
                gesture_start = frame_start
        prev_label = label

    # 處理最後一個手勢
    if prev_label is not None and prev_label != 'D0X':
        gestures.append((prev_label, gesture_start, frame_end))
        
    return gestures