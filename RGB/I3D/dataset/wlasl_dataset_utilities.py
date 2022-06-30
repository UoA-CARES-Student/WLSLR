import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
from torchvision import transforms


def wlasl_num_class(split_file: str) -> int:
    classes = set()
    with open(split_file, 'r') as f:
        content = json.load(f)

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)
    return len(classes)


def wlasl_make_dataset(split_file: str, split: str, root_dir: str, mode: str,
                       num_classes: int) -> list:
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    count_skipping = 0
    for vid in data.keys():
        if split == 'train':
            if data[vid]['subset'] not in ['train', 'val']:
                continue
        else:
            if data[vid]['subset'] != 'test':
                continue

        vid_root = root_dir['word']
        src = 0

        video_path = os.path.join(vid_root, vid + '.mp4')
        if not os.path.exists(video_path):
            continue

        num_frames = int(
            cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames - 0 < 9:
            print("Skip video ", vid)
            count_skipping += 1
            continue

        label = np.zeros((num_classes, num_frames), np.float32)

        for i in range(num_frames):
            c_ = data[vid]['action'][0]
            label[c_][i] = 1

        if len(vid) == 5:
            dataset.append((vid, label, src, 0,
                            data[vid]['action'][2] - data[vid]['action'][1]))
        elif len(vid) == 6:  # sign kws instances
            dataset.append((vid, label, src, data[vid]['action'][1],
                            data[vid]['action'][2] - data[vid]['action'][1]))

        i += 1

    print("Skipped videos: ", count_skipping)
    print(len(dataset))

    return dataset
