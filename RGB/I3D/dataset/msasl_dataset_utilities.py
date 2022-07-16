import pathlib

import cv2
import numpy as np


def msasl_num_class(root_dir: str, split: str) -> int:
    """
    Returns the number of classes in the dataset.
    """
    path = pathlib.Path(root_dir, split)

    classes = []
    for file in path.iterdir():
        class_num = file.stem[0:4]
        if class_num not in classes:
            classes.append(class_num)

    return len(classes)


def msasl_make_dataset(split: str, root_dir: str, mode: str,
                       num_classes: int) -> list:
    """
    Create a list containing details about the videos in the dataset to be
    used by the dataloader.
    """
    dataset = []

    path_root_dir = pathlib.Path(root_dir, split)
    print(path_root_dir)

    count_skipping = 0
    for vid in path_root_dir.iterdir():
        vid_name = vid.name

        num_frames = int(cv2.VideoCapture(str(vid)).get(cv2.CAP_PROP_FRAME_COUNT))

        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames < 9:
            print("Skip video ", vid)
            count_skipping += 1
            continue

        if int() > num_classes:
            raise RuntimeError(
                f"'{vid_name[:4]}' is in the subset range of {num_classes}")

        vid_label = np.zeros((num_classes, num_frames), np.float32)
        
        for i in range(num_frames):
            c_ = int(vid_name[:4])
            vid_label[c_][i] = 1

        #with np.printoptions(threshold=np.inf):
        #    print(vid_label)
        #break

        src = 0
        starting_frame = 0

        dataset.append((vid_name, vid_label, src, starting_frame, num_frames))

    print("Skipped videos: ", count_skipping)
    print(len(dataset))

    return dataset
