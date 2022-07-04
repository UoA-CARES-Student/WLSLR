import pathlib

import cv2


def msasl_num_class(root_dir: str, split: str) -> int:
    """
    Returns the number of classes in the dataset.
    """
    path = pathlib.Path(root_dir, split)

    classes = []
    for file in path.iterdir():
        class_num = file.stem[0:3]
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

    count_skipping = 0
    for vid in path_root_dir.iterdir():
        vid_name = vid.name

        vid_label = vid_name[:3]

        if int(vid_label) > num_classes:
            raise RuntimeError(
                f"'{vid_label}' is in the subset range of {num_classes}")

        src = 0
        starting_frame = 0

        num_frames = int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FRAME_COUNT))

        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames < 9:
            print("Skip video ", vid)
            count_skipping += 1
            continue

        dataset.append((vid_name, vid_label, src, starting_frame, num_frames))

    print("Skipped videos: ", count_skipping)
    print(len(dataset))

    return dataset
