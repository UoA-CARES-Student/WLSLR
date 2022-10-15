import math
import os
import os.path
import pathlib

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


import dataset.wlasl_dataset_utilities as wdu
import dataset.msasl_dataset_utilities as mdu

# Datasets that are compatible with this nslt_dataset construction
WLASL_DATASET = 'wlasl'
MSASL_DATASET = 'msasl'


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        try:
            img = cv2.imread(os.path.join(image_dir, vid, "image_" + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
        except:
            print(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
    video_path = os.path.join(vid_root, vid + '.mp4')
    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, img = vidcap.read()
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def pad(imgs, label, total_frames):
    if imgs.shape[0] < total_frames:
        num_padding = total_frames - imgs.shape[0]

        if num_padding:
            prob = np.random.random_sample()
            if prob > 0.5:
                pad_img = imgs[0]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
            else:
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
    else:
        padded_imgs = imgs

    if label is not None:
        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

    return padded_imgs, label


def make_dataset(split_file: str, dataset_type: str, split: str,
                 root_dir: str, mode: str, num_classes: int) -> list:

    if dataset_type == WLASL_DATASET:
        return wdu.wlasl_make_dataset(
            split_file=split_file,
            split=split,
            root_dir=root_dir,
            mode=mode,
            num_classes=num_classes)

    elif dataset_type == MSASL_DATASET:
        return mdu.msasl_make_dataset(
            split=split,
            root_dir=root_dir,
            mode=mode,
            num_classes=num_classes)


def get_num_class(split_file: str, dataset_type: str,
                  split: str, root_dir: str) -> int:

    if dataset_type == WLASL_DATASET:
        return wdu.wlasl_num_class(split_file=split_file)

    elif dataset_type == MSASL_DATASET:
        return mdu.msasl_num_class(root_dir=root_dir, split=split)


class NSLT(data_utl.Dataset):
    """
    Attributes:
        dataset_type: Indicates what dataset is being used [wlasl, msasl]
        split_file: Json file containing the labeling of the videos
        split: Determines the split of the dataset [train, val, test]
        root_dir: Path to the location of the videos
        mode: Indicates what mode this dataset will be used for [rgb, flow]
        transforms: [Optional] Transforms to be used on the videos
    """
    def __init__(self, dataset_type: str, split_file: str,
                 split: str, root_dir: str, mode: str,
                 num_classes=None,
                 transforms=None) -> None:
        self.dataset_type = dataset_type
        self.split_file = split_file
        self.split = split
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms

        if num_classes is None:
            self.num_classes = get_num_class(
                split_file=split_file,
                dataset_type=dataset_type,
                split=split,
                root_dir=root_dir)
        else:
            self.num_classes = num_classes

        self.data = make_dataset(
            split_file=split_file,
            dataset_type=dataset_type,
            split=split,
            root_dir=root_dir,
            mode=mode,
            num_classes=self.num_classes)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, src, start_frame, nf = self.data[index]

        total_frames = 64
        if self.dataset_type == WLASL_DATASET:
            imgs = load_rgb_frames_from_video(self.root_dir['word'], vid, start_frame, total_frames)

        elif self.dataset_type == MSASL_DATASET:
            vid_root = pathlib.Path(self.root_dir, self.split)
            vid_name = pathlib.Path(vid).stem
            imgs = load_rgb_frames_from_video(str(vid_root), vid_name, start_frame, total_frames)

        if imgs.size == 0:
            print(vid_root, vid_name, imgs)
            raise ValueError()

        imgs, label = pad(imgs, label, total_frames)

        imgs = self.transforms(imgs)

        ret_lab = torch.from_numpy(label)
        ret_img = video_to_tensor(imgs)

        return ret_img, ret_lab, vid

    def __len__(self):
        return len(self.data)

    @staticmethod
    def pad_wrap(imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                pad = imgs[:min(num_padding, imgs.shape[0])]
                k = num_padding // imgs.shape[0]
                tail = num_padding % imgs.shape[0]

                pad2 = imgs[:tail]
                if k > 0:
                    pad1 = np.array(k * [pad])[0]

                    padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
                else:
                    padded_imgs = np.concatenate([imgs, pad2], axis=0)
        else:
            padded_imgs = imgs

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label

