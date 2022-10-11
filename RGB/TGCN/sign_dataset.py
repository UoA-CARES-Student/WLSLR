import glob
import json
import math
import os
import random

import torch
import pathlib
import utils

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset

WLASL_CLASSES = "/home/myuser1/WLSLR/RGB/WLASL_Utilities/WLASL_classes.json"
MSASL_CLASSES = "/home/myuser1/WLSLR/RGB/MSASL_Utilies/MS-ASL/MSASL_classes.json"

def compute_difference(x):
    diff = []

    for i, xx in enumerate(x):
        temp = []
        for j, xxx in enumerate(x):
            if i != j:
                temp.append(xx - xxx)

        diff.append(temp)

    return diff


def read_pose_file(filepath, split):
    body_pose_exclude = {}

    try:
        content = json.load(open(filepath))
    except IndexError:
        return None

    path_parts = os.path.split(filepath)

    frame_id = path_parts[1][:11]
    vid = os.path.split(path_parts[0])[-1]

    save_to = os.path.join(f'/home/myuser1/WLSLR/RGB/WLASL-100/{split}/data/pose_per_individual_videos', vid)

    try:
        ft = torch.load(os.path.join(save_to, frame_id + '_ft.pt'))

        xy = ft[:, :2]
        # angles = torch.atan(ft[:, 110:]) / 90
        # ft = torch.cat([xy, angles], dim=1)
        return xy

    except FileNotFoundError:
        # print(filepath)
        body_pose = content["pose_keypoints_2d"]
        left_hand_pose = content["hand_left_keypoints_2d"]
        right_hand_pose = content["hand_right_keypoints_2d"]

        body_pose.extend(left_hand_pose)
        body_pose.extend(right_hand_pose)

        x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
        y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]
        # conf = [v for i, v in enumerate(body_pose) if i % 3 == 2 and i // 3 not in body_pose_exclude]

        x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
        y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)
        # conf = torch.FloatTensor(conf)

        x_diff = torch.FloatTensor(compute_difference(x)) / 2
        y_diff = torch.FloatTensor(compute_difference(y)) / 2

        zero_indices = (x_diff == 0).nonzero()

        orient = y_diff / x_diff
        orient[zero_indices] = 0

        xy = torch.stack([x, y]).transpose_(0, 1)

        ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

        path_parts = os.path.split(filepath)

        frame_id = path_parts[1][:11]
        vid = os.path.split(path_parts[0])[-1]

        save_to = os.path.join(f'/home/myuser1/WLSLR/RGB/WLASL-100/{split}/data/pose_per_individual_videos', vid)
        if not os.path.exists(save_to):
            os.mkdir(save_to)
        torch.save(ft, os.path.join(save_to, frame_id + '_ft.pt'))

        xy = ft[:, :2]
        # angles = torch.atan(ft[:, 110:]) / 90
        # ft = torch.cat([xy, angles], dim=1)
        #
        return xy

    # return ft


class Sign_Dataset(Dataset):
    def __init__(self, split, dataset_root, pose_ext, sample_strategy='rnd_start', num_samples=25, num_copies=4,
                 img_transforms=None, video_transforms=None, test_index_file=None):
        self.data = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        self.dataset_root = dataset_root
        self.pose_ext = pose_ext

        self.framename = 'image_{}_keypoints.json'
        self.sample_strategy = sample_strategy
        self.num_samples = num_samples
        self.test_index_file = test_index_file
        self._make_dataset(split)
        self.img_transforms = img_transforms
        self.video_transforms = video_transforms

        self.num_copies = num_copies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, gloss_cat, frame_start, frame_end, split = self.data[index]
        # frames of dimensions (T, H, W, C)
        x = self._load_poses(video_id, frame_start, frame_end, self.sample_strategy, self.num_samples, split)

        if self.video_transforms:
            x = self.video_transforms(x)

        y = gloss_cat

        return x, y, video_id

    def _make_dataset(self, splits):
        classes_data = json.load(open(WLASL_CLASSES))[0:100]
        self.label_encoder.fit(classes_data)
        # self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        for split in splits:
            path_root_dir = pathlib.Path(self.dataset_root, split, self.pose_ext)

            for datum_dir in path_root_dir.iterdir():
                gloss_index = int(os.path.basename(datum_dir)[:4]) - 1
                gloss_label = classes_data[gloss_index]
                gloss_cat = utils.labels2cat(self.label_encoder, [gloss_label])[0]

                # name, gloss, start frame, end frame and split
                instance_entry = datum_dir.name, gloss_cat, 1, len(glob.glob1(str(datum_dir), "*.json")) - 1, split
                self.data.append(instance_entry)

    def _load_poses(self, video_id, frame_start, frame_end, sample_strategy, num_samples, split):
        """
        Load frames of a video.
        Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
        """
        poses = []

        if sample_strategy == 'rnd_start':
            frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'seq':
            frames_to_sample = sequential_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'k_copies':
            frames_to_sample = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples,
                                                                         self.num_copies)
        else:
            raise NotImplementedError('Unimplemented sample strategy found: {}.'.format(sample_strategy))

        for i in frames_to_sample:
            pose_path = os.path.join(self.dataset_root, split, self.pose_ext, video_id,
                                     self.framename.format(str(i).zfill(5)))

            # pose = cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
            pose = read_pose_file(pose_path, split)

            if pose is not None:
                if self.img_transforms:
                    pose = self.img_transforms(pose)

                poses.append(pose)
            else:
                try:
                    poses.append(poses[-1])
                except IndexError:
                    pass
                    # print(pose_path)

        pad = None

        # if len(frames_to_sample) < num_samples:
        if len(poses) < num_samples:
            num_padding = num_samples - len(frames_to_sample)
            last_pose = poses[-1]
            pad = last_pose.repeat(1, num_padding)

        poses_across_time = torch.cat(poses, dim=1)
        if pad is not None:
            poses_across_time = torch.cat([poses_across_time, pad], dim=1)

        return poses_across_time


def rand_start_sampling(frame_start, frame_end, num_samples):
    """Randomly select a starting point and return the continuous ${num_samples} frames."""
    num_frames = frame_end - frame_start + 1

    if num_frames > num_samples:
        select_from = range(frame_start, frame_end - num_samples + 1)
        sample_start = random.choice(select_from)
        frames_to_sample = list(range(sample_start, sample_start + num_samples))
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def sequential_sampling(frame_start, frame_end, num_samples):
    """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []
    if num_frames > num_samples:
        frames_skip = set()

        num_skips = num_frames - num_samples
        interval = num_frames // num_skips

        for i in range(frame_start, frame_end + 1):
            if i % interval == 0 and len(frames_skip) <= num_skips:
                frames_skip.add(i)

        for i in range(frame_start, frame_end + 1):
            if i not in frames_skip:
                frames_to_sample.append(i)
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []

    if num_frames <= num_samples:
        num_pads = num_samples - num_frames

        frames_to_sample = list(range(frame_start, frame_end + 1))
        frames_to_sample.extend([frame_end] * num_pads)

        frames_to_sample *= num_copies

    elif num_samples * num_copies < num_frames:
        mid = (frame_start + frame_end) // 2
        half = num_samples * num_copies // 2

        frame_start = mid - half

        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * num_samples,
                                               frame_start + i * num_samples + num_samples)))

    else:
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * stride,
                                               frame_start + i * stride + num_samples)))

    return frames_to_sample


if __name__ == '__main__':
    # root = '/home/dxli/workspace/nslt'
    #
    # split_file = os.path.join(root, 'data/splits-with-dialect-annotated/asl100.json')
    # pose_data_root = os.path.join(root, 'data/pose/pose_per_individual_videos')
    #
    # num_samples = 64
    #
    # train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
    #                              img_transforms=None, video_transforms=None,
    #                              num_samples=num_samples)
    #
    # train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    #
    # cnt = 0
    # for batch_idx, data in enumerate(train_data_loader):
    #     print(batch_idx)
    #     x = data[0]
    #     y = data[1]
    #     print(x.size())
    #     print(y.size())

    print(k_copies_fixed_length_sequential_sampling(0, 2, 20, num_copies=3))
