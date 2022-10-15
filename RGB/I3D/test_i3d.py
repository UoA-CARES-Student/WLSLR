import argparse
import pathlib
import sys
import os

import videotransforms
import numpy as np

import torch
import torch.nn as nn
from pytorch_i3d import InceptionI3d
from torchvision import transforms

from dataset import nslt_dataset
from dataset.nslt_dataset import NSLT as Dataset

WLSLR_GIT_PATH = os.environ["WLSLR_GIT_PATH"]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_best_model(trained_models_dir: str) -> str:
    """
    Returns path to the best trained model in a given directory

    Args:
        trained_models_dir: Path to directory containing trained i3d models
    """
    models_dir = pathlib.Path(trained_models_dir)

    best_model_path = ''
    best_model_accuracy = 0
    #best_model_path = ".\RGB\I3D\\archived\\asl100\FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt"
    for model in models_dir.iterdir():
        if model.is_file() and model.suffix == '.pt':
            model_accuracy = float(model.stem.split('_')[-1])
            if model_accuracy > best_model_accuracy:
                best_model_accuracy = model_accuracy
                best_model_path = model

    return best_model_path


def test_i3d(
    dataset_type: str,
    trained_models_dir: str,
    root_dir: str,
    mode: str,
    split_file: str = None,
) -> None:
    if split_file is None and dataset_type == nslt_dataset.WLASL_DATASET:
        raise RuntimeError("No split file was provided when using the WLASL dataset.")

    print("CUDA is avaliable: ", torch.cuda.is_available())

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # Testing dataset
    val_dataset = Dataset(
        dataset_type=dataset_type,
        split_file=split_file,
        split='test',
        root_dir=root_dir,
        mode=mode,
        transforms=test_transforms,
        num_classes=100)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=False)

    # setup the model
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(val_dataset.num_classes)
    trained_model = get_best_model(trained_models_dir=trained_models_dir)
    i3d.load_state_dict(torch.load(trained_model))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    correct = 0
    count = 0
    correct_vids = []
    # Test the model with the dataset
    for data in val_dataloader:
        count = count + 1
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        per_frame_logits = i3d(inputs)

        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        out_probs = np.sort(predictions.cpu().detach().numpy()[0])

        data_class_num = labels[0].nonzero()[0][0].item()

        # Top-1 accuracy
        if torch.argmax(predictions[0]).item() == data_class_num:
            correct = correct + 1
            correct_vids.append(video_id)

    print(correct_vids)
    print(correct, "/", count)


def test_i3d_cli(argv) -> None:
    """
    Used to test a i3d model using a video dataset.

    Args:
        Check the cli help command -h for more description on the args.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        'dataset_type',
        help='Indicates which dataset is being used [wlasl, msasl]',
        choices=['wlasl', 'msasl'],
        type=str)

    parser.add_argument(
        'trained_models_dir',
        help='Path to directory of trained models',
        type=str)

    parser.add_argument(
        'root_dir',
        help='The location of the root directory containing the datasets',
        type=str)

    parser.add_argument(
        '--mode',
        help='rgb or flow',
        default='rgb',
        type=str)

    parser.add_argument(
        '--split-file',
        help='Path to the split file if training with the wlasl dataset',
        type=str)

    args = parser.parse_args(argv)

    test_i3d(
        dataset_type=args.dataset_type,
        trained_models_dir=args.trained_models_dir,
        root_dir=args.root_dir,
        mode=args.mode,
        split_file=args.split_file)


if __name__ == '__main__':
    test_i3d_cli(sys.argv[1:])
