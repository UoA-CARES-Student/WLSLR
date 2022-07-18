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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_best_model(trained_models_dir: str) -> str:
    """
    Returns path to the best trained model in a given directory
    
    Args:
        trained_models_dir: Path to directory containing trained i3d models
    """
    models_dir = pathlib.Path(trained_models_dir)

    best_model_path = ''
    best_model_accuracy = 0
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
    if split_file == None and dataset_type == nslt_dataset.WLASL_DATASET:
        raise RuntimeError("No split file was provided when using the WLASL dataset.")

    print("CUDA is avaliable: ", torch.cuda.is_available())

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # Testing dataset
    val_dataset = Dataset(
        dataset_type=dataset_type,
        split_file=split_file,
        split='val',
        root_dir=root_dir,
        mode=mode,
        transforms=test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=False)

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        weight_path = pathlib.Path(
            WLSLR_GIT_PATH, "RGB", "I3D", "weights", "flow_imagenet.pt")
        i3d.load_state_dict(torch.load(weight_path))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        weight_path = pathlib.Path(
            WLSLR_GIT_PATH, "RGB", "I3D", "weights", "rgb_imagenet.pt")
        i3d.load_state_dict(torch.load(weight_path))

    i3d.replace_logits(val_dataset.num_classes)
    trained_model = get_best_model(trained_models_dir=trained_models_dir)
    i3d.load_state_dict(torch.load(trained_model))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    # Set up results containers
    correct = 0
    correct_5 = 0
    correct_10 = 0

    top1_fp = np.zeros(val_dataset.num_classes, dtype=int)
    top1_tp = np.zeros(val_dataset.num_classes, dtype=int)

    top5_fp = np.zeros(val_dataset.num_classes, dtype=int)
    top5_tp = np.zeros(val_dataset.num_classes, dtype=int)

    top10_fp = np.zeros(val_dataset.num_classes, dtype=int)
    top10_tp = np.zeros(val_dataset.num_classes, dtype=int)

    # Test the model with the dataset
    for data in val_dataloader:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        per_frame_logits = i3d(inputs)

        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        out_probs = np.sort(predictions.cpu().detach().numpy()[0])
        
        data_class_num = labels[0].nonzero()[0][0].item()

        # Top-5 accuracy
        if data_class_num in out_labels[-5:]:
            correct_5 += 1
            top5_tp[data_class_num] += 1
        else:
            top5_fp[data_class_num] += 1

        # Top-10 accuracy
        if data_class_num in out_labels[-10:]:
            correct_10 += 1
            top10_tp[data_class_num] += 1
        else:
            top10_fp[data_class_num] += 1

        # Top-1 accuracy
        if torch.argmax(predictions[0]).item() == data_class_num:
            correct += 1
            top1_tp[data_class_num] += 1
        else:
            top1_fp[data_class_num] += 1
        
        print(video_id, float(correct) / len(val_dataloader), 
              float(correct_5) / len(val_dataloader),
              float(correct_10) / len(val_dataloader))

    # per-class accuracy
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))


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
