import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d

# from datasets.nslt_dataset import NSLT as Dataset
from dataset import nslt_dataset
from dataset.nslt_dataset import NSLT as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_i3d(
    dataset_type: str,
    root_dir: str,
    config_file: str,
    save_dir: str,
    mode: str,
    train_split: str,
    split_file: str = None,
    weights: str = None
) -> None:
    if split_file == None and dataset_type == nslt_dataset.WLASL_DATASET:
        raise RuntimeError("No split file was provided when using the WLASL dataset.")
    
    # Create Config from the config file
    configs = Config(config_file)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(
        dataset_type=dataset_type,
        split_file=split_file,
        split='train',
        root_dir=root_dir,
        mode=mode,
        transforms=train_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0
    # train it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < configs.max_steps and epoch < 400:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            collected_vids = []

            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1: # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    continue

                # inputs, labels, vid, src = data
                inputs, labels, vid = data

                # wrap them in Variable
                inputs = inputs.cuda()
                t = inputs.size(2)
                labels = labels.cuda()

                per_frame_logits = i3d(inputs, pretrained=False)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        print(
                            'Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                                 phase,
                                                                                                                 tot_loc_loss / (10 * num_steps_per_update),
                                                                                                                 tot_cls_loss / (10 * num_steps_per_update),
                                                                                                                 tot_loss / 10,
                                                                                                                 acc))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = save_model + "nslt_" + str(num_classes) + "_" + str(steps).zfill(
                                   6) + '_%3f.pt' % val_score

                    torch.save(i3d.module.state_dict(), model_name)
                    print(model_name)

                print('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                              tot_loc_loss / num_iter,
                                                                                                              tot_cls_loss / num_iter,
                                                                                                              (tot_loss * num_steps_per_update) / num_iter,
                                                                                                              val_score
                                                                                                              ))

                scheduler.step(tot_loss * num_steps_per_update / num_iter)


def train_i3d_cli(argv) -> None:
    """
    Used to train a i3d model using a video dataset.

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
        'root_dir',
        help='The location of the root directory containing the datasets',
        type=str)

    parser.add_argument(
        'config_file',
        help='Path to the config file',
        type=str)

    parser.add_argument(
        'save_dir',
        help='Location of directory to save the model',
        type=str)

    parser.add_argument(
        '--mode',
        help='rgb or flow',
        default='rgb',
        type=str)

    parser.add_argument(
        '--train-split',
        help='Path to split file for the WLASL dataset',
        type=str)

    parser.add_argument(
        '--split-file',
        help='Path to the split file if training with the wlasl dataset',
        type=str)

    parser.add_argument(
        '--weights',
        help='Path to pre-trained weights',
        type=str)

    args = parser.parse_args(argv)

    train_i3d(
        dataset_type=args.dataset_type,
        root_dir=args.root_dir,
        config_file=args.config_file,
        save_dir=args.save_dir,
        mode=args.mode,
        train_split=args.train_split,
        split_file=args.split_file,
        weights=args.weights)


if __name__ == '__main__':
    train_i3d_cli(sys.argv[1:])
