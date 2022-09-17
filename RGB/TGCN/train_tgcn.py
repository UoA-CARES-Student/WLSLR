import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import utils
import wandb

from configs import Config
from tgcn_model import GCN_muti_att
from sign_dataset import Sign_Dataset
from train_utils import train, validation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run(dataset_root, pose_ext, configs, save_model_to=None):
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # setup dataset

    train_dataset = Sign_Dataset(split=['train', 'val'], dataset_root=dataset_root, pose_ext=pose_ext,
                                 img_transforms=None, video_transforms=None, num_samples=num_samples)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=True)

    val_dataset = Sign_Dataset(split=['test'], dataset_root=dataset_root, pose_ext=pose_ext,
                               img_transforms=None, video_transforms=None,
                               num_samples=num_samples,
                               sample_strategy='k_copies')
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                                  shuffle=True)

    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))

    # setup the model
    model = GCN_muti_att(input_feature=num_samples * 2, hidden_feature=num_samples * 2,
                         num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p,
                         num_stage=num_stages).cuda()

    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    # optimizer = optim.SGD(vgg_gru.parameters(), lr=lr, momentum=0.00001)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    # start training
    for epoch in range(int(epochs)):
        # train, test model

        print('start training.')
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('start testing.')
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=save_model_to)
        # print('start testing.')
        # val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
        #                                                                         val_data_loader, epoch,
        #                                                                         save_to=save_model_to)

        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
        logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
        logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
        logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        wandb.log({"loss": val_loss}, step=epoch)
        wandb.log({"accuracy": val_score[0]}, step=epoch)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])

        # save all train test results
        np.save('output/epoch_training_losses.npy', np.array(epoch_train_losses))
        np.save('output/epoch_training_scores.npy', np.array(epoch_train_scores))
        np.save('output/epoch_test_loss.npy', np.array(epoch_val_losses))
        np.save('output/epoch_test_score.npy', np.array(epoch_val_scores))

        if val_score[0] > best_test_acc:
            best_test_acc = val_score[0]
            best_epoch_num = epoch

            torch.save(model.state_dict(),
                       os.path.join('/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/TGCN/checkpoints/WLASL-100',
                                    'gcn_epoch={}_val_acc={}.pth'.format(best_epoch_num, best_test_acc)))

    utils.plot_curves()

    class_names = train_dataset.label_encoder.classes_
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
                                save_to='/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/TGCN/output/wlasl100-train'
                                        '-conf-mat')
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False,
                                save_to='/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/TGCN/output/wlasl100-val'
                                        '-conf-mat')

    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                       y_true=val_gts, preds=val_preds,
                                                       class_names=class_names)})


if __name__ == "__main__":
    wandb.init(project="wlasl-100-tgcn", entity="p4p_p23")

    root = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL-100'

    subset = 'asl100'

    # split_file = os.path.join(root, 'data/splits/{}.json'.format(subset))
    pose_ext = os.path.join('data/pose_per_individual_videos')
    config_file = os.path.join(root,
                               '/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/TGCN/configs/{}.ini'.format(subset))
    configs = Config(config_file)

    logging.basicConfig(filename='/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/TGCN/output/wlasl100.log',
                        level=logging.DEBUG, filemode='w+')

    logging.info('Calling main.run()')
    run(configs=configs, pose_ext=pose_ext, dataset_root=root)
    logging.info('Finished main.run()')
    # utils.plot_curves()
