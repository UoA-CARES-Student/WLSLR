import torch
import torch.nn as nn
import numpy as np
import torchvision
#import skvideo
#skvideo.setFFmpegPath("C:/Users/isabe/Downloads/ffmpeg-5.0.1-full_build-shared/bin")
#import skvideo.io
import dataset.nslt_dataset as nslt_dataset
from pytorch_i3d import InceptionI3d
import videotransforms
from torchvision import transforms
import cv2
import pathlib

if __name__ == "__main__":
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    vid_path = pathlib.Path('/home/myuser1/WLSLR/RGB/data/test/0006005.mp4')
    vidcap = cv2.VideoCapture(vid_path)
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    imgs = nslt_dataset.load_rgb_frames_from_video(
        vid_root=str(vid_path.parent),
        vid=str(vid_path.name),
        start=0,
        num=total_frames)

    # padding
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

    transformed_imgs = test_transforms(padded_imgs)
    inputs = torch.from_numpy(transformed_imgs.transpose([3, 0, 1, 2]))

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(100)

    trained_model = "/home/myuser1/msasl_i3d_saved_models/nslt_100_031372_0.645594.pt"
    model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))
    model = nn.DataParallel(model)
    model.eval()

    #data = skvideo.io.vread("RGB/data/test/book0.mp4")
    #data = torch.from_numpy(data.transpose([3, 0, 1, 2]))
    #data.unsqueeze_(0)
    #print(data.shape)

    with torch.no_grad():
        #output = model(data.float())
        # output.to(device=torch.device('cpu'), dtype=float)
        output = model(inputs)

        prediction = torch.max(output, dim=2)[0]
        # out_labels = np.argsort(prediction.cpu().detach().numpy()[0])
        print(prediction[0])
        print(torch.argmax(prediction[0]).item())
        print(str(vid_path.name)[:4])
