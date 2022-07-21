import cv2
import pathlib
import numpy as np

import torch
import torch.nn as nn
from pytorch_i3d import InceptionI3d
from torchvision import transforms

# import skvideo
# skvideo.setFFmpegPath("C:/Users/isabe/Downloads/ffmpeg-5.0.1-full_build-shared/bin")
# import skvideo.io

import videotransforms

import dataset.nslt_dataset as nslt_dataset


def predict_single_video(
    vid_path: str,
    trained_model: str = "C:/Users/isabe/Documents/UoA/Sem 1 2022/P4P/WLSLR/RGB/data/models/nslt_100_031372_0.645594.pt"
) -> None:
    vid_path = pathlib.Path(vid_path)
    vidcap = cv2.VideoCapture(str(vid_path))
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    imgs = nslt_dataset.load_rgb_frames_from_video(
        vid_root=str(vid_path.parent),
        vid=str(vid_path.stem),
        start=int(0),
        num=int(total_frames))

    # Padding
    padded_imgs, _ = nslt_dataset.pad(
        imgs=imgs,
        label=None,
        total_frames=total_frames)

    # Transforms
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    transformed_imgs = test_transforms(padded_imgs)
    inputs = torch.from_numpy(transformed_imgs.transpose([3, 0, 1, 2]))
    inputs.unsqueeze_(0)

    # Create model
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(100)   # Current set to 100 classes

    # Load trained weights
    model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))
    model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        output = model(inputs)

        prediction = torch.max(output, dim=2)[0]
        print(prediction[0])
        print("Prediction: ", torch.argmax(prediction[0]).item())

        probs = nn.functional.softmax(prediction, dim=1)
        print("Confidence: {}".format(torch.topk(probs, 5)))


if __name__ == "__main__":
    predict_single_video('C:/Users/isabe/Documents/UoA/Sem 1 2022/P4P/WLSLR/RGB/data/test/nosign0.mp4')
