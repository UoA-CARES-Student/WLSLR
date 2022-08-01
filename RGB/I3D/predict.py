import queue
import cv2
import pathlib
import numpy as np

import torch
import torch.nn as nn
from pytorch_i3d import InceptionI3d
from torchvision import transforms

import json
import videotransforms
import dataset.nslt_dataset as nslt_dataset

class_glosses = json.load(open("C:/Users/isabe/Documents/UoA/Sem 1 2022/P4P/WLSLR/RGB/MSASL_Utilies/MS-ASL/MSASL_classes.json"))

def predict_from_mp4(vid_path: str, pred: queue.Queue,) -> None:
    vid_path = pathlib.Path(vid_path)
    vidcap = cv2.VideoCapture(str(vid_path))
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    imgs = nslt_dataset.load_rgb_frames_from_video(
        vid_root=str(vid_path.parent),
        vid=str(vid_path.stem),
        start=int(0),
        num=int(total_frames))
    predict_from_array(imgs=imgs, pred=pred)
    

def predict_from_array(
    imgs: np.ndarray,
    pred: queue.Queue,
    trained_model: str = "C:/Users/isabe/Documents/UoA/Sem 1 2022/P4P/WLSLR/RGB/data/models/nslt_100_031372_0.645594.pt"
):
    # Padding
    padded_imgs, _ = nslt_dataset.pad(
        imgs=imgs,
        label=None,
        total_frames=imgs.shape[0])

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
        output = model(inputs.float())

        prediction = torch.max(output, dim=2)[0]
        print(prediction[0])
        # print("Prediction: ", torch.argmax(prediction[0]).item())
        print("Prediction:", class_glosses[torch.argmax(prediction[0]).item()])

        probs = nn.functional.softmax(prediction, dim=1)
        print("Confidence: {}".format(torch.max(probs)))

    pred.put( (class_glosses[torch.argmax(prediction[0]).item()], torch.max(probs)) )
   


if __name__ == "__main__":
    predict_from_mp4('C:/Users/isabe/Documents/UoA/Sem 1 2022/P4P/WLSLR/RGB/data/MS-ASL-100/val/0032003.mp4')
