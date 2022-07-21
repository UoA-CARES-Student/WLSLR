import torch
import torch.nn as nn
import numpy as np
import torchvision
import skvideo
skvideo.setFFmpegPath("C:/Users/isabe/Downloads/ffmpeg-5.0.1-full_build-shared/bin")
import skvideo.io
from pytorch_i3d import InceptionI3d

if __name__ == "__main__":
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(100)

    model.load_state_dict(torch.load("RGB/I3D/archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt",
        map_location=torch.device('cpu')))
    model = nn.DataParallel(model)
    model.eval()

    data = skvideo.io.vread("RGB/data/test/book0.mp4")
    data = torch.from_numpy(data.transpose([3, 0, 1, 2]))
    data.unsqueeze_(0)
    print(data.shape)

    with torch.no_grad():
        output = model(data.float())
        # output.to(device=torch.device('cpu'), dtype=float)

        prediction = torch.max(output, dim=2)[0]
        # out_labels = np.argsort(prediction.cpu().detach().numpy()[0])
        print(prediction[0])
        print(torch.argmax(prediction[0]).item())
