import cv2
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

DATASET_DIR = "/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL2000"
POSE_INFO_DIR = DATASET_DIR + "/data/pose_per_individual_videos"

pose_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage1.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

pose_model = init_pose_model(pose_config, pose_checkpoint)
det_model = init_detector(det_config, det_checkpoint)

vid = "/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL-100/test/0000000.mp4"

# inference detection
mmdet_results = inference_detector(det_model, vid)

# extract person (COCO_ID=1) bounding boxes from the detection results
person_results = process_mmdet_results(mmdet_results, cat_id=1)

pose_results, returned_outputs = inference_top_down_pose_model(
    pose_model,
    vid,
    person_results,
    bbox_thr=0.3,
    format='xyxy',
    dataset=pose_model.cfg.data.test.type)

# show pose estimation results
vis_result = vis_pose_result(
    pose_model,
    vid,
    pose_results,
    dataset=pose_model.cfg.data.test.type,
    show=False)
# reduce image size
vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)

from IPython.display import Image, display
import tempfile
import os.path as osp
with tempfile.TemporaryDirectory() as tmpdir:
    file_name = osp.join(tmpdir, 'pose_results.png')
    cv2.imwrite(file_name, vis_result)
    display(Image(file_name))