import cv2
import mmcv
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

pose_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage1.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

pose_model = init_pose_model(pose_config, pose_checkpoint)
det_model = init_detector(det_config, det_checkpoint)

vid = mmcv.VideoReader("/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL-100/train/0000050.mp4")

fps = vid.fps
size = (vid.width, vid.height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/pose/test_vis_output/vis_test.mp4", fourcc, fps, size)

for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(vid)):
    # inference detection
    mmdet_results = inference_detector(det_model, cur_frame)

    # extract person (COCO_ID=1) bounding boxes from the detection results
    person_results = process_mmdet_results(mmdet_results, cat_id=1)

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        cur_frame,
        person_results,
        bbox_thr=0.3,
        format='xyxy',
        dataset=pose_model.cfg.data.test.type)

    # show pose estimation results
    vis_result = vis_pose_result(
        pose_model,
        cur_frame,
        pose_results,
        dataset=pose_model.cfg.data.test.type,
        show=False)
    # reduce image size
    videoWriter.write(vis_result)

videoWriter.release()