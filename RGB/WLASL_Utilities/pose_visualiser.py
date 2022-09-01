import json

import cv2
import mmcv
import numpy
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

pose_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco' \
              '/hrnet_w48_coco_256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
pose_det_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
pose_det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco' \
                      '/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

hand_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap' \
              '/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256.py'
hand_checkpoint = 'https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_coco_wholebody_hand_256x256' \
                  '-1c028db7_20210908.pth'
hand_det_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/demo/mmdetection_cfg' \
                  '/cascade_rcnn_x101_64x4d_fpn_1class.py'
hand_det_checkpoint = 'https://download.openmmlab.com/mmpose/mmdet_pretrained' \
                      '/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'

wholebody_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/configs/wholebody/2d_kpt_sview_rgb_img' \
                   '/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288.py'
wholebody_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288' \
                       '-6e061c6a_20200922.pth'
wholebody_det_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
wholebody_det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco' \
                           '/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


def predict_and_visualise(model, det_model, input_vid, output_vid):
    fps = input_vid.fps
    size = (input_vid.width, input_vid.height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/pose/test_vis_output/" + output_vid,
                                   fourcc,
                                   fps,
                                   size)

    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(input_vid)):
        # inference detection
        mmdet_results = inference_detector(det_model, cur_frame)

        # extract person (COCO_ID=1) bounding boxes from the detection results
        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        pose_results, returned_outputs = inference_top_down_pose_model(
            model,
            cur_frame,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=model.cfg.data.test.type)

        for _ in pose_results:
            while len(pose_results) > 1:
                pose_results.pop(1)

        index = 1
        for _ in pose_results[0]['keypoints']:
            if 13 < index < 92:
                pose_results[0]['keypoints'][index - 1] = [0, 0, 0]
            index += 1

        if frame_id == 1:
            index = 0
            for _ in pose_results:
                pose_results[index].update({'bbox': numpy.ndarray.tolist(pose_results[index].get('bbox'))})
                pose_results[index].update({'keypoints': numpy.ndarray.tolist(pose_results[index].get('keypoints'))})
                index += 1

            with open("pose_test.json", "w") as outfile:
                json.dump(pose_results, outfile)

        # show pose estimation results
        vis_result = vis_pose_result(
            model,
            cur_frame,
            pose_results,
            dataset=model.cfg.data.test.type,
            show=False)
        # reduce image size
        video_writer.write(vis_result)

    video_writer.release()


if __name__ == "__main__":
    # pose_model = init_pose_model(pose_config, pose_checkpoint)
    # pose_det_model = init_detector(pose_det_config, pose_det_checkpoint)
    #
    # hand_model = init_pose_model(hand_config, hand_checkpoint)
    # hand_det_model = init_detector(hand_det_config, hand_det_checkpoint)

    wholebody_model = init_pose_model(wholebody_config, wholebody_checkpoint)
    wholebody_det_model = init_detector(wholebody_det_config, wholebody_det_checkpoint)

    # vid = mmcv.VideoReader("/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/MS-ASL-100/train/0002008.mp4")
    vid = mmcv.VideoReader("/home/izzy/Downloads/video-14611836-56f0-44ed-9c4e-44576ccb6572-1662074039.mp4")

    # predict_and_visualise(pose_model, pose_det_model, vid, "pose_test.mp4")
    # predict_and_visualise(hand_model, hand_det_model, vid, "hand_test.mp4")
    predict_and_visualise(wholebody_model, wholebody_det_model, vid, "wholebody_test.mp4")
