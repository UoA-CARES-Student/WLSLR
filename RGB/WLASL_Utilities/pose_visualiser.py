import json
import os

import traceback
import mmcv
from datetime import datetime
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

wholebody_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/configs/wholebody/2d_kpt_sview_rgb_img' \
                   '/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288.py'
wholebody_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288' \
                       '-6e061c6a_20200922.pth'
wholebody_det_config = '/home/izzy/Documents/UoA/Sem_1_2022/P4P/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
wholebody_det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco' \
                           '/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

DATASET_BASE = "/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL-100"

def predict_and_visualise(model,
                          det_model,
                          input_vid_path,
                          # output_vid
                          ):
    input_vid = mmcv.VideoReader(input_vid_path)
    # fps = input_vid.fps
    # size = (input_vid.width, input_vid.height)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter("/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/pose/test_vis_output/" + output_vid,
    #                                fourcc,
    #                                fps,
    #                                size)

    offset = 0

    video_name = os.path.basename(input_vid_path)
    split_name = input_vid_path.split(os.sep)[-2]

    print(
        f"Now processing video {video_name} from the {split_name} directory")
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(input_vid)):
        path_dir = f"{DATASET_BASE}/{split_name}/data/pose_per_individual_videos/{os.path.splitext(video_name)[0]}"

        frame_num = frame_id - offset

        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        output_json = {}

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

        if len(pose_results) > 0:
            # get just wholebody prediction
            for _ in pose_results:
                while len(pose_results) > 1:
                    pose_results.pop(1)

            # cull unneeded keypoints:
            # Keypoints 1 - 13: upper body
            # Keypoints 92 - 112: right hand
            # Keypoints 113 - 133: left hand
            pose_keypoints_2d = []
            hand_left_keypoints_2d = []
            hand_right_keypoints_2d = []

            index = 1

            try:
                for _ in pose_results[0]['keypoints']:
                    if 13 < index < 92:
                        pose_results[0]['keypoints'][index - 1] = [0, 0, 0]
                    elif index <= 13:
                        pose_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][0]))
                        pose_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][1]))
                        pose_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][2]))
                    elif 92 <= index <= 112:
                        hand_left_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][0]))
                        hand_left_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][1]))
                        hand_left_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][2]))
                    else:
                        hand_right_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][0]))
                        hand_right_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][1]))
                        hand_right_keypoints_2d.append(float(pose_results[0]['keypoints'][index - 1][2]))
                    index += 1
            except Exception as e:
                traceback.print_exc()
                print(f"Error occurred on frame {frame_id}")
                with open('/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL_Utilities/log.txt', 'a') as logfile:
                    logfile.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    logfile.write(f" Error occurred on frame {frame_id}, video {os.path.splitext(video_name)[0]}, split {split_name}\n")
                    logfile.write(str(e))
                    logfile.write(traceback.format_exc())
                    logfile.write("--------------------------------\n")

            output_json["pose_keypoints_2d"] = pose_keypoints_2d
            output_json["hand_left_keypoints_2d"] = hand_left_keypoints_2d
            output_json["hand_right_keypoints_2d"] = hand_right_keypoints_2d

            with open(f"{path_dir}/image_{frame_num:05}_keypoints.json", "w") as outfile:
                json.dump(output_json, outfile)

        #     # show pose estimation results
        #     vis_result = vis_pose_result(
        #         model,
        #         cur_frame,
        #         pose_results,
        #         dataset=model.cfg.data.test.type,
        #         show=False)
        #     # reduce image size
        #     video_writer.write(vis_result)
        #
        # video_writer.release()

        else:
            offset += 1


if __name__ == "__main__":
    wholebody_model = init_pose_model(wholebody_config, wholebody_checkpoint)
    wholebody_det_model = init_detector(wholebody_det_config, wholebody_det_checkpoint)

    for split_dir in os.listdir(DATASET_BASE):
        files = os.listdir(os.path.join(DATASET_BASE, split_dir))

        for file in files:
            if file.endswith(".mp4"):
                predict_and_visualise(wholebody_model, wholebody_det_model, os.path.join(DATASET_BASE, split_dir, file))