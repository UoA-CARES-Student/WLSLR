import http.client
import json
import os
import urllib.error

from moviepy.video.fx.crop import crop
from moviepy.video.io.VideoFileClip import VideoFileClip

import video_utilities

import pytube.exceptions
from pytube import YouTube


def download_set(file, classes):
    dir_name = os.path.join("MS-ASL-{}".format(classes), file.split('_')[1].split('.')[0])

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    num_videos = 0
    skipped_videos = 0
    num_retries = 3;

    f = open(file)
    dataset = json.load(f)
    for data in dataset:
        if data['label'] == 689:
            complete = False
            while not complete:
                try:
                    index = 0
                    for root, _dir, files in os.walk(dir_name):
                        for file in files:
                            if "{:04d}".format(data['label']) == file[0:4]:
                                index += 1
                    file_name = "{:04d}{:03d}".format(data['label'], index)

                    print("Now downloading {}".format(data['url']))
                    YouTube(data['url']).streams \
                        .get_highest_resolution() \
                        .download(output_path=dir_name,
                                  filename=file_name + "_init",
                                  max_retries=10,
                                  timeout=600)
                    num_videos += 1
                    print("Downloaded! Now preprocessing...")
                    clip = VideoFileClip(os.path.join(dir_name, file_name) + "_init", audio=False). \
                        subclip(t_start=data['start_time'], t_end=data['end_time'])
                    crop(clip,
                         y1=int(clip.h * data['box'][0]),
                         x1=int(clip.w * data['box'][1]),
                         y2=int(clip.h * data['box'][2]),
                         x2=int(clip.w * data['box'][3]))
                    clip.write_videofile(os.path.join(dir_name, file_name) + ".mp4", codec='libx264')
                    os.remove(os.path.join(dir_name, file_name)+"_init")
                    print("Done!")
                    complete = True
                except (pytube.exceptions.VideoPrivate,
                        pytube.exceptions.VideoUnavailable,
                        pytube.exceptions.VideoRegionBlocked):
                    print('Video is private or unavailable. Skipping...')
                    complete = True
                    num_videos += 1
                    skipped_videos += 1
                except (urllib.error.URLError, http.client.IncompleteRead):
                    print("Network error. Retrying...")
                    num_retries -= 1
                    if num_retries > 0:
                        complete = False
                    else:
                        complete = True
                        print('Unknown error. Skipping...')
                        num_retries = 3
                except pytube.exceptions.MaxRetriesExceeded:
                    print('Max retries exceeded. Skipping...')
                    complete = True
                    num_videos += 1
                    skipped_videos += 1
                except pytube.exceptions.PytubeError:
                    print('Some unknown error occurred. Skipping...')
                    complete = True
                    num_videos += 1
                    skipped_videos += 1
    return [num_videos, skipped_videos]


if __name__ == '__main__':
    subsets = [1000]

    for subset in subsets:
        num_vids = download_set('MS-ASL/MSASL_train.json', subset)
        print("Complete! {} videos traversed, with {} videos skipped. {} videos downloaded overall."
              .format(num_vids[0], num_vids[1], num_vids[0] - num_vids[1]))

        num_vids = download_set('MS-ASL/MSASL_test.json', subset)
        print("Complete! {} videos traversed, with {} videos skipped. {} videos downloaded overall."
              .format(num_vids[0], num_vids[1], num_vids[0] - num_vids[1]))

        num_vids = download_set('MS-ASL/MSASL_val.json', subset)
        print("Complete! {} videos traversed, with {} videos skipped. {} videos downloaded overall."
              .format(num_vids[0], num_vids[1], num_vids[0] - num_vids[1]))
