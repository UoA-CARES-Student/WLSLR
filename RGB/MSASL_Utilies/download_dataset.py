import json
import os
import urllib.error

import pytube.exceptions
from pytube import YouTube


def download_set(file, classes):
    dir_name = os.path.join("MS-ASL-{}".format(classes), file.split('_')[1].split('.')[0])

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    num_videos = 0
    skipped_videos = 0

    f = open(file)
    dataset = json.load(f)
    for data in dataset:
        if data['label'] < int(classes):
            complete = False
            while not complete:
                try:
                    index = 0
                    for root, _dir, files in os.walk(dir_name):
                        for file in files:
                            if data['clean_text'] in file:
                                index += 1

                    print("Now downloading {}".format(data['url']))
                    YouTube(data['url']).streams \
                        .get_highest_resolution() \
                        .download(output_path=dir_name,
                                  filename="{}_{}".format(data['clean_text'], index),
                                  max_retries=10)
                    num_videos += 1
                    print("Done!")
                    complete = True
                except (pytube.exceptions.VideoPrivate,
                        pytube.exceptions.VideoUnavailable,
                        pytube.exceptions.VideoRegionBlocked):
                    print('Video is private or unavailable. Skipping...')
                    complete = True
                    num_videos += 1
                    skipped_videos += 1
                except urllib.error.URLError:
                    print("Network error. Retrying...")
                    complete = False
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
    subsets = [100, 500, 1000]

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
