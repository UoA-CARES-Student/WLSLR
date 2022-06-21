import os

import ffmpeg
from moviepy.config import FFMPEG_BINARY
from moviepy.tools import subprocess_call


def compress_video(
        video_full_path: str,
        size_upper_bound: int,
        total_bitrate_lower_bound=11000,
        audio_bitrate=0,
        min_video_bitrate=100000,
        two_pass=True,
        filename_suffix='copy'):
    """
    Compress video file to max-supported size.

    Args:
        video_full_path: the video you want to compress.
        size_upper_bound: Max video size in KB.
        total_bitrate_lower_bound: Lower bound of (video + audio) bitrate.
        audio_bitrate: The audio bitrate.
        min_video_bitrate: Minimum video bitrate.
        two_pass: Set to True to enable two-pass calculation.
        filename_suffix: Add a suffix for new video.
    Return:
        output file name if compression successfully or False if unsuccessful.
    Side-effects:
        Creates a compressed copy of the video file and then deletes the orginal.

    Reference: https://gist.github.com/ESWZY/a420a308d3118f21274a0bc3a6feb1ff
    """
    filename, extension = os.path.splitext(video_full_path)
    extension = '.mp4'
    output_file_name = filename + filename_suffix + extension

    try:
        # Bitrate reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
        probe = ffmpeg.probe(video_full_path)

        # Video duration, in s.
        duration = float(probe['format']['duration'])

        # Target total bitrate, in bps.
        target_total_bitrate = (size_upper_bound * 1024 * 8) / (1.073741824 * duration)
        if target_total_bitrate < total_bitrate_lower_bound:
            print('Bitrate is extremely low! Setting bitrate to lowest possible.')
            target_total_bitrate = total_bitrate_lower_bound

        # Best min size, in kB.
        best_min_size = min_video_bitrate * (1.073741824 * duration) / (8 * 1024)
        if size_upper_bound < best_min_size:
            print('Quality not good! Recommended minimum size:', '{:,}'.format(int(best_min_size)), 'KB.')
            print('Setting upper bound to best minium size...')
            size_upper_bound = best_min_size

        # Target video bitrate, in bps.
        video_bitrate = target_total_bitrate - audio_bitrate
        if os.path.getsize(video_full_path) <= size_upper_bound * 1024:
            os.rename(video_full_path, filename + extension)
            return False

        # Compression
        i = ffmpeg.input(video_full_path)
        if two_pass:
            ffmpeg.output(i, '/dev/null' if os.path.exists('/dev/null') else 'NUL',
                          **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4'}
                          ).overwrite_output().run()
            ffmpeg.output(i, output_file_name,
                          **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}
                          ).overwrite_output().run()
        else:
            ffmpeg.output(i, output_file_name,
                          **{'c:v': 'libx264', 'b:v': video_bitrate, 'c:a': 'aac', 'b:a': audio_bitrate}
                          ).overwrite_output().run()

        # Check if amount of compression is sufficent
        if os.path.getsize(output_file_name) <= size_upper_bound * 1024:
            # Replace old video with newly compressed video
            os.remove(video_full_path)
            os.rename(output_file_name, filename + extension)
            output_file_name = filename + extension
            return output_file_name
        elif os.path.getsize(output_file_name) < os.path.getsize(video_full_path):  # Do it again
            # Replace old video with newly compressed video
            os.remove(video_full_path)
            os.rename(output_file_name, filename + extension)
            output_file_name = filename + extension
            return compress_video(output_file_name, size_upper_bound)
        else:
            return False

    except FileNotFoundError as e:
        print('You do not have ffmpeg installed!', e)
        print('You can install ffmpeg by reading https://github.com/kkroening/ffmpeg-python/issues/251')
        return False


def extract_video_subclip(
        inputfile: str,
        start_time: float,
        end_time: float,
        outputfile: str,
        logger="bar"):
    """
    Makes a new video file playing video file between two times.

    Args:
        inputfile: Path to the file from which the subclip will be extracted.
        start_time: Moment of the input clip that marks the start of the produced subclip (seconds).
        end_time: Moment of the input clip that marks the end of the produced subclip (seconds).
        outputfile: Path to the output file (e.g. "../video_name.mp4").
    Returns:
        Saves a subclip of the orginal video at the output location.

    Reference: https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_tools.py#L27
    """
    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-ss",
        "%0.2f" % start_time,
        "-i",
        inputfile,
        "-t",
        "%0.2f" % (end_time - start_time),
        "-map",
        "0",
        "-vcodec",
        "copy",
        "-acodec",
        "copy",
        "-copyts",
        outputfile,
    ]
    subprocess_call(cmd, logger=logger)
