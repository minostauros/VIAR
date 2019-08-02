#!/usr/bin/env python3
import os
import argparse
import subprocess

def main():
  args = get_args()

  # Parameters
  video_ext = '.avi'

  videos = os.listdir(args.videos_dir)
  videos = [f for f in videos if f.endswith(video_ext) and not f.startswith('.')]

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  length = len(videos)
  for ind, video in enumerate(videos):
    print('{}/{}. Extracting PNGs from {}...'.format(ind+1, length, video))
    video_file = os.path.join(args.videos_dir, video)
    target_dir = os.path.join(args.output_dir, os.path.splitext(video)[0])
    if not os.path.isdir(target_dir):
      os.makedirs(target_dir)
    target_img_path = os.path.join(target_dir, 'image_%06d.png')
    target_fps_cmd = ''
    if args.target_fps is not None:
      target_fps_cmd = '-r {}'.format(args.target_fps)
    cmd = 'ffmpeg -i \"{}\" -vf scale={}:{} {} -sws_flags lanczos -q:v {} \"{}\"'.\
      format(video_file, args.target_width, args.target_height, target_fps_cmd, args.target_quality, target_img_path)
    subprocess.call(cmd, shell=True)

def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  # Input Directories and Files
  parser.add_argument('--videos-dir', dest='videos_dir',
    default='./dataset/NTU_RGB+D/RGBVideos/nturgb+d_rgb',
    help='Directory that contains videos')

  # Output Directories
  parser.add_argument('--output-dir', dest='output_dir',
    default='./dataset/NTU_RGB+D/RGBVideos/nturgb+d_rgb_png',
    help='Directory for outputs, extracted pngs.')

  # Parameters
  parser.add_argument('--target-width', dest='target_width',
    type=int, default=320, help='Target width of extracted image.')
  parser.add_argument('--target-height', dest='target_height',
    type=int, default=240, help='Target height of extracted image.')
  parser.add_argument('--target-fps', dest='target_fps',
    type=int, default=None, help='Target FPS of extracted images.')
  parser.add_argument('--target-quality', dest='target_quality',
    type=int, default=5, help='Target quality of extracted image.')

  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()
