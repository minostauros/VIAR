#!/usr/bin/env python3
'''
Extract 3D Flow using PD-Flow (https://github.com/minostauros/PD-Flow)
This code requires pre-built binary of PD-Flow, path specified with command line.
Input images are all required to be size of 320x240.
'''
import os
import time
import math
import shutil
import argparse
import datetime
import subprocess

# Parameters
image_ext = '.png' # png for lossless image
done = []

def main():
  args = get_args()

  videonames = next(os.walk(args.depth_dir))[1]
  videonames = sorted([d for d in videonames if not d.startswith('.')])

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  length = len(videonames)

  if args.num_worker < 1:
    for ind, videoname in enumerate(videonames):
      extract_flow(args, ind, videoname, length)
  else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.num_worker)(
      delayed(extract_flow)(args, ind, videoname, length) for ind, videoname in enumerate(videonames)
      )

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since):
  now = time.time()
  s = now - since
  return '%s' % (asMinutes(s))

def extract_flow(args, ind, videoname, length):
  if videoname in done:
    print("Skipping: {}/{}. {}".format(ind+1, length, videoname))
    return
  starttime = time.time() 
  rgbs_dir = os.path.join(args.rgb_dir, videoname + '_rgb')
  rgbs = os.listdir(rgbs_dir)
  rgbs = sorted([f for f in rgbs if f.endswith(image_ext) and not f.startswith('.')])
  depths_dir = os.path.join(args.depth_dir, videoname)
  depths = os.listdir(depths_dir)
  depths = sorted([f for f in depths if f.endswith(image_ext) and not f.startswith('.')])
  if len(rgbs) != len(depths):
    raise RuntimeError('The number of RGB images and the number of depths '
                       'images do not match ({}).'.format(videoname))
  target_dir = os.path.join(args.output_dir, videoname)
  if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
  else:
    shutil.rmtree(target_dir)
    print('Deleted existing target directory {} and creating a new one.'.format(target_dir))
    os.makedirs(target_dir)
  target_path_and_prefix = os.path.join(target_dir, '3dflow')
  cmd = ('CUDA_VISIBLE_DEVICES={} {} --rows 240 --idir {} ' 
        '--zdir {} --out {} --no-show'.format(
          args.gpus[ind % len(args.gpus)], args.pd_flow_path, rgbs_dir, 
          depths_dir, target_path_and_prefix))
  try:
    result = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, timeout=300)
    if result == 0:
      print('{}/{}. Extracted 3D flow from {}...took {}'.format(
        ind+1, length, videoname, timeSince(starttime)) )
    else:
      print('Error: {}/{}. Failed to extract 3D flow from {}...took {}'.format(
        ind+1, length, videoname, timeSince(starttime)) )
  except subprocess.TimeoutExpired:
    print('Error: {}/{}. Timeout (300s) fail to extract 3D flow from {}...'.format(
      ind+1, length, videoname) )

def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  # Input Directories and Files
  parser.add_argument('--pd-flow-path', dest='pd_flow_path',
    default='Scene-Flow-Impair',
    help='Path to the binary executable of PD-Flow\'s Scene-Flow-Impair')
  parser.add_argument('--rgb-dir', dest='rgb_dir',
    default='./dataset/NTU_RGB+D/RGBVideos/nturgb+d_rgb_png_grayscale',
    help='Directory that contains directories of rgb images.')
  parser.add_argument('--depth-dir', dest='depth_dir',
    default='./dataset/NTU_RGB+D/MaskedDepthMaps_resized/nturgb+d_depth_masked',
    help='Directory that contains directories of depth images.')

  # Output Directories
  parser.add_argument('--output-dir', dest='output_dir',
    default='./dataset/NTU_RGB+D/Extracted3DFlow/',
    help='Directory for outputs, extracted 3D flows.')

  # Parallelism
  parser.add_argument('--num-worker', dest='num_worker',
    type=int, default=0, 
    help='Number of parallel jobs for resizing. 0 for no parallelism')
  parser.add_argument('--gpus', dest='gpus',
    default=[0], nargs="*",
    help='List of GPU numbers to be used.')

  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()
