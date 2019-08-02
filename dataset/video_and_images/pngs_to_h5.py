#!/usr/bin/env python3
import os
import h5py
import json
import time
import math
import argparse
import numpy as np
import collections

# Parameters
dt = h5py.special_dtype(vlen=np.uint8)

def main():
  args = get_args()

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  with open(args.videoname_json_path, 'r') as fp:
    meta = json.load(fp, object_pairs_hook=collections.OrderedDict)

  videonames = meta['videonames']
  videonames = sorted(videonames)
  
  length = len(videonames)

  if args.num_worker < 1:
    for ind, videoname in enumerate(videonames):
      pngs_to_h5(args, ind, videoname, length)
  else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.num_worker)(
      delayed(pngs_to_h5)(args, ind, videoname, length) for ind, videoname in enumerate(videonames)
      )

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since):
  now = time.time()
  s = now - since
  return '%s' % (asMinutes(s))

def pngs_to_h5(args, videoname_ind, videoname, length):
  start_time = time.time()
  video_dir = os.path.join(args.top_dir, videoname + '_rgb')
  files = os.listdir(video_dir)
  files = [os.path.join(video_dir, f) for f in files if f.startswith('image') and f.endswith('.png')]
  files = sorted(files)

  outfile_path = os.path.join(args.output_dir, videoname + '_pngs.h5')
  outfile = h5py.File(outfile_path, 'w')
  dset = outfile.create_dataset('pngs', (len(files),), 
    maxshape=(len(files),), chunks=True, dtype=dt)

  for f_ind, f in enumerate(files):
    # read png as binary and put into h5
    png = open(f, 'rb')
    binary_data = png.read()
    dset[f_ind] = np.fromstring(binary_data, dtype=np.uint8)
    png.close()

  outfile.close()
  print('{}/{}. converting pngs of {} to h5 done...took {}'.format(
    videoname_ind+1, length, videoname, timeSince(start_time)) )

def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  # Input Directories and Files
  parser.add_argument('--videoname-json-path', dest='videoname_json_path',
    default='./dataset/NTU_RGB+D/ntu_rgbd_videonames.json',
    help='Path to the JSON file with videonames of NTU RGB+D dataset videos.')
  parser.add_argument('--top-dir', dest='top_dir',
    default='./dataset/NTU_RGB+D/RGBVideos/nturgb+d_rgb_png',
    help='Top Directory with subdirectories containing extracted'
         ' 3d flow results in text files.')

  # Output Directories
  parser.add_argument('--output-dir', dest='output_dir',
    default='./dataset/NTU_RGB+D/nturgb+d_rgb_pngs_320x240_lanczos_h5/',
    help='Directory for outputs, extracted 3D flows in HDF5 format.')

  # Parallelism
  parser.add_argument('--num-worker', dest='num_worker',
    type=int, default=0, 
    help='Number of parallel jobs for resizing. 0 for no parallelism.'
         ' Choose this number wisely for speed and I/O bottleneck tradeoff.')

  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()
