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
input_height = 240
input_width = 320

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
      flow_text_to_h5(args, ind, videoname, length)
  else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.num_worker)(
      delayed(flow_text_to_h5)(args, ind, videoname, length) for ind, videoname in enumerate(videonames)
      )

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since):
  now = time.time()
  s = now - since
  return '%s' % (asMinutes(s))

def bin_ndarray(ndarray, new_shape, operation='sum'):
  """
  Bins an ndarray in all axes based on the target shape, by summing or
      averaging.

  Number of output dimensions must match number of input dimensions and 
      new axes must divide old ones.

  Example
  -------
  >>> m = np.arange(0,100,1).reshape((10,10))
  >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
  >>> print(n)

  [[ 22  30  38  46  54]
   [102 110 118 126 134]
   [182 190 198 206 214]
   [262 270 278 286 294]
   [342 350 358 366 374]]

  """
  operation = operation.lower()
  if not operation in ['sum', 'mean']:
    raise ValueError("Operation not supported.")
  if ndarray.ndim != len(new_shape):
    raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                         new_shape))
  compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                ndarray.shape)]
  flattened = [l for p in compression_pairs for l in p]
  ndarray = ndarray.reshape(flattened)
  for i in range(len(new_shape)):
    op = getattr(ndarray, operation)
    ndarray = op(-1*(i+1))
  return ndarray

def flow_text_to_h5(args, videoname_ind, videoname, length):
  start_time = time.time()
  video_dir = os.path.join(args.top_dir, videoname)
  files = os.listdir(video_dir)
  files = [os.path.join(video_dir, f) for f in files if f.startswith('3dflow_results') and f.endswith('.txt')]
  files = sorted(files)

  target_height = input_height
  target_width = input_width
  if args.patch_size > 0:
    target_height = input_height // args.patch_size
    target_width = input_width // args.patch_size

  outfile_path = os.path.join(args.output_dir, videoname + '_3dflow.h5')
  outfile = h5py.File(outfile_path, 'w')
  dset = outfile.create_dataset('flow', 
    (len(files),target_height,target_width,3), 
    maxshape=(len(files),target_height,target_width,3), 
    chunks=True, dtype='f4')

  for f_ind, f in enumerate(files):
    # read jpeg as binary and put into h5
    flowtxt = np.loadtxt(f, dtype='f8')
    flow = flowtxt[:,2:5] # remove x, y coordinate speicified in text files
    flow = flow.reshape((input_height,input_width,3)) # 3 for x,y,z in 3d flow
    flow = bin_ndarray(flow, 
      new_shape=(target_height,target_width,3), 
      operation='mean'
      )
    dset[f_ind,:] = flow[:]

  outfile.close()
  print('{}/{}. converting flows of {} to h5 done...{}'.format(
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
    default='./dataset/NTU_RGB+D/Extracted3DFlow',
    help='Top Directory with subdirectories containing extracted'
         ' 3d flow results in text files.')

  # Output Directories
  parser.add_argument('--output-dir', dest='output_dir',
    default='./dataset/NTU_RGB+D/Extracted3DFlowH5/',
    help='Directory for outputs, extracted 3D flows in HDF5 format.')

  # Parameters
  parser.add_argument('--patch-size', dest='patch_size',
    default=8, type=int,
    help='To make flow as low-dimensional signal, calculate mean of each '
         'non-overlapping (patch_size x patch_size) patch. Choose value'
         'that can divide both height and width.'
         'If 0, do not use patch and just save the whole thing.')

  # Parallelism
  parser.add_argument('--num-worker', dest='num_worker',
    type=int, default=0, 
    help='Number of parallel jobs for resizing. 0 for no parallelism.'
         ' Choose this number wisely for speed and I/O bottleneck tradeoff.')

  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()
