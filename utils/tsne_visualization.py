#!/usr/bin/env python3
import os
import h5py
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # for headless environment
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE

def main():
  args = get_args()

  if not os.path.isdir(os.path.join(args.output_dir)):
    os.makedirs(os.path.join(args.output_dir))

  feat_h5 = h5py.File(args.feature_h5, 'r')
  videonames = list(feat_h5.keys())
  videonames_size = len(videonames)

  rng = random.SystemRandom()
  rng.shuffle(videonames)
  videonames = videonames[:args.sample_size]

  # Put features of sampled videonames on RAM
  features = []
  for videoname in videonames:
    features.append(feat_h5[videoname][:])
  features = np.vstack(features)
  features = np.reshape(features, (features.shape[0], -1))

  # find tsne coords for 2 dimensions
  tsne = TSNE(n_components=2)
  np.set_printoptions(suppress=True)
  Y = tsne.fit_transform(features)

  x_coords = Y[:, 0]
  y_coords = Y[:, 1]
  # display scatter plot
  fig = plt.figure(figsize=[100,100])
  plt.scatter(x_coords, y_coords)

  label_postfix = []
  for frame_ind in range(1,args.target_length+1):
    label_postfix.append('_{}'.format(frame_ind))
  label_postfix = np.array(label_postfix)[:,np.newaxis]
  label_postfix = np.repeat(label_postfix, args.sample_size,axis=1)
  label_postfix = np.transpose(label_postfix)
  label_postfix = np.hstack(label_postfix)

  point_labels = np.core.defchararray.add(
    np.repeat(videonames,args.target_length), 
    label_postfix
    )

  for label, x, y in zip(point_labels, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
  plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
  plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
  plt.show()
  plt.savefig(os.path.join(args.output_dir, 'figure.pdf'))
  plt.close(fig)

def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--feature-h5', dest='feature_h5',
    default='./VIAR_extracted_features.h5', 
    help='Input HDF5 file containing extracted ConvLSTM features.')
  parser.add_argument('--sample-size', dest='sample_size',
    type=int, default=100, 
    help='The number of videonames to be randomly sampled.')
  parser.add_argument('-o', '--output-dir', dest='output_dir',
    default='./output', help='Output directory for figure.')
  parser.add_argument('--target-legnth', dest='target_length',
    type=int, default=6,
    help='Target length of each video (Extracted number of frames per video).')

  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()