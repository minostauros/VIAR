#!/usr/bin/env python3
"""
  View-Invariant Action Representations, referenced from
J. Li, Y. Wong, Q. Zhao, and M. S. Kankanhalli, “Unsupervised Learning of 
View-invariant Action Representations.," NeurIPS, 2018.
"""
import os
import sys
import numpy as np

import json
import time
import argparse
import operator
from functools import reduce

import torch
import torch.nn as nn

from networks import CNN, Encoder, CrossViewDecoder, \
                     ReconstructionDecoder, ViewClassifier

sys.path.append('..')
from dataloader.NTURGBDwithFlowLoader import NTURGBDwithFlowLoader
from utils.utils import setCheckpointFileDict, testIters, trainIters

RGB_INPUT_SHAPE = (3,224,224)
DEPTH_INPUT_SHAPE = (1,224,224)
FLOW_SHAPE = (3,28,28)
ALL_MODELS = ['encodercnn', 'encoder','crossviewdecoder','crossviewdecodercnn',
              'reconstructiondecoder','viewclassifier']
LOG_PREFIX = 'VIAR'

def main():
  args = get_args()

  margs = {}
  margs['json_file'] = os.path.join(args.ntu_dir, 'ntu_rgbd_videonames.min.json')
  margs['label_file'] = os.path.join(args.ntu_dir, 'ntu_rgbd_action_labels.txt')
  margs['flow_h5_dir'] = os.path.join(args.ntu_dir, 'Extracted3DFlowH5')
  margs['rgb_h5_dir'] = os.path.join(args.ntu_dir, 'nturgb+d_rgb_pngs_320x240_lanczos_h5')
  margs['depth_h5_dir'] = os.path.join(args.ntu_dir, 'MaskedDepthMaps_320x240_h5')

  # Use cuda device if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  margs['device'] = device

  checkpoint_files = setCheckpointFileDict(ALL_MODELS, args.checkpoint_files)

  models = build_models(args, device=device)

  if args.for_what == 'train':
    main_train(checkpoint_files, args, margs, models)

  elif args.for_what == 'test':
    main_test(checkpoint_files, args, margs, models)

  else:
    raise NotImplementedError(
      'Given "{}" mode is not implemented'.format(args.for_what)
      )

def build_models(args, device='cuda'):
  models = {}
  
  models['encodercnn'] = CNN(
    input_shape=RGB_INPUT_SHAPE, model_name=args.encoder_cnn_model).to(device)

  models['encoder'] = Encoder(
    input_shape=models['encodercnn'].out_size, encoder_block='convbilstm', 
    hidden_size=args.encoder_hid_size).to(device)

  models['crossviewdecodercnn'] = CNN(
    input_shape=DEPTH_INPUT_SHAPE, model_name=args.encoder_cnn_model, 
    input_channel=1).to(device)

  crossviewdecoder_in_size = list(models['crossviewdecodercnn'].out_size)
  crossviewdecoder_in_size[0] = crossviewdecoder_in_size[0] * 3
  crossviewdecoder_in_size = torch.Size(crossviewdecoder_in_size)
  models['crossviewdecoder'] = CrossViewDecoder(
    input_shape=crossviewdecoder_in_size).to(device)

  models['reconstructiondecoder'] = ReconstructionDecoder(
    input_shape=models['encoder'].out_size[1:]).to(device)

  models['viewclassifier'] = ViewClassifier(
    input_size=reduce(operator.mul, models['encoder'].out_size[1:]), 
    num_classes=5, 
    reverse=(not args.disable_grl)).to(device)

  return models

def main_train(checkpoint_files, args, margs, models):
  train_loader, train_dataset = NTURGBDwithFlowLoader(
    json_file=margs['json_file'], 
    label_file=margs['label_file'], 
    rgb_h5_dir=margs['rgb_h5_dir'], 
    depth_h5_dir=margs['depth_h5_dir'], 
    flow_h5_dir=margs['flow_h5_dir'], 
    target_length=args.target_length, 
    subset='train', 
    visual_transform=args.visual_transform, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.num_workers, 
    pin_memory=True
    ) 

  val_loader, val_dataset = NTURGBDwithFlowLoader(
    json_file=margs['json_file'], 
    label_file=margs['label_file'], 
    rgb_h5_dir=margs['rgb_h5_dir'], 
    depth_h5_dir=margs['depth_h5_dir'], 
    flow_h5_dir=margs['flow_h5_dir'], 
    target_length=args.target_length, 
    subset='test', 
    visual_transform=args.visual_transform, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.num_workers, 
    pin_memory=True
    ) 

  trainIters(run, args.target_modules, 
             train_loader, train_dataset, val_loader, val_dataset,
             models=models, all_models=ALL_MODELS, log_prefix=LOG_PREFIX, 
             checkpoint_files=checkpoint_files, save_dir=args.save_dir, 
             args=args, device=margs['device'])

def main_test(checkpoint_files, args, margs, models):
  test_loader, test_dataset = NTURGBDwithFlowLoader(
    json_file=margs['json_file'], 
    label_file=margs['label_file'], 
    rgb_h5_dir=margs['rgb_h5_dir'], 
    depth_h5_dir=margs['depth_h5_dir'], 
    flow_h5_dir=margs['flow_h5_dir'], 
    target_length=args.target_length, 
    subset='test', 
    visual_transform=args.visual_transform, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=args.num_workers, 
    pin_memory=True
    ) 

  testIters(run, test_loader, test_dataset, models=models, 
            checkpoint_files=checkpoint_files, args=args, device=margs['device'])

def run(split, sample, models, target_modules=[], device='cuda',
        optimizers=None, criterions=None, args=None):
  result = {}
  result['logs'] = {}
  result['output'] = {}
  if split == 'train':
    set_grad = True
    for m in models:
      if m in target_modules:
        models[m].train()
        optimizers[m].zero_grad()
      else:
        models[m].eval()
  else:
    set_grad = False
    for m in models:
      models[m].eval()

  batch_size = len(sample['videoname'])
  target_length = len(sample['rgbs'][0])

  # Encoder
  rgb_input = sample['rgbs'].view(
    (batch_size*target_length,) + RGB_INPUT_SHAPE
    ).to(device)
  encodercnn_output = models['encodercnn'](rgb_input)
  encodercnn_output = encodercnn_output.view(
    (batch_size, target_length) + models['encodercnn'].out_size )
  encoder_output, _ = models['encoder'](encodercnn_output) # (batch, seq_len, c, h, w)
  if split == 'test':
    result['output']['encoder_output'] = encoder_output
  encoder_output = encoder_output.contiguous().view(
    (batch_size*target_length,) + models['encoder'].out_size[1:] )

  # CrossViewDecoder
  otherview_depth_input = sample['otherview_depths'].view(
    (batch_size*target_length,) + DEPTH_INPUT_SHAPE
    ).to(device)
  crossviewcnn_output = models['crossviewdecodercnn'](otherview_depth_input)
  crossview_output = models['crossviewdecoder'](crossviewcnn_output, encoder_output)
  crossview_output = crossview_output.view(
    (batch_size, target_length) + models['crossviewdecoder'].out_size )

  # ReconstructionDecoder
  reconstruct_output = models['reconstructiondecoder'](encoder_output)
  reconstruct_output = reconstruct_output.view(
    (batch_size, target_length) + models['reconstructiondecoder'].out_size )

  # ViewClassifier
  viewclassify_output = models['viewclassifier'](
    encoder_output.view(batch_size*target_length,-1) 
    )
  viewclassify_output = viewclassify_output.view(
    (batch_size, target_length) + (models['viewclassifier'].num_classes,) )

  if split in ['train', 'validate']:
    if set_grad:
      if 'encodercnn' in target_modules: optimizers['encodercnn'].zero_grad()
      if 'encoder' in target_modules: optimizers['encoder'].zero_grad()
      if 'crossviewdecodercnn' in target_modules: optimizers['crossviewdecodercnn'].zero_grad()
      if 'crossviewdecoder' in target_modules: optimizers['crossviewdecoder'].zero_grad()
      if 'reconstructiondecoder' in target_modules: optimizers['reconstructiondecoder'].zero_grad()
      if 'viewclassifier' in target_modules: optimizers['viewclassifier'].zero_grad()
    total_loss = 0
    crossview_loss = criterions['crossview'](crossview_output, sample['otherview_flows'].to(device))
    reconstruct_loss = criterions['reconstruct'](reconstruct_output, sample['flows'].to(device))
    viewclassify_loss = criterions['viewclassify'](viewclassify_output, sample['view_id'].long().to(device))
    total_loss += (crossview_loss + 0.5 * reconstruct_loss + 0.05 * viewclassify_loss)
    if set_grad and total_loss != 0:
      total_loss.backward()
      if 'encodercnn' in target_modules: optimizers['encodercnn'].step()
      if 'encoder' in target_modules: optimizers['encoder'].step()
      if 'crossviewdecodercnn' in target_modules: optimizers['crossviewdecodercnn'].step()
      if 'crossviewdecoder' in target_modules: optimizers['crossviewdecoder'].step()
      if 'reconstructiondecoder' in target_modules: optimizers['reconstructiondecoder'].step()
      if 'viewclassifier' in target_modules: optimizers['viewclassifier'].step()

    result['logs']['loss'] = total_loss.item() if total_loss > 0 else 0
    result['logs']['crossview_loss'] = crossview_loss.item() if crossview_loss > 0 else 0
    result['logs']['reconstruct_loss'] = reconstruct_loss.item() if reconstruct_loss > 0 else 0
    result['logs']['viewclassify_loss'] = viewclassify_loss.item() if viewclassify_loss > 0 else 0

  return result

def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  # What To Do
  parser.add_argument('--train', dest='for_what', 
    action='store_const', const='train', default='train', help='Train')
  parser.add_argument('--test', dest='for_what', 
    action='store_const', const='test', help='Test')
  parser.add_argument('--target-modules', dest='target_modules', 
    default=ALL_MODELS, nargs="*", choices=ALL_MODELS, 
    help='Modules to train or test')

  # Input Directories and Files
  parser.add_argument('--ntu-dir', dest='ntu_dir',
    default='./dataset/NTU_RGB+D_processed/',
    help='Directory that contains json, labels, rgb_h5_dir, depth_h5_dir, and'
         'flow_h5_dir directories')
  parser.add_argument('--checkpoint-files', default='{ }',
    type=json.loads, 
    help='JSON string to indicate checkpoint file for each module. '
         'Beside module names like encoder, and viewclassifier, you can use '
         'special name "else", and "all".'
         'Example: {"encoder": "encodercheck.tar", "else": "allcheck.tar"}')

  # Output Directories
  parser.add_argument('--save-dir', dest='save_dir', default='./VIAR',
    help='Directory to save checkpoints and logs')
  parser.add_argument('--unique-name', dest='unique_name',
    default=None, help='Uniquely names directory within save_dir')
  parser.add_argument('--output-dir', dest='output_dir',
    default='./', help='Output directory for outputs, e.g. extracted features.')

  # Networks
  parser.add_argument('--encoder-cnn-model', dest='encoder_cnn_model',
    default='resnet18', help='Choose between options, for CNN inside Encoder',
    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
  parser.add_argument('--encoder-hid-size', dest='encoder_hid_size',
    type=int, default=64, help='Hidden size of ConvBiLSTM inside Encoder')
  parser.add_argument('--disable-grl', dest='disable_grl',
    default=False, action='store_true',
    help='Disable Gradient Reversal Layer inside ViewClassifier')

  # Training Parameters
  parser.add_argument('--batch-size', dest='batch_size',
    type=int, default=1, help='Input minibatch size')
  parser.add_argument('--learning-rate', dest='learning_rate',
    type=float, default=1e-5, help='Learning rate for training')
  parser.add_argument('--val-every-iter', dest='val_every_iter',
    type=int, default=None, 
    help='Run validation every val_every_iter iterations. If None, validation '
         'is done after training set iteration.')
  parser.add_argument('--val-size', dest='val_size',
    type=int, default=None, 
    help='Size of minibatches for each run of validation. If None, all samples'
         ' will be used for validation.')
  parser.add_argument('--record-every-iter', dest='record_every_iter',
    type=int, default=1, 
    help='Print and log to tensorboard every record_every_iter iteration')

  # Dataloaders
  parser.add_argument('--visual-transform', dest='visual_transform',
    default='normalize', choices=[None, 'normalize'],
    help='Transform to apply in NTURGBDwithFlow')
  parser.add_argument('--target-length', dest='target_length',
    type=int, default=6,
    help='Length of sequences (frames) used by networks. Will be uniformly '
         'sampled within each video.')
  parser.add_argument('--num-workers', dest='num_workers', default=1,
    type=int, help='Number of workers to load train/test data samples')

  args = parser.parse_args()
  
  params = str(vars(args))

  s = params.find('[') # handle a list
  e = params.find(']', s)
  params = params[:s] + params[s:e+1].replace(', ', ' ') + params[e+1:]
  s = params.find('checkpoint_files\': {') # handle a dict
  e = params.find('}', s)
  params = params[:s] + params[s:e+1].replace(', ', ' ') + params[e+1:]
  params = sorted( params[1:-1].replace("'","").split(', ') )
  print( "\nRunning with following parameters: \n  {}\n".format('\n  '.join(params)) )

  return args

if __name__ == "__main__":
  main()