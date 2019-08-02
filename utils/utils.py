#!/usr/bin/env python3
import os
import sys
import math
import time
import h5py
from datetime import datetime

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter # tensorboard for any library

import sklearn.metrics as skm

def asDays(s):
  d = s // 86400
  if d > 0:
    d = '%dd ' % d
  else:
    d = ''
  s = s % 86400
  h = s // 3600
  if h > 0:
    h = '%dh ' % h
  else:
    h = ''
  s = s % 3600
  m = s // 60
  if m > 0:
    m = '%dm ' % m
  else:
    m = ''
  s = int(s % 60)
  
  return '{}{}{}{:d}s'.format(d, h, m, s)

def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (asDays(s), asDays(rs))

class movingAverage:
  def __init__(self, length):
    self._length = length
    self._items = []
    self._sum = 0
    self._count = 0

  def add(self, item):
    self._items.append(item)
    self._sum += item
    self._count += 1
    if self._count > self._length:
      self._sum -= self._items.pop(0)
      self._count = self._length

  def mean(self):
    return self._sum / self._count

class runningMean:
  def __init__(self):
    self._count = 0
    self._sum = 0

  def add(self, item):
    self._sum += item
    self._count += 1

  def mean(self):
    return self._sum / self._count

def setCheckpointFileDict(all_models, checkpoint_files):
  output = {}
  if len(checkpoint_files) > 0:
    models_name = all_models + ['iter']
    if 'all' in checkpoint_files:
      for m in models_name:
        output[m] = checkpoint_files['all']
      checkpoint_files.pop('all')
    elif 'else' in checkpoint_files:
      for m in models_name:
        output[m] = checkpoint_files['else']
      checkpoint_files.pop('else')
    if len(checkpoint_files) > 0:
      for m in checkpoint_files:
        output[m] = checkpoint_files[m]

  return output

def loadCheckpoints(models, modules, optimizers, checkpoint_files={}):
  # checkpoint_files should specify model name as key and to specify starting 
  # iteration number, 'iter' key should be in checkpoint_files
  if len(checkpoint_files) > 0:
    for m in checkpoint_files:
      if m == 'iter':
        continue
      checkpoint = torch.load(checkpoint_files[m])
      trained_dict = checkpoint[m]
      model_dict = models[m].state_dict()

      # 1. filter out unnecessary keys
      not_used_keys = {k: v for k, v in trained_dict.items() if k not in model_dict}
      trained_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
      if len(not_used_keys.keys()) > 0:
        print('Unused keys in checkpoint for model {}:'.format(m), not_used_keys.keys())
      # 2. overwrite entries in the existing state dict
      model_dict.update(trained_dict) 
      # 3. load the new state dict
      models[m].load_state_dict(trained_dict)
      print('{} checkpoint file loaded ({})'.format(m, checkpoint_files[m]))

      if m in modules:
        if '{}_optimizer'.format(m) in checkpoint:
          optimizers[m].load_state_dict(checkpoint['{}_optimizer'.format(m)])
          print('{}_optimizer checkpoint file loaded ({})'.format(m, checkpoint_files[m]))
        else:
          print('{}_optimizer is not in the checkpoint file. '
                'Not loaded ({})'.format(m, checkpoint_files[m]))
      
    if 'iter' in checkpoint_files:
      checkpoint = torch.load(checkpoint_files['iter'])
      epoch_start = checkpoint['epoch'] + 1
      iter = checkpoint['iter'] + 1

      return epoch_start, iter

  return 1, 1

def makeUniquePath(save_dir='./VIAR', log_prefix='VIAR', unique_name=None):
  if not os.path.isdir(os.path.join(save_dir)):
    os.makedirs(os.path.join(save_dir))
  if unique_name is None or not unique_name.startswith(log_prefix):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    unique_name = log_prefix + '_' + current_time
    print('New directory created for logs and checkpoints: {}'.format(
      os.path.join(save_dir, unique_name)))
  else:
    print('Reusing directory for logs and checkpoints: {}'.format(
      os.path.join(save_dir, unique_name)))
  checkpoints_dir = os.path.join(save_dir, unique_name, 'checkpoints')
  writer_dir = os.path.join(save_dir, unique_name, 'logs')
  if not os.path.isdir(os.path.join(checkpoints_dir)):
    os.makedirs(os.path.join(checkpoints_dir))
  if not os.path.isdir(os.path.join(writer_dir)):
    os.makedirs(os.path.join(writer_dir))
  save_path = os.path.join(checkpoints_dir, '{}_Ep{}_Iter{}.tar')

  return save_path, unique_name, writer_dir

def testIters(run, test_loader, test_dataset, models={}, checkpoint_files={}, 
              args=None, device='cuda'):
  if len(models) < 1:
    raise ValueError('No model is given for testing. Check if models parameter'
                     ' is properly set for testIters.')

  # disable requires_grad for models
  for m in models:
    for param in models[m].parameters():
      param.requires_grad = False

  if not os.path.isdir(os.path.join(args.output_dir)):
    os.makedirs(os.path.join(args.output_dir))

  loadCheckpoints(models, [], {}, checkpoint_files)

  print('Test ongoing...')
  out_h5_path = os.path.join(args.output_dir, 'VIAR_encoder_outputs.h5')
  out_h5 = h5py.File(out_h5_path)
  for test_batch_ind, test_sample in enumerate(test_loader):
    test_result = run(target_modules=[],
                      split='test',
                      sample=test_sample, 
                      models=models, 
                      criterions={},
                      args=args,
                      device=device)

    encoder_output = test_result['output']['encoder_output'].cpu().numpy()
    for name_ind in range(len(test_sample['videoname'])):
      dset = out_h5.create_dataset(test_sample['videoname'][name_ind], 
        encoder_output.shape[1:], # e.g. (6, 128, 7, 7)
        maxshape=encoder_output.shape[1:], 
        chunks=True, dtype='f8')
      dset[:] = encoder_output[name_ind]
      print('Saved encoder output of {}'.format(test_sample['videoname'][name_ind]))


def trainIters(run, target_modules, train_loader, train_dataset, val_loader, val_dataset,
               models={}, all_models=[], log_prefix='VIAR', 
               checkpoint_files={}, n_epoch=2000, save_dir='./VIAR', 
               args=None, device='cuda'):
  if len(models) < 1:
    raise ValueError('No model is given for training. Check if models parameter'
                     ' is properly set for trainIters.')

  # disable requires_grad for non-target models
  for m in all_models:
    if m in target_modules:
      if m not in models:
        raise ValueError('Target module {} is not in given models.'.format(m))
      for param in models[m].parameters():
        param.requires_grad = True
    else:
      if m in models:
        for param in models[m].parameters():
          param.requires_grad = False

  # 1 iteration = 1 backpropagation or 1 training minibatch
  # 1 epoch = iterations of all training samples
  if not os.path.isdir(os.path.join(save_dir)):
    os.makedirs(os.path.join(save_dir))

  # Directory and logging initialization
  save_path, unique_name, writer_dir = makeUniquePath(save_dir=save_dir,
                                                      log_prefix=log_prefix,
                                                      unique_name=args.unique_name)

  # Variable sanity check
  if args.val_every_iter is not None and args.val_every_iter > len(train_dataset):
    print('Warning: val_every_iter should be smaller than len(train_dataset). '
          'Otherwise, redundant save files will be created. '
          'Currently, val_every_iter: {}, and len(train_dataset): {}'.format(
            args.val_every_iter, len(train_dataset) ) 
          )

  if args.val_size is None:
    args.val_size = len(val_dataset)
    
  if args.val_size < 2:
    print('Warning: val_size must bigger than 1. val_size will be set to 100.')
    args.val_size = 100

  start = time.time()

  # optimizers and criterions
  optimizers = {}
  for m in target_modules:
    optimizers[m] = optim.Adam(models[m].parameters(), 
                               lr=args.learning_rate, weight_decay=5e-4)

  criterions = {}
  criterions['crossview'] = nn.MSELoss(size_average=False, reduce=True)
  criterions['reconstruct'] = nn.MSELoss(size_average=False, reduce=True)
  criterions['viewclassify'] = nn.CrossEntropyLoss(size_average=False, reduce=True)

  n_iters = math.ceil(n_epoch * len(train_dataset) / args.batch_size)

  epoch_start, iter = 1, 1
  epoch_start, iter = loadCheckpoints(models, target_modules, optimizers, checkpoint_files)
  print('Starts from epoch {} and iteration {}'.format(epoch_start, iter))

  # Create tensorboard and record command line arguments
  if iter > 1:
    writer = SummaryWriter(log_dir=writer_dir, purge_step=iter)
  else:
    writer = SummaryWriter(log_dir=writer_dir)
  if args is not None:
    writer.add_text('Parameters', str(vars(args))[1:-1], iter)
    cmdline = 'python ' + ' '.join(sys.argv)
    cmdline = cmdline.replace('{', '\'{').replace('}', '}\'')
    writer.add_text('Command line', cmdline, iter)

  for epoch_num in range(epoch_start, n_epoch + 1):
    for batch_ind, sample in enumerate(train_loader):
      train_result = run(target_modules=target_modules,
                         split='train',
                         sample=sample, 
                         models=models, 
                         optimizers=optimizers, 
                         criterions=criterions,
                         args=args,
                         device=device)

      if iter % args.record_every_iter == 0:
        for log in train_result['logs']:
          writer.add_scalar(
            'train_data/{}'.format(log), train_result['logs'][log], iter
            )
        print('{}: Epoch: {:d}, Sample: {:d}, ET: {}'.format(
          unique_name, epoch_num, iter, timeSince(start, iter / n_iters)
          ) )
      # iteration += 1 for every sample
      iter += 1

      if args.val_every_iter is not None:
        if (args.val_every_iter < len(train_dataset)) and ((iter-1) % args.val_every_iter == 0):
          break # out of train_loader loop
       
    print('Validation ongoing...') 
    # in validation use iter-1 for iteration # since iter was incremented in train loop
    val_start = time.time()
    val_logs = {}
    for val_batch_ind, val_sample in enumerate(val_loader):
      val_result = run(target_modules=target_modules,
                       split='validate',
                       sample=val_sample, 
                       models=models, 
                       criterions=criterions,
                       args=args,
                       device=device)

      for log in val_result['logs']:
        if log not in val_logs:
          val_logs[log] = []
        val_logs[log].append(val_result['logs'][log])

      print('{}: Epoch: {:d}, Validation Sample: {:d}, ET: {}'.format(
        unique_name, epoch_num, val_batch_ind + 1, 
        timeSince(val_start, (val_batch_ind + 1) / args.val_size)
        ) )

      if (args.val_size < len(val_dataset)) and (val_batch_ind + 1 >= args.val_size):
        break # out of val_loader loop

    val_log_mean = {}
    for log in val_logs:
      val_log_mean[log] = torch.Tensor(val_logs[log]).mean()
      writer.add_scalar(
        'val_data/{}'.format(log), val_log_mean[log], iter-1
        )

    print('Validation Flow Prediction Error: {}'.format(val_log_mean['reconstruct_loss']))

    # Save checkpoint every epoch    
    save_checkpoint(epoch_num, iter-1, models, target_modules, optimizers,
      filename=save_path.format(unique_name, epoch_num, iter-1) ) # mind iter-1


def save_checkpoint(epoch, iter, models, modules, optimizers, 
                    filename='checkpoint.pth.tar'):
  save_dict = {}
  save_dict['epoch'] = epoch
  save_dict['iter'] = iter
  for m in models:
    save_dict[m] = models[m].state_dict()
    if m in modules:
      save_dict[m + '_optimizer'] = optimizers[m].state_dict()
  torch.save(save_dict, filename)
  print('Saved checkpoint to: {}'.format(filename))