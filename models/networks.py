#!/usr/bin/env python3
"""
References
[1] J. Li, Y. Wong, Q. Zhao, and M. S. Kankanhalli, “Unsupervised Learning of 
    View-invariant Action Representations.," NeurIPS, 2018.
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchvision

import ConvolutionalRNN
from RevGrad import RevGrad

class CNN(nn.Module):
  def __init__(self, input_shape, model_name='resnet18', input_channel=3):
    super(CNN, self).__init__()
    self.model_name = model_name
    self.out_size = None

    # CNN
    if self.model_name.startswith('resnet'):
      self.front_model = getattr(torchvision.models, self.model_name)(pretrained=False)
      if input_channel != 3:
        self.front_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, 
                                           stride=2, padding=3, bias=False)
        self.front_model.conv1.apply(self._init_weights)
      self.front_model = nn.Sequential(*list(self.front_model.children())[:-2])
      last_conv_outsize = self._get_intermediate_outsize(input_shape, interrupt=1)

      # Add a 1 × 1 × 64 convolutional layer to reduce the feature size,
      # following [1]
      self.front_model = nn.Sequential(
        self.front_model, 
        nn.Conv2d(last_conv_outsize[0], 64, kernel_size=1, bias=False)
        )
      self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)
    else:
      raise NotImplementedError('model_name {} not implemented.'.format(self.model_name))

  def forward(self, x, interrupt=0):
    if self.model_name.startswith('resnet'):
      x = self.front_model( x )
    if interrupt == 1: 
      # _get_intermediate_outsize
      return x

    return x

  def _init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      m.bias.data.fill_(0.01)
    else:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  # generate input sample and forward to get output shape
  def _get_intermediate_outsize(self, input_shape, interrupt):
    input = Variable(torch.rand(1, *input_shape)) # 1 for batch_size
    output_feat = self.forward(input, interrupt=interrupt)
    n_size = output_feat.size()[1:]
    return n_size

class Encoder(nn.Module):
  def __init__(self, input_shape, encoder_block='convbilstm', hidden_size=64):
    super(Encoder, self).__init__()
    self.input_shape = input_shape
    self.encoder_block=encoder_block
    self.hidden_size = hidden_size
    self.out_size = None

    # Encoder
    if self.encoder_block == 'convbilstm':
      self.convlstm = ConvolutionalRNN.Conv2dLSTM(
        in_channels=self.input_shape[0],  # Corresponds to input size
        out_channels=self.hidden_size,  # Corresponds to hidden size
        kernel_size=7,  # Int or List[int]
        num_layers=1,
        bidirectional=True,
        # dilation=2, stride=2, dropout=0.5,
        batch_first=True
        )

      # self.convlstm.apply(self._init_weights) # TODO
    else:
      raise NotImplementedError(
        'Given argument encoder_block: {} is not implemented.'.format(self.encoder_block)
        )

    # indicate out_size according to what you are going to do
    if self.encoder_block == 'convbilstm':
      self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)

  def _init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      m.bias.data.fill_(0.01)
    elif type(m) == ConvolutionalRNN.Conv2dLSTM:
      for weight in m.parameters():
        nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    else:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, x, interrupt=0):
    if self.encoder_block == 'convbilstm':
      # Hidden states are initialized automatically when None is given
      hidden = None

      # Go through Encoder
      # Unsqueeze x to set batch_size as 1 and sequence length as the number
      #   of frames. 
      # Input  shape: (batch, seq_len, input_size)
      # Output shape: (batch, seq_len, num_directions * hidden_size)
      # Hidden shape: (batch, num_layers * num_directions, hidden_size)
      output, hidden = self.convlstm(x, hidden)

      if interrupt == 1: 
        # _get_intermediate_outsize
        return output

    return output, hidden

  # generate input sample and forward to get output shape
  def _get_intermediate_outsize(self, input_shape, interrupt):
    input = Variable(torch.rand(1, 1, *input_shape)) # 1, 1 for batch_size and seq_len
    output_feat = self.forward(input, interrupt=interrupt)
    n_size = output_feat.size()[1:]
    return n_size

class CrossViewDecoder(nn.Module):
  def __init__(self, input_shape):
    super(CrossViewDecoder, self).__init__()
    self.input_shape = input_shape
    self.out_size = None

    # Transposed Conv
    self.deconv2d_1 = nn.ConvTranspose2d(
      in_channels=self.input_shape[0], out_channels=80, kernel_size=3, 
      stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
    self.deconv2d_1_bn = nn.BatchNorm2d(80)

    self.deconv2d_2 = nn.ConvTranspose2d(
      in_channels=80, out_channels=36, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    self.deconv2d_2_bn = nn.BatchNorm2d(36)

    self.deconv2d_3 = nn.ConvTranspose2d(
      in_channels=36, out_channels=17, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    self.deconv2d_3_bn = nn.BatchNorm2d(17)

    self.deconv2d_4 = nn.ConvTranspose2d(
      in_channels=17, out_channels=3, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

    # indicate out_size according to what you are going to do
    self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)

    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x, e, interrupt=0):
    # e is the output of Encoder to be concatenated with x
    # Shapes:
    #   x: (k, h, w)
    #   e: (2k, h, w)
    #   x concat e: (3k, h, w)
    x = torch.cat((x,e), dim=1) # dim 0 is batch dimension
    x = F.relu( self.deconv2d_1_bn( self.deconv2d_1(x) ) )
    x = F.relu( self.deconv2d_2_bn( self.deconv2d_2(x) ) )
    x = F.relu( self.deconv2d_3_bn( self.deconv2d_3(x) ) )
    x = self.deconv2d_4(x) # x.size(0) for batch size
    if interrupt == 1: 
      # _get_intermediate_outsize
      return x

    return x

  # generate input sample and forward to get output shape
  def _get_intermediate_outsize(self, input_shape, interrupt):
    input1_shape = (input_shape[0]/3,) + input_shape[1:]
    input2_shape = (input_shape[0]*2/3,) + input_shape[1:]
    input = Variable(torch.rand(1, *input1_shape)) # 1 for batch_size
    input2 = Variable(torch.rand(1, *input2_shape)) # 1 for batch_size
    output_feat = self.forward(input, input2, interrupt=interrupt)
    n_size = output_feat.size()[1:]
    return n_size

class ReconstructionDecoder(nn.Module):
  def __init__(self, input_shape):
    super(ReconstructionDecoder, self).__init__()
    self.input_shape = input_shape
    self.out_size = None

    # Transposed Conv
    self.deconv2d_1 = nn.ConvTranspose2d(
      in_channels=self.input_shape[0], out_channels=64, kernel_size=3, 
      stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
    self.deconv2d_1_bn = nn.BatchNorm2d(64)

    self.deconv2d_2 = nn.ConvTranspose2d(
      in_channels=64, out_channels=32, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    self.deconv2d_2_bn = nn.BatchNorm2d(32)

    self.deconv2d_3 = nn.ConvTranspose2d(
      in_channels=32, out_channels=16, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    self.deconv2d_3_bn = nn.BatchNorm2d(16)

    self.deconv2d_4 = nn.ConvTranspose2d(
      in_channels=16, out_channels=3, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

    # indicate out_size according to what you are going to do
    self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)

    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x, interrupt=0):
    x = F.relu( self.deconv2d_1_bn( self.deconv2d_1(x) ) )
    x = F.relu( self.deconv2d_2_bn( self.deconv2d_2(x) ) )
    x = F.relu( self.deconv2d_3_bn( self.deconv2d_3(x) ) )
    x = self.deconv2d_4(x) # x.size(0) for batch size
    if interrupt == 1: 
      # _get_intermediate_outsize
      return x

    return x

  # generate input sample and forward to get output shape
  def _get_intermediate_outsize(self, input_shape, interrupt):
    input = Variable(torch.rand(1, *input_shape)) # 1 for batch_size
    output_feat = self.forward(input, interrupt=interrupt)
    n_size = output_feat.size()[1:]
    return n_size

class ViewClassifier(nn.Module):
  def __init__(self, input_size, num_classes, reverse=True):
    super(ViewClassifier, self).__init__()
    self.num_classes = num_classes
    
    # Gradient Reversal Layer with two 
    self.grl = RevGrad(reverse=reverse)
    self.fc1 = nn.Linear(input_size, 1024)
    self.fc2 = nn.Linear(1024, self.num_classes)

    self.fc1.apply(self._init_weights)
    self.fc2.apply(self._init_weights)

  def _init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      m.bias.data.fill_(0.01)
    else:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, x):
    x = self.grl(x)
    x = F.relu( self.fc1( x ) )
    x = self.fc2(x)
    return x