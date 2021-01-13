import torch
import cv2
import numpy
import os
import getopt
import sys
import math

from PIL import Image
from asset import flow, softsplat
from asset.utils import create_dir, create_video, extract_keyframe, read_flo, backwarp

##########################################################
assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0
##########################################################

if __name__ == '__main__':

  ##########################################################
  assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0
  torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
  torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
  ##########################################################

  arg_width = 640
  arg_height = 360
  arg_FPS = 2
  arg_Flow = './out.flo'
  arg_threshold = 0

  for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--video' and strArgument != '': arg_video = strArgument
    if strOption == '--width' and strArgument != '': arg_width = strArgument
    if strOption == '--height' and strArgument != '': arg_height = strArgument
    if strOption == '--fps' and strArgument != '': arg_FPS = strArgument
    if strOption == '--flow' and strArgument != '': arg_Flow = strArgument
    if strOption == '--threshold' and strArgument != '': arg_threshold = strArgument
  # end

  arg_FPS = int(arg_FPS)
  arg_width = int(arg_width)
  arg_height = int(arg_height)
  arg_threshold = int(arg_threshold)

  time = numpy.linspace(0.0, 1.0, arg_FPS + 1).tolist()
  time.remove(0.0)
  time.remove(1.0)
  #print(time)

  create_dir()
  FPS, frameNum = extract_keyframe(arg_video, arg_FPS, arg_width, arg_height, arg_threshold)
  
  frame_array = []
  listA = os.listdir('video')
  listB = os.listdir('video')
  listA.sort()
  listB.sort()
  listB.remove('0000.png')
  
  i = 0
  for A, B in zip(listA, listB):

    #if i == 30:
    #  break
    if i % (arg_FPS * FPS) == 0: print('-- processing %d / %d' % (i, frameNum))
    i += 1

    fullpath_A = 'video/' + A
    fullpath_B = 'video/' + B
    pic_A = cv2.imread(filename=fullpath_A, flags=-1)
    pic_B = cv2.imread(filename=fullpath_B, flags=-1)

    tenFirst = torch.FloatTensor(numpy.ascontiguousarray(pic_A[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenSecond = torch.FloatTensor(numpy.ascontiguousarray(pic_B[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenFlow = flow.estimate(tenFirst, tenSecond)

    objOutput = open(arg_Flow, 'wb')
    numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
    numpy.array([tenFlow.shape[2], tenFlow.shape[1]], numpy.int32).tofile(objOutput)
    numpy.array(tenFlow.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
    objOutput.close()

    tenFirst = torch.FloatTensor(numpy.ascontiguousarray(pic_A.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
    tenSecond = torch.FloatTensor(numpy.ascontiguousarray(pic_B.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
    tenFlow = torch.FloatTensor(numpy.ascontiguousarray(read_flo(arg_Flow).transpose(2, 0, 1)[None, :, :, :])).cuda()

    tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)

    frame_array.append(pic_A)
    for t in time:
      tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * t, tenMetric=-20.0 * tenMetric, strType='softmax')
      frame = tenSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
      frame = (frame * 255).astype(numpy.uint8)
      frame_array.append(frame)

  print('-- processing %d / %d' % (frameNum, frameNum))
  print()

  create_video(frame_array, arg_FPS*FPS, arg_width, arg_height)
  print('Done!')
# end
