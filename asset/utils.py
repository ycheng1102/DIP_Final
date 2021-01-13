import torch
import cv2
import numpy
import shutil
import os
import math

def read_flo(strFile):
  with open(strFile, 'rb') as objFile:
    strFlow = objFile.read()
  assert(numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=1, offset=0) == 202021.25)
  intWidth = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=4)[0]
  intHeight = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=8)[0]
  return numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])
# end

backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
  if str(tenFlow.shape) not in backwarp_tenGrid:
    tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
    tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
    backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
  # end
  tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
  return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

def extract_keyframe(arg_video, arg_FPS, arg_width, arg_height, arg_threshold):

  vidcap = cv2.VideoCapture(arg_video)
  FPS = math.ceil(vidcap.get(cv2.CAP_PROP_FPS))

  assert FPS < arg_FPS * FPS, "Original FPS is bigger than the desired FPS!"
  print('Convert from %dfps to %dfps.' % (FPS, arg_FPS * FPS))
  print('Capture frame from video...')

  i = 0
  success,prev_frame = vidcap.read()
  if success:
    prev_frame = cv2.resize(prev_frame,(arg_width, arg_height),interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('video/' + f'{i:04d}.png', prev_frame)
    i += 1
  #end
  success,cur_frame = vidcap.read()
  while success:
    cur_frame = cv2.resize(cur_frame,(arg_width, arg_height),interpolation=cv2.INTER_LANCZOS4)
    diff = cv2.absdiff(cur_frame, prev_frame)
    non_zero_count = numpy.count_nonzero(diff)
    if non_zero_count > arg_threshold:
      cv2.imwrite('video/' + f'{i:04d}.png', cur_frame)    
      i += 1
    prev_frame = cur_frame
    success,cur_frame = vidcap.read()
  # end
  vidcap.release()
  cv2.destroyAllWindows()
  print("Total Number of keyframes : {}".format(i))
  print()

  return FPS, i
# end

def create_dir():
  if os.path.isdir('video'):
    shutil.rmtree('video')
  os.makedirs('video')
# end

def create_video(frame_array, FPS, arg_width, arg_height):
  print('Creating video...')
  out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), FPS , (arg_width, arg_height))
  print('Total frames : ' + str(len(frame_array)))
  for i in range(len(frame_array)):
    out.write(frame_array[i])
  out.release()
  print('Done!')
#end