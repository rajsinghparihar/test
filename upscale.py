import os
import warnings
import sys
warnings.filterwarnings('ignore')

# import time, sys
# from datetime import timedelta
sys.path.append(".")
sys.path.append("./taming-transformers")
sys.path.insert(0, "./latent-diffusion")
from taming.models import vqgan

mode = "superresolution"
from utils import get_model
from glob import glob
from os.path import isdir, join

model = get_model(mode)

from utils import run
import torch
import numpy as np
from PIL import Image
from os import path as ntpath
import requests

# Input (str, boolean): path, remove first slash
# Output (string): path with missing / at the end, opt: remove from beginning
def fix_path(path, add_slash=False):
  if path.endswith('/'):
    path = path #path[:-1]
  if not path.endswith('/'):
    path = path+"/"
  if path.startswith('/') and add_slash == True:
    path = path[1:]
  return path
  
# Input (str): path
# Output (str): filename with extension
def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail or ntpath.basename(head)

# Input (str): file path
# Output (str): enclosing directory
def path_dir(path):
  return path.replace(path_leaf(path), '')

def gen_id(type='short'):
  id = ''
  if type is 'short':
    id = requests.get('https://api.inha.asia/k/?type=short').text
  if type is 'long':
    id = requests.get('https://api.inha.asia/k').text
  return id

def list_images(path, format='all'):
  imagefiles = []
  if format is 'all':
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.gif' '*.JPG', '*.JPEG', '*.PNG', '*.GIF'):
      imagefiles.extend(glob(join(path, ext)))
  else:
    imagefiles = glob(path+'/*.'+format)
  imagefiles.sort()
  return imagefiles

drive_root = "./"
input = "input_images"
output_dir = "output/"

steps = 10

uniq_id = gen_id()

if os.path.isfile(drive_root+input):
  inputs = [drive_root+input]
  dir_in = path_dir(drive_root+input)
elif os.path.isdir(drive_root+input):
  dir_in = drive_root+fix_path(input)
  inputs = list_images(dir_in)


# inputs = ["./" + input]
if output_dir == '':
  dir_out = dir_in
else:
  if not os.path.isdir(drive_root+output_dir):
    os.mkdir(drive_root+output_dir)
  dir_out = drive_root+fix_path(output_dir)

for input in inputs:
  print('in:', input )
  img_out = dir_out + uniq_id + '_'+str(steps)+'steps_'+path_leaf(input)
  

  logs = run(model["model"], input, mode, steps)

  sample = logs["sample"]
  sample = sample.detach().cpu()
  sample = torch.clamp(sample, -1., 1.)
  sample = (sample + 1.) / 2. * 255
  sample = sample.numpy().astype(np.uint8)
  sample = np.transpose(sample, (0, 2, 3, 1))
  a = Image.fromarray(sample[0])
  a.save(img_out)
#   display(a)

  if os.path.isfile(img_out):
    print("image saved")
    # op(c.ok, 'Upscaled image saved as', img_out.replace(drive_root, ''))
  else:
    print("error")
    # op(c.fail, 'Error occurred: ', input.replace(drive_root, ''))

