#prepare_split_aug_images.py
#UPDATED CHECK
from skimage import io
import numpy as np
from tqdm import tqdm
import shutil
import os
from aicsimageio import AICSImage, imread
import shutil
import time
import numpy
import random
from aicsimageio import AICSImage, imread
from aicsimageio.writers import png_writer 
from tqdm import tqdm
from google.colab.patches import cv2_imshow
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from tqdm import tqdm
from timeit import default_timer as timer
import imageio
import tifffile 
from aicsimageio.transforms import reshape_data
from datetime import datetime


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def make_folder_with_date(save_location, name):
  today = datetime.now()
  if today.hour < 12:
    h = "00"
  else:
    h = "12"
  sub_save_location = save_location + "/" + today.strftime('%Y%m%d%H')+ "_"+ today.strftime('%H%M%S')+ "_%s"%name
  os.mkdir(sub_save_location)
  return sub_save_location


def diplay_img_info(img, divisor, use_RGB):
  ### display image data
    nr_z_slices = img.shape[1]
    nr_timepoints = img.shape[0]
    x_dim = img.shape[-2]
    y_dim = img.shape[-2] 
    x_div = x_dim//divisor
    y_div = y_dim//divisor
    print(img.shape)
    print("The Resolution is: " + str(x_dim))
    print("The number of z-slizes is: " + str(nr_z_slices))
    print("The number of timepoints: " + str(nr_timepoints))
    if use_RGB:
        nr_channels = img.shape[-1]
        print("The number of channels: " + str(nr_channels))
        nr_channels = 1
    else:
        nr_channels = 1
    return nr_z_slices, nr_channels, nr_timepoints, x_dim, y_dim, x_div, y_div 


def rotation_aug(source_img, name, path, use_RGB, flip=False):
    print(source_img.shape)
    # Source Rotation
    if use_RGB:
      source_img_90 = np.rot90(source_img,axes=(-3,-2))
      source_img_180 = np.rot90(source_img_90,axes=(-3,-2))
      source_img_270 = np.rot90(source_img_180,axes=(-3,-2))
    if not use_RGB:
      source_img_90 = np.rot90(source_img,axes=(-2,-1))
      source_img_180 = np.rot90(source_img_90,axes=(-2,-1))
      source_img_270 = np.rot90(source_img_180,axes=(-2,-1))
    # Add a flip to the rotation
    if flip == True:
      source_img_lr = np.flip(source_img)
      source_img_90_lr = np.flip(source_img_90)
      source_img_180_lr = np.flip(source_img_180)
      source_img_270_lr = np.flip(source_img_270)

      #source_img_90_ud = np.flipud(source_img_90)
    # Save the augmented files
    # Source images
    io.imsave(path + "/"+"{}_permutation-00.tif".format(name),source_img)
    io.imsave(path + "/"+"{}_permutation-01.tif".format(name),source_img_90)
    io.imsave(path + "/"+"{}_permutation-02.tif".format(name),source_img_180)
    io.imsave(path + "/"+"{}_permutation-03.tif".format(name),source_img_270)
    # Target images
   
    if flip == True:
      io.imsave(path + "/"+"{}_permutation-04.tif".format(name),source_img_lr)
      io.imsave(path + "/"+"{}_permutation-05.tif".format(name),source_img_90_lr)
      io.imsave(path + "/"+"{}_permutation-06.tif".format(name),source_img_180_lr) 
      io.imsave(path + "/"+"{}_permutation-07.tif".format(name),source_img_270_lr)

 
def flip(source_img, name, path):
    source_img_lr = np.flip(source_img)
    io.imsave(path + "/"+"{}_permutation-00.tif".format(name),source_img)
    io.imsave(path + "/"+"{}_permutation-04.tif".format(name),source_img_lr)

    
def correct_channels(img):
  '''For 2D + T rgb a artificial z channel gets created'''
  if img.shape[-1] ==3:
    use_RGB = True
  else:
    use_RGB = False
  if len(img.shape) ==4 and use_RGB:
    t, x,y,c = img.shape
    zeros = np.zeros((t,1,y,x,c))
    zeros[:,0,:,:,:] = img
    img = zeros
  return img, use_RGB
    
#def change_axis(img):
#    img = img.get_image_data("STCZYX")  # returns 4D CZYX numpy array
#    img = np.swapaxes(img, 1, 2)
#    return img
