#prepare_split_aug_images.py
#UPDATED CHECK
from skimage import io
import numpy as np
from tqdm import tqdm
import shutil
import os
import shutil
import time
import numpy
import random
from tqdm import tqdm
from google.colab.patches import cv2_imshow
from tqdm import tqdm
from timeit import default_timer as timer
import imageio
import tifffile 
from datetime import datetime

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import csv
import sys
_ = (sys.path.append("/usr/local/lib/python3.6/site-packages"))
sys.path.insert(0,'/content/DAIN/load_functions')

from prepare_dataset_train_test_folders import load_img
from prepare_dataset_train_test_folders import correct_channels



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
  """This function creates a folder starting with the date and time given the folder location and folder name"""
  today = datetime.now()
  if today.hour < 12:
    h = "00"
  else:
    h = "12"
  sub_save_location = save_location + "/" + today.strftime('%Y%m%d%H')+ "_"+ today.strftime('%H%M%S')+ "_%s"%name
  os.mkdir(sub_save_location)
  return sub_save_location


def rotation_aug(source_img, name, path, flip=False):
    print(source_img.shape)
    # Source Rotation
    source_img_90 = np.rot90(source_img,axes=(2,3))
    source_img_180 = np.rot90(source_img_90,axes=(2,3))
    source_img_270 = np.rot90(source_img_180,axes=(2,3))
    # Add a flip to the rotation
    if flip == True:
      source_img_lr = np.fliplr(source_img)
      source_img_90_lr = np.fliplr(source_img_90)
      source_img_180_lr = np.fliplr(source_img_180)
      source_img_270_lr = np.fliplr(source_img_270)

      #source_img_90_ud = np.flipud(source_img_90)
    # Save the augmented files
    # Source images
    io.imsave("{}.tif".format(path + "/" + name), source_img)
    io.imsave("{}_90.tif".format(path + "/" + name), source_img_90)
    io.imsave("{}_180.tif".format(path + "/" + name), source_img_180)
    io.imsave("{}_270.tif".format(path + "/" + name), source_img_270)
   
    if flip == True:

      io.imsave("{}_lr.tif".format(path + "/" + name), source_img_lr)
      io.imsave("{}_90_lr.tif".format(path + "/" + name), source_img_90_lr)
      io.imsave("{}_180_lr.tif".format(path + "/" + name), source_img_180_lr)
      io.imsave("{}_270_lr.tif".format(path + "/" + name), source_img_270_lr)

 
def flip(source_img, name, path):
    source_img_lr = np.fliplr(source_img)
    io.imsave("{}.tif".format(path + "/" + name), source_img)
    io.imsave("{}_lr.tif".format(path + "/" + name), source_img_lr)


def diplay_img_info(img, divisor, use_RGB):
    """ This function displays the information of the image dimensions as a print"""
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

    


def split_img_small(img_list, Source_path, divisor, split_img_folder_path, log_path_file):
    """This function splits the bigger image in to smaller batches and saves them with a structured name "img-xxx_fraction-xx" """
    for image_num in tqdm(range(len(img_list))):
        img_path = os.path.join(Source_path,img_list[image_num])
        t, z, y_dim,x_dim, img, use_RGB =load_img(img_path)
        nr_z_slices, nr_channels, nr_timepoints, x_dim, y_dim, x_div, y_div = diplay_img_info(img, divisor, use_RGB)
        multiplyer = x_dim/divisor
        os.chdir(split_img_folder_path)
        for i in range(x_div):
          for j in range(y_div):
            img_crop = img
            print(img.shape)
            if use_RGB==True:
              img_crop = img_crop[:,:,(i*divisor):((i+1)*divisor),(j*divisor):((j+1)*divisor), :]
            else:
              img_crop = img_crop[:,:,(i*divisor):((i+1)*divisor),(j*divisor):((j+1)*divisor)]
            
            if use_RGB:
              name = ("img-%03d" %(image_num)+"_fraction-%02d_RGB" %((i*multiplyer)+j))
            else:
              name = ("img-%03d" %(image_num)+"_fraction-%02d" %((i*multiplyer)+j))
            print("saving image {}".format(name))
            io.imsave("{}.tif".format(name),img_crop)
            # cv2_imshow(img_crop)    
            with open(log_path_file, "a", newline='') as name_log:
                writer = csv.writer(name_log)
                writer.writerow([f"{img_list[image_num][:-4]}",f"{name}.tif"])
    return use_RGB
