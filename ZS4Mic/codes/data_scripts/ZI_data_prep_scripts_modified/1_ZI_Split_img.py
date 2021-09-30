# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 08:22:06 2021

@author: Martin_Priessner
"""
import os
import os
import math
from tqdm import tqdm
import cv2 
import numpy as np
from skimage import io
import shutil
import random
import pandas as pd
from tqdm import tqdm


#### Load the necessary functions
def correct_channels(img):
  '''For 2D + T (with or without RGB) a artificial z channel gets created'''
  if img.shape[-1] ==3:
    use_RGB = True
  else:
    use_RGB = False
  if len(img.shape) ==4 and use_RGB:
    t, y, x, c = img.shape
    zeros = np.zeros((t,1,y,x,c))
    zeros[:,0,:,:,:] = img
    img = zeros
  elif len(img.shape) ==3 and not use_RGB:
    t, x, y = img.shape
    zeros = np.zeros((t,1,y,x))
    zeros[:,0,:,:] = img
    img = zeros
  elif len(img.shape) ==3 and use_RGB: # to be able to handle normal 2D + RGB images
    y, x, c = img.shape
    zeros = np.zeros((1,1,y,x,c))
    zeros[0,0,:,:,:] = img
    img = zeros
  return img, use_RGB
def get_img_dim(img):
  """This function gets the right x,y,t,z dimensions and if it is an RGB image or not"""
  if (img.shape[-1] ==3 and len(img.shape) ==5):
    use_RGB = True
    t_dim, z_dim, y_dim, x_dim, _ = img.shape
  elif (img.shape[-1] ==3 and len(img.shape) ==4):
    use_RGB = True
    t_dim, y_dim, x_dim, channel = img.shape
    zeros = np.zeros((t_dim,1,y_dim,x_dim,channel))
    zeros[:,0,:,:,:] = img
    img = zeros
    t_dim, z_dim, y_dim, x_dim, _ = img.shape
  elif (img.shape[-1] !=3 and len(img.shape) ==3):  # create a 4th dimension
    use_RGB = False
    t, y, x = img.shape
    zeros = np.zeros((t,1,y,x))
    zeros[:,0,:,:] = img
    img = zeros
    t_dim, z_dim, y_dim, x_dim= img.shape
  elif (img.shape[-1] !=3 and len(img.shape) ==4):
    use_RGB = False
    t_dim, z_dim, y_dim, x_dim = img.shape
  return t_dim, z_dim, y_dim, x_dim, use_RGB

def get_all_filepaths_in_folder(folder_path):
    '''This function gets the paths from each file in folder and subfolder of a given location'''
    flist = []
    for path, subdirs, files in tqdm(os.walk(folder_path)):
          for name in files:
            flist.append(os.path.join(path, name))
    return flist




######################## SELECT SOURCE FOLDER ########################
#@markdown Provide the folder with the training data
# Define the necessary paths needed later
Source_path = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\1024'#@param {type:"string"}

Parent_path = os.path.dirname(Source_path)
test_train_seq_path = os.path.join(Parent_path, "sequences")

######################## SELECT test_train_split  ########################
# Paramenters
test_train_split = 0.1 #@param {type:"slider", min:0.1, max:1, step:0.1}
N_frames = 7 

# create seq_lists *.txt
train_seq_txt = os.path.join(Parent_path, "sep_trainlist.txt")
test_seq_txt = os.path.join(Parent_path, "sep_testlist.txt")
with open(train_seq_txt, "w") as f:
  f.write("")
with open(test_seq_txt, "w") as f:
  f.write("")

# delete test_train_folder if already exists
if os.path.isdir(test_train_seq_path):
  shutil.rmtree(test_train_seq_path)
os.mkdir(test_train_seq_path)

#get all files in the selected folder
flist = get_all_filepaths_in_folder(Source_path)

# split the different images and save them in the sequence folder with a given folderstructure
# and create the test train split seq txt files
for counter_1, file_path in tqdm(enumerate(flist)):
  os.chdir(test_train_seq_path)
  file_folder = "%05d"%(counter_1+1)
  print(file_folder)
  os.mkdir(file_folder)
  file_folder_path = os.path.join(test_train_seq_path, file_folder)
  os.chdir(file_folder_path)

  img = io.imread(file_path)
  # makes 3D into 4D dataset if needed
  img, _ = correct_channels(img)
  t_dim, z_dim, y_dim, x_dim, use_RGB = get_img_dim(img)

  #calculate how many folders need to be created to cover all the images
  N_folders_per_slice = math.ceil(t_dim/N_frames)
  counter_2 = 1
  for z in tqdm(range(z_dim)):
    if (use_RGB and len(img.shape) ==5) or (use_RGB and len(img.shape) ==4):
       img_slice = img[:, z, :, :, :]
    else:
      img_slice = img[:, z, :, :]
   
    for seq in range(1,N_folders_per_slice):
      seq_folder = "%04d"%(counter_2)
      seq_folder_path = os.path.join(file_folder_path, seq_folder)
      os.mkdir(seq_folder_path)
      counter_2 += 1
      #create lever to randomly shift samples to test or train depending on the chosen split 
      test_train_lever = random.uniform(0, 1)
      if test_train_lever < test_train_split:
        with open(test_seq_txt, "a") as f:
          f.write(f"{file_folder}/{seq_folder}\n")
      else:
        with open(train_seq_txt, "a") as f:
          f.write(f"{file_folder}/{seq_folder}\n")

      #save a given number of images as png in the new folder
      for im_num in range(1, N_frames+1):
        # print(((seq-1)*N_frames+(im_num-1)), y_dim, x_dim)
        png_img_path = os.path.join(seq_folder_path, f"im{im_num}.png")
        if use_RGB:
            png_img = img_slice[((seq-1)*N_frames+(im_num-1)), :, :, :]
            img_channels = png_img
        else:
            png_img = img_slice[((seq-1)*N_frames+(im_num-1)), :, :]
            img_channels = np.zeros((y_dim, x_dim, 3))
            img_channels[:,:,0] = png_img
            img_channels[:,:,1] = png_img
            img_channels[:,:,2] = png_img
        io.imsave(png_img_path, img_channels)
