# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:10:13 2021

@author: Martin_Priessner
"""
####### INSTRUCTIONS #############
# Load all the functions below and change the "Source_path" location and "input_size" accordingly and run the code
# this will create the split training data in the folder: 
# e.g. ...*\spit_source_DAIN\512_img_separation\2021070816_163008_prep_t_train
##############################################
##############################################
##############################################
##############################################

import sys
import os
import csv
from tqdm import tqdm
from skimage import io
from datetime import datetime
import numpy as np
import random
from skimage.color import gray2rgb


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


def load_img(img_path):
    """This Function loads the image, corrects the channels (in case if it is not a 4D image file) and returns the dimensions of the file with a boolean of whether it is a RGB file"""
    img = io.imread(img_path)
    img, use_RGB = correct_channels(img)
    if img.shape[-1]==3:
      use_RGB = True
      t, z, y_dim, x_dim, _ = img.shape 
      print("This image will be processed as a RGB image")
    else:
      use_RGB = False
      t, z, y_dim, x_dim = img.shape 
    print("The image dimensions are: " + str(img.shape))
    return t, z, y_dim,x_dim, img, use_RGB
  

def correct_channels(img):
  '''For 2D + T (with or without RGB) a artificial z channel gets created'''
  if img.shape[-1] ==3:
    use_RGB = True
  else:
    use_RGB = False
  if len(img.shape) ==4 and use_RGB:
    t, x, y, c = img.shape
    zeros = np.zeros((t,1,y,x,c), dtype=np.uint8 )
    zeros[:,0,:,:,:] = img
    img = zeros
  elif len(img.shape) ==3 and not use_RGB:
    t, x, y = img.shape
    zeros = np.zeros((t,1,y,x), dtype=np.uint8 )
    zeros[:,0,:,:] = img
    img = zeros
  return img, use_RGB


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


def data_train_test_preparation(folder_option, split_img_folder_path, save_location, split_training_test):
    """This function executs the folder creation option to prepare the test or train folder necessary for the network"""
    if folder_option == "prep_t_train":
      name = "prep_t_train"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        train_folder = prepare_t_train_data(img_path_list, file_num, sub_save_location, split_training_test)
        folder_steps = ""
        folder_gt = ""
        split_folder  = ""
    elif folder_option == "prep_z_train":
      name = "prep_z_train"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        train_folder = prepare_z_train_data(img_path_list, file_num, sub_save_location, split_training_test)    
        folder_steps = ""
        folder_gt = ""
        split_folder  = ""

    return split_folder, folder_steps, folder_gt, train_folder, sub_save_location

def get_img_path_list(img_path_list, img_folder_path):
      ''' Creates a list of full image-paths of a provided folder. It also takes an empty list to store the found paths and returns that list'''
      flist = os.listdir(img_folder_path)
      flist.sort()
      for i in flist:
        img_slice_path = os.path.join(img_folder_path, i)
        img_path_list.append(img_slice_path)
      return img_path_list



def prepare_t_train_data(img_path_list, file_num,  sub_save_location, split_training_test):
    os.chdir(sub_save_location)
    sub_folder = "sequences"
    sequence_path = os.path.join(sub_save_location, sub_folder)
    if not os.path.exists(sequence_path):
      os.mkdir(sequence_path)
      os.chdir(sequence_path)
    else:
      os.chdir(sequence_path)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])

    for z_num in tqdm(range(0,z)):
    #create new directory-path
      file_folder = "%02d" % (z_num+1+file_num*z)
      z_folder = os.path.join(sequence_path, file_folder)
      os.mkdir(z_folder)  
      os.chdir(z_folder)

      for t_num in range(0,t-2):
        slice_folder = "%04d" % (t_num+1)
        three_t_folder = os.path.join(z_folder, slice_folder)
        os.mkdir(three_t_folder)  
        os.chdir(three_t_folder)
        #add new folder to txt-file
        decision_train_test = random.random()
        if decision_train_test < split_training_test:
          txt_file_train = open(sub_save_location + "/tri_trainlist.txt", "a")
          txt_file_train.write("{}/{}\n".format(file_folder,slice_folder))
          txt_file_train.close()
        else:
          txt_file_test = open(sub_save_location + "/tri_testlist.txt", "a")
          txt_file_test.write("{}/{}\n".format(file_folder,slice_folder))
          txt_file_test.close()
          
        #converting images in rgb and uint8 to save it like that
        if use_RGB == False:
          img_save_1 = img[t_num,z_num, :, :] 
          img_save_1 = gray2rgb(img_save_1)

          img_save_2 = img[t_num+1,z_num, :, :] 
          img_save_2 = gray2rgb(img_save_2)

          
          img_save_3 = img[t_num+2,z_num, :, :] 
          img_save_3 = gray2rgb(img_save_3)

        if use_RGB == True:
          img_save_1 = img[t_num,z_num, :, :, :] 
          # img_save_1 = convert(img_save_1, 0, 255, np.uint8)
          img_save_2 = img[t_num+1,z_num, :, :, :] 
          # img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          img_save_3 = img[t_num+2,z_num, :, :, :] 
          # img_save_3 = convert(img_save_3, 0, 255, np.uint8)
        # saving images as PNG
        io.imsave("{}.png".format("im1"), img_save_1)
        io.imsave("{}.png".format("im2"), img_save_2)
        io.imsave("{}.png".format("im3"), img_save_3)
        print("{}/{}\n".format(file_folder,slice_folder))
    return sequence_path

def prepare_z_train_data(img_path_list, file_num,  sub_save_location, split_training_test):
    os.chdir(sub_save_location)
    sub_folder = "sequences"
    sequence_path = os.path.join(sub_save_location, sub_folder)
    if not os.path.exists(sequence_path):
      os.mkdir(sequence_path)
      os.chdir(sequence_path)
    else:
      os.chdir(sequence_path)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])

    for t_num in tqdm(range(0,t)):
    #create new directory-path
      file_folder = "%02d" % (t_num+1+file_num*t)
      t_folder = os.path.join(sequence_path, file_folder)
      os.mkdir(t_folder)  
      os.chdir(t_folder)

      for z_num in range(0,z-2):
        slice_folder = "%04d" % (z_num+1)
        three_z_folder = os.path.join(t_folder, slice_folder)
        os.mkdir(three_z_folder)  
        os.chdir(three_z_folder)
        #add new folder to txt-file
        decision_train_test = random.random()
        if decision_train_test < split_training_test:
          txt_file_train = open(sub_save_location + "/tri_trainlist.txt", "a")
          txt_file_train.write("{}/{}\n".format(file_folder,slice_folder))
          txt_file_train.close()
        else:
          txt_file_test = open(sub_save_location + "/tri_testlist.txt", "a")
          txt_file_test.write("{}/{}\n".format(file_folder,slice_folder))
          txt_file_test.close()
          
        #converting images in rgb and uint8 to save it like that
        if use_RGB == False:

            img_save_1 = img[t_num,z_num, :, :] 
            img_save_1 = gray2rgb(img_save_1)

            img_save_2 = img[t_num,z_num+1, :, :] 
            img_save_2 = gray2rgb(img_save_2)

            img_save_3 = img[t_num,z_num+2, :, :] 
            img_save_3 = gray2rgb(img_save_3)

        if use_RGB == True:
            img_save_1 = img[t_num,z_num, :, :, :] 
            # img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            img_save_2 = img[t_num,z_num+1, :, :, :] 
            # img_save_2 = convert(img_save_2, 0, 255, np.uint8)
            img_save_3 = img[t_num,z_num+2, :, :, :] 
            # img_save_3 = convert(img_save_3, 0, 255, np.uint8)
        # saving images as PNG
        io.imsave("{}.png".format("im1"), img_save_1)
        io.imsave("{}.png".format("im2"), img_save_2)
        io.imsave("{}.png".format("im3"), img_save_3)

        print("{}/{}\n".format(file_folder,slice_folder))
    return sequence_path



##############################################
##############################################
##############################################
##############################################

# provide source image path
Source_path = r"E:\TRAINING_DATA\Cell_data_training_data\8 Bit BF\512" 

# provide image size

img_size = 512 #@"64", "128", "256", "512", "1024", "2048"


Parent_path = "\\".join(Source_path.split("\\")[:-1])

if img_size<=512:
  divisor = img_size
else: 
  divisor = 512



# Write a log file for keeping track of the names
log_path_file = Parent_path + "\\" + "split_log.csv"

with open(log_path_file, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["file_name", "split_name"])

# Get all the paths of the images
img_list = [f for f in os.listdir(Source_path) if f.endswith('.tif')]

# Create parent-folder where each split execution will be saved
aug_saving_path = Parent_path+'\\spit_source_DAIN'
if not os.path.exists(aug_saving_path):
  os.mkdir(aug_saving_path)


# changes the dimensions for the listdatasets.py file scriptn - might not be needed
# manipulate_listdatasets(divisor)

# create a folder where split images are being stored
# dirname_path = os.path.dirname((Source_path))
split_img_folder_path = make_folder_with_date(aug_saving_path, "split")

split_img_small(img_list, Source_path, divisor, split_img_folder_path, log_path_file)

print(bcolors.WARNING + "Finished section 1: split_img_small ")



##############################################
#@title Prepare data for Training or Interpolation


# create new folder
save_location = "/".join(split_img_folder_path.split("/")[:-1])
folder_name = f"{divisor}_img_separation"
save_location = os.path.join(save_location,folder_name)
if not os.path.exists(save_location):
  os.mkdir(save_location)
os.chdir(save_location)  

#param ["prep_t_train", "prep_z_train", "prep_predict_t", "prep_predict_z", "upsample_t", "upsample_z"]
folder_option = "prep_t_train" 
split_training_test = 0.01

# split_folder, folder_steps, folder_gt, train_folder, sub_save_location
_, _, _, _, sub_save_location = data_train_test_preparation(folder_option, split_img_folder_path, save_location, split_training_test)
print(bcolors.WARNING + "Finished section 2: data train test preparation")

###############################################
