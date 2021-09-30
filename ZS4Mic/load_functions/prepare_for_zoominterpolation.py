#WORKING
import os
import sys
sys.path.insert(0,'/content/ZoomInterpolation/load_functions')
from skimage import io
import numpy as np
from tqdm import tqdm
import shutil
from aicsimageio import AICSImage, imread
import time
import random
from aicsimageio import AICSImage, imread
from aicsimageio.writers import png_writer 
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from timeit import default_timer as timer
import imageio
import tifffile 
from aicsimageio.transforms import reshape_data
from datetime import datetime

def downsample_z_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]

    #create new directory-path
    for num_t in tqdm(range(0,t)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "t-%03d"%(num_t)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder)
        os.chdir(folder)
        for num_z in range(z):
          if (num_z % 2) == 0:
            #create new directory-path
            file_name = ("dz_%03d" %(num_z))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning
            if use_RGB == False:
              img_save_1 = img[num_t,num_z, :, :] 
              img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            elif use_RGB == True:
              img_save_1 = img[num_t,num_z, :, :, :] 
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
              # # saving images as PNG
            io.imsave("{}.png".format(file_name), img_save_1)

          #save the last slide on top labeled with x
          if num_z == z-1 and (num_z % 2) != 0:
            file_name = ("dz_%03d" %(num_z))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning 
            if use_RGB == False:
              img_save_1 = img[num_t,num_z, :, :] 
              img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            elif use_RGB == True:
              img_save_1 = img[num_t,num_z, :, :, :] 
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)              
            # # saving images as PNG
            io.imsave("{}-x.png".format(file_name), img_save_1)


def downsample_t_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]

    #create new directory-path
    for num_z in tqdm(range(0,z)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "z-%03d"%(num_z)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder)
        os.chdir(folder)
        for num_t in range(t):
          if (num_t % 2) == 0:
            #create new directory-path
            file_name = ("dt_%03d" %(num_t))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning
            if use_RGB == False:
              img_save_1 = img[num_t,num_z, :, :] 
              img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            elif use_RGB == True:
              img_save_1 = img[num_t,num_z, :, :, :] 
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
              # # saving images as PNG
            io.imsave("{}.png".format(file_name), img_save_1)

          #save the last slide on top labeled with x
          if num_t == t-1 and (num_t % 2) != 0:
            file_name = ("dt_%03d" %(num_t))

            if use_RGB == False:
              # #here put the image pngs into the folder (instead of creating the folder)
              # #convert image to unit8 otherwise warning
              img_save_1 = img[num_t,num_z, :, :] 
              img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            elif use_RGB == True:
              img_save_1 = img[num_t,num_z, :, :, :] 
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
              # # saving images as PNG
            io.imsave("{}-x.png".format(file_name), img_save_1)

            
def upsample_t_creation(img_path_list, file_num, sub_save_location, folder_option):
    # to differentiate between zoom and normal upsampling in t dim
    if folder_option =="zoom":
      marker = "z"
    else:
      marker = "u"
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_path = img_path_list[file_num]
    img_nr = img_path.split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path.split("/")[-1].split(".")[0].split("-")[2][:2]
    
    #create new directory-path
    for num_z in tqdm(range(0,z)):   # dim_2 = zdimension
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "z-%03d"%(num_z) # z doesn't need to be the z dimension because it is also used for the t dimension
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder_name)
        os.chdir(folder_name)
        for num_t in range(t):
          #create new directory-path
          file_name = (f"{marker}t_%03d" %(num_t))

          # #here put the image pngs into the folder (instead of creating the folder)
          # #convert image to unit8 otherwise warning

          if use_RGB == False:
            img_save_1 = img[num_t,num_z, :, :] 
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
          elif use_RGB == True:
            img_save_1 = img[num_t,num_z, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            # # saving images as PNG
          io.imsave("{}.png".format(file_name), img_save_1)
            # writer1.save(img_save_1)



def upsample_z_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num]) #dim_1=t, dim_2=z
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]
    # folder_file_path = os.path.join(sub_save_location,file_to_folder_name)
    # os.mkdir(folder_file_path)

    #create new directory-path
    for num_t in tqdm(range(0,t)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "z-%03d"%(num_t)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder_name)
        os.chdir(folder_name)
        for num_z in range(z):
          #create new directory-path
          file_name = ("uz_%03d"%(num_z))
          # #convert image to unit8 otherwise warning

          if use_RGB == False:
            img_save_1 = img[num_t,num_z, :, :] 
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
          elif use_RGB == True:
            img_save_1 = img[num_t,num_z, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            # # saving images as PNG
          io.imsave("{}.png".format(file_name), img_save_1)

            
def get_img_path_list(img_path_list, img_folder_path):
  ''' Creates a list of image-path that will be used for loading the images later'''
  flist = os.listdir(img_folder_path)
  flist.sort()
  for i in flist:
    img_slice_path = os.path.join(img_folder_path, i)
    img_path_list.append(img_slice_path)
  return img_path_list
# img_path_list = get_img_path_list_T(img_path_list, filepath, folder_list)
# img_path_list


def load_img(img_path):
    img = io.imread(img_path)
    if img.shape[-1]==3:
      use_RGB = True
      t, z, y_dim, x_dim, _ = img.shape 
      print("This image will be processed as a RGB image")
    else:
      use_RGB = False
      t, z, y_dim, x_dim = img.shape 
    print("The image dimensions are: " + str(img.shape))
    return t, z, y_dim,x_dim, img, use_RGB
  
    
def make_folder_with_date(save_location, name):
  today = datetime.now()
  if today.hour < 12:
    h = "00"
  else:
    h = "12"
  sub_save_location = save_location + "/" + today.strftime('%Y%m%d')+ "_"+ today.strftime('%H%M%S')+ "_%s"%name
  os.mkdir(sub_save_location)
  return sub_save_location


def create_3D_image(img, x_dim, y_dim):
# creates 3D image with 3 times the same values for RGB because the NN was generated for normal rgb images dim(3,x,y)
  # print(img.shape)
  image_3D = np.zeros((x_dim,y_dim,3))
  image_3D[:,:,0] = img
  image_3D[:,:,1] = img
  image_3D[:,:,2] = img
  return image_3D


def convert(img, target_type_min, target_type_max, target_type):
  # this function converts images from float32 to unit8 
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

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

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove


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

def correct_channels(img):
  '''For 2D + T (with or without RGB) a artificial z channel gets created'''
  if img.shape[-1] ==3:
    use_RGB = True
  else:
    use_RGB = False
  if len(img.shape) ==4 and use_RGB:
    t, x, y, c = img.shape
    zeros = np.zeros((t,1,y,x,c))
    zeros[:,0,:,:,:] = img
    img = zeros
  elif len(img.shape) ==3 and not use_RGB:
    t, x, y = img.shape
    zeros = np.zeros((t,1,y,x))
    zeros[:,0,:,:] = img
    img = zeros
  return img, use_RGB
    

def change_train_file(zoomfactor, model_path):
  """This function changes the resolution value in the file: Vimeo7_dataset.py"""
  file_path_2 = "/content/ZoomInterpolation/codes/test_new.py"
  fh_2, abs_path_2 = mkstemp()
  with fdopen(fh_2,'w') as new_file:
    with open(file_path_2) as old_file:
      for counter, line in enumerate(old_file):
        if counter ==27:
          new_file.write(f"    scale = {zoomfactor}\n")
        elif counter == 34:
          new_file.write(f"    model_path = '{model_path}'\n")
        else:
          new_file.write(line)
  copymode(file_path_2, abs_path_2)
  #Remove original file
  remove(file_path_2)
  #Move new file
  move(abs_path_2, file_path_2) 



#############################################################################




def data_preparation_for_zoominterpolation(folder_option, save_location, split_img_folder_path):
    if folder_option == "upsample-t":
      name = "upsample-t"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        upsample_t_creation(img_path_list, file_num, sub_save_location, folder_option)

    elif folder_option == "upsample-z":
      name = "upsample-z"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        upsample_z_creation(img_path_list, file_num, sub_save_location,)

    elif folder_option == "downsample-t":
      name = "downsample-t"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        downsample_t_creation(img_path_list, file_num, sub_save_location,)

    elif folder_option == "downsample-z":
      name = "downsample-z"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        downsample_z_creation(img_path_list, file_num, sub_save_location)
        
    elif folder_option == "zoom":
      name = "zoom"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        upsample_t_creation(img_path_list, file_num, sub_save_location, folder_option)
    return sub_save_location


def get_zoomfactor(zoomfactor):
    if zoomfactor == 16:
      zoomfactor_1 = 4
      zoomfactor_2 = 4
    if zoomfactor == 8:
      zoomfactor_1 = 4
      zoomfactor_2 = 2
    if zoomfactor == 4:
      zoomfactor_1 = 4
      zoomfactor_2 = 1
    if zoomfactor == 2:
      zoomfactor_1 = 2
      zoomfactor_2 = 1
    if zoomfactor == 1:
      zoomfactor_1 = 1
      zoomfactor_2 = 1
    return zoomfactor_1, zoomfactor_2


from preparation_for_training import change_Sakuya_arch
def prepare_files_for_zoominterpolation_step(sub_save_location, zoomfactor):
    
    img_folder_path_interpolate = sub_save_location
    shutil.rmtree("/content/ZoomInterpolation/test_example")
    shutil.copytree(img_folder_path_interpolate,"/content/ZoomInterpolation/test_example")
    os.chdir("/content/ZoomInterpolation/codes")

    # if use_fine_tuned_models:
    #   if zoomfactor ==1:
    #     change_train_file(zoomfactor, pretrained_model_path)
    #     change_Sakuya_arch(zoomfactor)
    #   elif zoomfactor ==2:
    #     change_train_file(zoomfactor, pretrained_model_path)
    #     change_Sakuya_arch(zoomfactor)
    #   elif zoomfactor ==4:
    #     change_train_file(zoomfactor, pretrained_model_path)
    #     change_Sakuya_arch(zoomfactor)
    # else:
    pretrained_model_path_1x = "/content/ZoomInterpolation/experiments/pretrained_models/pretrained_1x.pth"
    pretrained_model_path_2x = "/content/ZoomInterpolation/experiments/pretrained_models/pretrained_2x.pth"
    pretrained_model_path_4x = "/content/ZoomInterpolation/experiments/pretrained_models/pretrained_4x.pth"
    if zoomfactor ==1:
      change_train_file(zoomfactor, pretrained_model_path_1x)
      change_Sakuya_arch(zoomfactor)
    elif zoomfactor ==2:
      change_train_file(zoomfactor, pretrained_model_path_2x)
      change_Sakuya_arch(zoomfactor)
    elif zoomfactor ==4:
      change_train_file(zoomfactor, pretrained_model_path_4x)
      change_Sakuya_arch(zoomfactor)
    return img_folder_path_interpolate

import sys
sys.path.insert(0,'/content/ZoomInterpolation/load_functions')
from reconstruct_image import get_file_list
from reconstruct_image import get_folder_list
from reconstruct_image import save_image
from reconstruct_image import save_as_h5py
# from prepare_dataset_test_folders import make_folder_with_date
import os
import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np
from timeit import default_timer as timer
from datetime import datetime
import h5py
import math
from pympler import asizeof

def save_interpolated_image(interpolate_location, Saving_path, log_path_file, divisor, zoomfactor, folder_option, use_RGB):

    # create a list of the identifyer for 
    img_list         = []
    fraction_list    = []
    zt_list          = []

    # Get all the different identifier from the foldername
    # which provides the information of how many images and 
    # dimensions the reconstructed image will have
    folder_list = get_folder_list(interpolate_location)
    folder_name_list = [i.split("/")[-1] for i in folder_list]
    for folder_name in folder_name_list:
      image_nr =        folder_name.split("_")[0][:]
      fraction_nr =     folder_name.split("_")[1][:]
      # permutation_nr =  folder_name.split("_")[2][:]
      zt_nr =           folder_name.split("_")[2][:]
      file_nr =         os.listdir(folder_list[0])
      file_nr.sort()
      if image_nr not in img_list:
        img_list.append(image_nr)
      if fraction_nr not in fraction_list:
        fraction_list.append(fraction_nr)
      # if permutation_nr not in permutation_list:
      #   permutation_list.append(permutation_nr)
      if zt_nr not in zt_list:
        zt_list.append(zt_nr)

    # find the dimensions of the reconstructed image (important for big images
    # that are split in smaller pieces) 
    # find out what is the output dimension of the image
    img_multiplyer = len(fraction_list)
    if img_multiplyer == 1:
        multiplyer = 1
        product_image_shape = divisor* multiplyer*zoomfactor
    elif img_multiplyer == 4:
        multiplyer = 2
        product_image_shape = divisor * multiplyer*zoomfactor
    elif img_multiplyer == 16:
        multiplyer = 4
        product_image_shape = divisor *multiplyer*zoomfactor

    print(f"img_list is: {img_list}")
    print(f"fraction_list is: {fraction_list}")
    # print(f"Permutation_list is: {permutation_list}")
    print(f"zt_list is: {zt_list}")
    print(f"Product_image_shape is: {product_image_shape}")
    print(f"Files is: {file_nr}")
    print(f"File_nr is: {len(file_nr)}")

    print(f"Folder_list is: {folder_list}")
    print(f"Image shape is: {product_image_shape}")

    # Save all images in one big h5py-file per image 
    h5py_safe_location_list = save_as_h5py(img_list, fraction_list, zt_list, file_nr, interpolate_location, multiplyer, product_image_shape, use_RGB)
    print(f"There are {len(h5py_safe_location_list)} images to reconstruct")

    #@markdown If the notebook crashes because of insufficient RAM 
    #@markdown you can select a critical size of GB of the reconstructed images which will prevent the crashing of the notebook.

    # Create folder where reconstructed images are stored (depending on the mode)
    if folder_option == "downsample-t" or folder_option == "upsample-t":
      save_location_image = make_folder_with_date(Saving_path, "t_interpolation")
    elif folder_option == "downsample-z" or folder_option == "upsample-z":
      save_location_image = make_folder_with_date(Saving_path, "z_interpolation")
    elif folder_option == "zoom":
      save_location_image = make_folder_with_date(Saving_path, "zoom")

    # Read log-file for naming the files correctly
    df_files = pd.read_csv(log_path_file)

    #----------------Save Image Stack as TIF file from h5py------------------------------#
    file_count = 0 # necessarey in case the file is split because of a too big size
  #  available_ram = 8 #@param {type:"slider", min:1, max:20, step:1}

    #@markdown The reconstructed files will be saved in a new folder in the provided source_path labelled with mode, date and time.

    for h5py_safe_location in tqdm(h5py_safe_location_list):
      # available_ram = 8 # if out of ram error this value can be changed
      with h5py.File(h5py_safe_location, 'r') as f:
          file_name = df_files.at[file_count, 'file_name']
          list_keys = list(f.keys())

          if use_RGB:
            tz_dim, xy_dim, xy_dim, channels = f[list_keys[0]].shape  
            if folder_option == "zoom":       
                tz_dim = math.ceil(tz_dim/2)  # half the dimension because it takes every second image from the t-stack
            temp_img = np.zeros((1 ,tz_dim, xy_dim, xy_dim, channels)).astype('uint8')
          else:
            tz_dim, xy_dim,xy_dim = f[list_keys[0]].shape  
            if folder_option == "zoom":       
                tz_dim = math.ceil(tz_dim/2)  # half the dimension because it takes every second image from the t-stack
            temp_img = np.zeros((1 ,tz_dim, xy_dim, xy_dim)).astype('uint8')
          # import IPython; IPython.embed();# exit(1)

          for image in f.values():
            if folder_option == "zoom":  
              if use_RGB:
                image = image[::2,:,:,:] # take every second image in the t dimension
              else:
                image = image[::2,:,:] # take every second image in the t dimension

            # if asizeof.asizeof(temp_img) < available_ram*1000000000:
            if use_RGB:
              temp_img = np.append(temp_img,[image[:,:,:,:]],axis=0)
            else:
              temp_img = np.append(temp_img,[image[:,:,:]],axis=0)
              # print(asizeof.asized(temp_img, detail=1).format())
            # save_image(temp_img, folder_option, file_count, save_location_image, file_name, zoomfactor, tz_dim, use_RGB)
              # temp_img = np.zeros((1 ,tz_dim, xy_dim,xy_dim)).astype('uint8')
              # temp_img = np.append(temp_img,[image[:,:,:]],axis=0)
            # import IPython; IPython.embed();# exit(1)

          save_image(temp_img, folder_option, file_count, save_location_image, file_name, zoomfactor, tz_dim, use_RGB)
          file_count += 1

    return save_location_image
