import os
import shutil
import time
import numpy
import random
import numpy as np
from tqdm import tqdm
from google.colab.patches import cv2_imshow
from tqdm import tqdm
from timeit import default_timer as timer
import imageio
import tifffile 
from datetime import datetime
import math
from skimage import io

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
    zeros = np.zeros((t,1,y,x,c))
    zeros[:,0,:,:,:] = img
    img = zeros
  elif len(img.shape) ==3 and not use_RGB:
    t, x, y = img.shape
    zeros = np.zeros((t,1,y,x))
    zeros[:,0,:,:] = img
    img = zeros
  return img, use_RGB


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


''' the following funcitons prepare the dataset in a way that it form the necessary folder system for the NN to handle the data correctly
'''
def upsample_z(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])
    folder_steps = "steps"
    split_folder = os.path.join(sub_save_location,folder_steps)
    if not os.path.exists(split_folder):
      os.mkdir(split_folder)

    fraction_num = img_path_list[file_num][-6:-4]

    #create new directory-path
    for t_num in tqdm(range(0,t)):
        for z_num in range(z-1):
          #create new directory-path
          file_folder = ("i-%03d" %(file_num) + f"_f-{fraction_num}" + "_t-%03d" %(t_num) +"_z-%03d"%(z_num))
          os.chdir(split_folder)
          os.mkdir(file_folder)
          steps_path_folder = os.path.join(split_folder, file_folder)
          os.chdir(steps_path_folder)

          # #here put the image pngs into the folder (instead of creating the folder)
          # #convert image to unit8 otherwise warning
          if use_RGB == False:
            img_save_1 = img[t_num,z_num, :, :] 
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)

            img_save_2 = img[t_num,z_num+1, :, :] 
            img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          if use_RGB == True:
            img_save_1 = img[t_num,z_num, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)

            img_save_2 = img[t_num,z_num+1, :, :, :] 
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          # # saving images as PNG
          io.imsave("{}.png".format("im1"), img_save_1)
          io.imsave("{}.png".format("im3"), img_save_2)
    return folder_steps


def upsample_t(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])
    folder_steps = "steps"
    split_folder = os.path.join(sub_save_location,folder_steps)
    if not os.path.exists(split_folder):
      os.mkdir(split_folder)

    fraction_num = img_path_list[file_num][-6:-4]

    images_jump =2
    #create new directory-path
    for z_num in tqdm(range(0,z)):
        for t_num in range(t-1):
          #create new directory-path
          file_folder = ("i-%03d" %(file_num) + f"_f-{fraction_num}" + "_t-%03d" %(t_num) +"_z-%03d"%(z_num))
          os.chdir(split_folder)
          os.mkdir(file_folder)
          steps_path_folder = os.path.join(split_folder, file_folder)
          os.chdir(steps_path_folder)

          # #here put the image pngs into the folder (instead of creating the folder)
          # #convert image to unit8 otherwise warning
          if use_RGB == False:
            img_save_1 = img[t_num,z_num, :, :] 
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)

            img_save_2 = img[t_num+1,z_num, :, :] 
            img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          if use_RGB == True:
            img_save_1 = img[t_num,z_num, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)

            img_save_2 = img[t_num+1,z_num, :, :, :] 
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          # # saving images as PNG
          io.imsave("{}.png".format("im1"), img_save_1)
          io.imsave("{}.png".format("im3"), img_save_2)
    return split_folder

def perform_prep_predict_z_creation(img_path_list, file_num,  sub_save_location):
    """This function creates a foldersystem that can be used to perform a prediction on the netowrk - Check that again"""
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])

    folder_gt = "gt"
    folder_gt = os.path.join(sub_save_location,folder_gt)
    os.mkdir(folder_gt)
    folder_steps = "steps"
    folder_steps = os.path.join(sub_save_location,folder_steps)

    if not os.path.exists(folder_steps):
      os.mkdir(folder_steps)
    if not os.path.exists(folder_steps):
      os.mkdir(folder_gt)

    fraction_num = img_path_list[file_num][-6:-4]

    images_jump =2
    #create new directory-path
    for t_num in tqdm(range(0,t)):
        for z_num in range(math.ceil(z/images_jump)-1): # rounds up to then remove the last one to not overshoot in the counting
        #create new directory-path
          file_folder = ("i-%03d" %(file_num) + f"_f-{fraction_num}" + "_t-%03d" %(t_num) +"_z-%03d"%(z_num))
          os.chdir(folder_gt)
          os.mkdir(file_folder)
          os.chdir(folder_steps)
          os.mkdir(file_folder)
          GT_path_folder = os.path.join(folder_gt, file_folder)
          steps_path_folder = os.path.join(folder_steps, file_folder)
          os.chdir(steps_path_folder)

          #here put the image pngs into the folder (instead of creating the folder)
          #convert image to unit8 otherwise warning
          first = z_num* images_jump
          second = z_num*images_jump+images_jump
          if use_RGB == False:
            img_save_1 = img[t_num,first, :, :] 
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)

            img_save_3 = img[t_num,second, :, :] 
            img_save_3 = create_3D_image(img_save_3, x_dim, y_dim)
            img_save_3 = convert(img_save_3, 0, 255, np.uint8)

            img_save_2 = img[t_num,first+1, :, :] 
            img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          if use_RGB == True:
            img_save_1 = img[t_num,first, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)

            img_save_3 = img[t_num,second, :, :, :] 
            img_save_3 = convert(img_save_3, 0, 255, np.uint8)

            img_save_2 = img[t_num,first+1, :, :, :] 
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          # saving images as PNG
          io.imsave("{}.png".format("im1"), img_save_1)
          io.imsave("{}.png".format("im3"), img_save_3)
          os.chdir(GT_path_folder)
          io.imsave("{}.png".format("im2"), img_save_2)
    return folder_steps, folder_gt

def perform_prep_predict_t_creation(img_path_list, file_num,  sub_save_location, folder_option):
    """This function creates a foldersystem that can be used to perform a prediction on the netowrk - Check that again"""
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])

    folder_gt = "gt"
    folder_gt = os.path.join(sub_save_location,folder_gt)
    os.mkdir(folder_gt)
    folder_steps = "steps"
    folder_steps = os.path.join(sub_save_location,folder_steps)

    if not os.path.exists(folder_steps):
      os.mkdir(folder_steps)
    if not os.path.exists(folder_steps):
      os.mkdir(folder_gt)

    fraction_num = img_path_list[file_num][-6:-4]

    images_jump =2
    #create new directory-path
    for z_num in tqdm(range(0,z)):
        for t_num in range(math.ceil(t/images_jump)-1):
        #create new directory-path
          file_folder = ("i-%02d" %(file_num) + f"_f-{fraction_num}" + "_t-%03d" %(t_num) +"_z-%03d"%(z_num))
          os.chdir(folder_gt)
          os.mkdir(file_folder)
          os.chdir(folder_steps)
          os.mkdir(file_folder)
          GT_path_folder = os.path.join(folder_gt, file_folder)
          steps_path_folder = os.path.join(folder_steps, file_folder)
          os.chdir(steps_path_folder)

          #here put the image pngs into the folder (instead of creating the folder)
          #convert image to unit8 otherwise warning
          first = t_num* images_jump
          second = t_num*images_jump+images_jump
          if use_RGB == False:
            img_save_1 = img[first,z_num, :, :] 
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)

            img_save_3 = img[second,z_num, :, :] 
            img_save_3 = create_3D_image(img_save_3, x_dim, y_dim)
            img_save_3 = convert(img_save_3, 0, 255, np.uint8)

            img_save_2 = img[first+1,z_num, :, :] 
            img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          if use_RGB == True:
            img_save_1 = img[first,z_num, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)

            img_save_3 = img[second,z_num, :, :, :] 
            img_save_3 = convert(img_save_3, 0, 255, np.uint8)

            img_save_2 = img[first+1,z_num, :, :, :] 
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          # saving images as PNG
          io.imsave("{}.png".format("im1"), img_save_1)
          io.imsave("{}.png".format("im3"), img_save_3)
          os.chdir(GT_path_folder)
          io.imsave("{}.png".format("im2"), img_save_2)
    return folder_steps, folder_gt
      


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
          img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
          img_save_1 = convert(img_save_1, 0, 255, np.uint8)
          img_save_2 = img[t_num+1,z_num, :, :] 
          img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
          img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          img_save_3 = img[t_num+2,z_num, :, :] 
          img_save_3 = create_3D_image(img_save_3, x_dim, y_dim)
          img_save_3 = convert(img_save_3, 0, 255, np.uint8)
        if use_RGB == True:
          img_save_1 = img[t_num,z_num, :, :, :] 
          img_save_1 = convert(img_save_1, 0, 255, np.uint8)
          img_save_2 = img[t_num+1,z_num, :, :, :] 
          img_save_2 = convert(img_save_2, 0, 255, np.uint8)
          img_save_3 = img[t_num+2,z_num, :, :, :] 
          img_save_3 = convert(img_save_3, 0, 255, np.uint8)
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
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            img_save_2 = img[t_num,z_num+1, :, :] 
            img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
            img_save_3 = img[t_num,z_num+2, :, :] 
            img_save_3 = create_3D_image(img_save_3, x_dim, y_dim)
            img_save_3 = convert(img_save_3, 0, 255, np.uint8)
        if use_RGB == True:
            img_save_1 = img[t_num,z_num, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            img_save_2 = img[t_num,z_num+1, :, :, :] 
            img_save_2 = convert(img_save_2, 0, 255, np.uint8)
            img_save_3 = img[t_num,z_num+2, :, :, :] 
            img_save_3 = convert(img_save_3, 0, 255, np.uint8)
        # saving images as PNG
        io.imsave("{}.png".format("im1"), img_save_1)
        io.imsave("{}.png".format("im2"), img_save_2)
        io.imsave("{}.png".format("im3"), img_save_3)

        print("{}/{}\n".format(file_folder,slice_folder))
    return sequence_path
