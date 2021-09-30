import os
from skimage import io
import glob 
import cv2
from tqdm import tqdm
import os
import shutil
from aicsimageio import AICSImage, imread
from aicsimageio.transforms import reshape_data
from aicsimageio.writers import png_writer 
import numpy as np
import h5py


def get_file_list(folder_path):
  """This function takes a folder_path and returns a list of files sorted"""
  # get a list of files in the folder
  flist = os.listdir(folder_path)
  flist.sort()
  return flist

def get_folder_list(source_path):
  """ This function creates a list of folders from a given source path"""
  folder_list = [x[0] for x in os.walk(source_path)]
  folder_list.sort()
  folder_list = folder_list[1:]
  return folder_list




def save_image(temp_img, folder_option, file_count, save_location_image, file_name, zoomfactor, tz_dim, use_RGB):
  """ This function saves the temp image and re-structures the channels in the right order for the z-dimension"""
  # remove the first slice of zeros
  # print(f"save temp_img: {temp_img.shape}")
  if use_RGB:
    temp_img_final = temp_img[1:,:,:,:,:]
  else:
    temp_img_final = temp_img[1:,:,:,:]
    
  if folder_option == "upsample-z" or folder_option == "downsample-z":
    if (tz_dim % 2) == 0: # this is necessary because if the number is uneven then i added the last image, which otherwise would have been ignored and therefore I need to remove it here again to recreate the same dimensions as the input image
      if use_RGB:
        temp_img_final = temp_img_final[:,:-1,:,:,:] # remove the last image to get the same dimensions
      else: 
        temp_img_final = temp_img_final[:,:-1,:,:] # remove the last image to get the same dimensions
    io.imsave(save_location_image+f"/ZI_{zoomfactor}x_{file_name}_Z.tif", temp_img_final)

  elif folder_option == "upsample-t" or folder_option == "downsample-t" :
    temp_img_final = np.swapaxes(temp_img_final, 0, 1)
    if (tz_dim % 2) == 0:  # this is necessary because if the number is uneven then i added the last image, which otherwise would have been ignored and therefore I need to remove it here again to recreate the same dimensions as the input image
      if use_RGB:
        temp_img_final = temp_img_final[:-1,:,:,:,:] # remove the last image to get the same dimensions
      else:
        temp_img_final = temp_img_final[:-1,:,:,:] # remove the last image to get the same dimensions
    io.imsave(save_location_image+f"/ZI_{zoomfactor}x_{file_name}_T.tif", temp_img_final)

  elif folder_option == "zoom":
    temp_img_final = np.swapaxes(temp_img_final, 0, 1)
    io.imsave(save_location_image+f"/Z_{zoomfactor}x_{file_name}.tif", temp_img_final)
   

def save_as_h5py(img_list, fraction_list, zt_list, file_nr, interpolate_location, multiplyer, product_image_shape, use_RGB):
    '''this function saves the the single images of each 4D file into one h5py file'''
    zt_dim = len(zt_list)
    xy_dim = int(product_image_shape/multiplyer)
    h5py_safe_location_list = []
    
    # saving all the images in the xyz dimension in a h5py file
    for image in img_list:
      h5py_safe_location = f"/content/ZoomInterpolation/results/{image}.hdf5"
      h5py_safe_location_list.append(h5py_safe_location)
      with h5py.File(h5py_safe_location, 'w') as f:
        
        # for permutation in permutation_list:
        for zt in tqdm(zt_list):

          if use_RGB:
            temp_img_3D = np.zeros((len(file_nr), multiplyer*xy_dim, multiplyer*xy_dim, 3))
          else:
            temp_img_3D = np.zeros((len(file_nr), multiplyer*xy_dim, multiplyer*xy_dim))
            
          for single_file_nr, single_file in enumerate(file_nr):

            if use_RGB:
                temp_img_2D = np.zeros((multiplyer*xy_dim, multiplyer*xy_dim, 3))
            else:
                temp_img_2D = np.zeros((multiplyer*xy_dim, multiplyer*xy_dim))
            
            counter_x = 0
            counter_y = 0
            for num, fraction in enumerate(fraction_list):

                if counter_x == multiplyer:
                  counter_x = 0
                  counter_y+=1

                key = f"{image}_{fraction}_{zt}/{single_file}"
                img_path = os.path.join(interpolate_location, key)
                img = io.imread(img_path)

                if use_RGB == True:
                  img = img # otherwise there are 3 channels
                else:
                  img = img[:,:,0] # otherwise there are 3 channels

                img = img.astype('uint8')

                if use_RGB:
                  temp_img_2D[counter_y*xy_dim:(counter_y+1)*xy_dim,counter_x*xy_dim:(counter_x+1)*xy_dim,:] = img
                else:
                  temp_img_2D[counter_y*xy_dim:(counter_y+1)*xy_dim,counter_x*xy_dim:(counter_x+1)*xy_dim] = img

                counter_x += 1

            if use_RGB:
              temp_img_3D[single_file_nr,:,:,:] = temp_img_2D
            else:  
              temp_img_3D[single_file_nr,:,:] = temp_img_2D

          name = f"{image}_{zt}"
          f.create_dataset(f"{name}", data=np.array(temp_img_3D, dtype=np.uint8))

    return h5py_safe_location_list

