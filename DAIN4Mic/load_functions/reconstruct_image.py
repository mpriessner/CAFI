import os
from skimage import io
import glob 
import cv2
from tqdm import tqdm
import os
import shutil
import numpy as np
import h5py
import sys
import pandas as pd
_ = (sys.path.append("/usr/local/lib/python3.6/site-packages"))
sys.path.insert(0,'/content/DAIN/load_functions')
from prepare_split_images import make_folder_with_date

def prep_folder_for_resconstruction(folder_option, img_list, fraction_list, z_list, t_list, file_nr, interpolate_location, sub_folder_location):
    """This function just prepares a folderstructure where all the T or Z images are each in one folder together (same as in the zoomInterpolation script)"""
    if (folder_option == "upsample_t" or folder_option == "prep_predict_t"):
      for image in img_list:
        for fraction in fraction_list:
          for z in tqdm(z_list):
            counter = 0
            for t in t_list:
              for single_file_nr, single_file in enumerate(file_nr):
                  key_origin = f"{image}_{fraction}_{z}_{t}/{single_file}"
                  counter_files_name = "%04d.png"%counter
                  key_destination = f"{image}_{fraction}_{z}/{counter_files_name}"
                  img_path = os.path.join(interpolate_location, key_origin)

                  if single_file != file_nr[-1]:
                    img_path_origin = os.path.join(interpolate_location, key_origin)
                    img_path_destination = os.path.join(sub_folder_location, key_destination)
                    new_folder = "/".join(img_path_destination.split("/")[:-1])
                    if not os.path.exists(new_folder):
                      os.mkdir(new_folder)
                    shutil.move(img_path_origin, img_path_destination)
                    counter += 1

                  if t ==t_list[-1] and single_file == file_nr[-1]:
                    img_path_origin = os.path.join(interpolate_location, key_origin)
                    img_path_destination = os.path.join(sub_folder_location, key_destination) 
                    new_folder = "/".join(img_path_destination.split("/")[:-1])
                    if not os.path.exists(new_folder):
                      os.mkdir(new_folder) 
                    shutil.move(img_path_origin, img_path_destination)

    elif (folder_option == "upsample_z" or folder_option == "prep_predict_z"):
      for image in img_list:
        for fraction in fraction_list:
          for t in tqdm(t_list):
            counter = 0
            for z in z_list:
              for single_file_nr, single_file in enumerate(file_nr):
                  key_origin = f"{image}_{fraction}_{t}_{z}/{single_file}"
                  counter_files_name = "%04d.png"%counter
                  key_destination = f"{image}_{fraction}_{t}/{counter_files_name}"
                  img_path = os.path.join(interpolate_location, key_origin)

                  if single_file != file_nr[-1]:
                    img_path_origin = os.path.join(interpolate_location, key_origin)
                    img_path_destination = os.path.join(sub_folder_location, key_destination)
                    new_folder = "/".join(img_path_destination.split("/")[:-1])
                    if not os.path.exists(new_folder):
                      os.mkdir(new_folder)
                    shutil.move(img_path_origin, img_path_destination)
                    counter += 1

                  if z ==z_list[-1] and single_file == file_nr[-1]:
                    img_path_origin = os.path.join(interpolate_location, key_origin)
                    img_path_destination = os.path.join(sub_folder_location, key_destination)   
                    new_folder = "/".join(img_path_destination.split("/")[:-1])
                    if not os.path.exists(new_folder):
                      os.mkdir(new_folder)
                    shutil.move(img_path_origin, img_path_destination)



def save_image(temp_img, folder_option, save_location_image, file_name, use_RGB):
  """ This function saves the temp image and re-structures the channels in the right order for the z-dimension"""
  # remove the first slice of zeros
  # print(f"save temp_img: {temp_img.shape}")
  if use_RGB ==False:
    temp_img_final = temp_img[1:,:,:,:]
  else:
    temp_img_final = temp_img[1:,:,:,:,:]

  if folder_option == "upsample_z" or folder_option == "prep_predict_z":
    io.imsave(save_location_image+f"/{file_name}x_Z.tif", temp_img_final)

  elif folder_option == "upsample_t" or folder_option == "prep_predict_t" :
    temp_img_final = np.swapaxes(temp_img_final, 0, 1)
    io.imsave(save_location_image+f"/DAIN_{file_name[:-4]}_T.tif", temp_img_final)



def get_folder_list(source_path):
  """ This function creates a list of folders from a given source path"""
  folder_list = [x[0] for x in os.walk(source_path)]
  folder_list.sort()
  folder_list = folder_list[1:]
  return folder_list

def get_file_list(folder_path):
  """This function takes a folder_path and returns a list of files sorted"""
  # get a list of files in the folder
  flist = os.listdir(folder_path)
  flist.sort()
  return flist




def save_as_h5py(img_list, fraction_list, zt_list, file_nr, interpolate_location, multiplyer, product_image_shape, use_RGB):
    '''this function saves the the single images of each 4D file into one h5py file'''
    zt_dim = len(zt_list)
    xy_dim = int(product_image_shape/multiplyer)
    h5py_safe_location_list = []
    
    # saving all the images in the xyz dimension in a h5py file
    for image in img_list:
      h5py_safe_location = f"/content/DAIN/MiddleBurySet/other-result-author/{image}.hdf5"
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


def restructure_folder_for_processing(interpolate_location, Saving_path, log_path_file, divisor, folder_option, use_RGB):

  # create a list of the identifyer for 
  img_list         = []
  fraction_list    = []
  z_list          = []
  t_list          = []

  # Get all the different identifier from the foldername
  # which provides the information of how many images and 
  # dimensions the reconstructed image will have
  folder_list = get_folder_list(interpolate_location)
  folder_name_list = [i.split("/")[-1] for i in folder_list]

  for folder_name in folder_name_list:
    image_nr =        folder_name.split("_")[0][:]
    if image_nr not in img_list:
      img_list.append(image_nr)

    fraction_nr =     folder_name.split("_")[1][:]
    if fraction_nr not in fraction_list:
      fraction_list.append(fraction_nr)
    # permutation_nr =  folder_name.split("_")[2][:]
    # necessary because otherwise the reconstruction in T dimension goes wrong
    if folder_option == "upsample_z" or folder_option == "prep_z_train":
        z_nr =           folder_name.split("_")[-1][:]
        if z_nr not in z_list:
          z_list.append(z_nr)  

        t_nr =           folder_name.split("_")[-2][:]
        if t_nr not in t_list:
            t_list.append(t_nr)
    elif folder_option == "upsample_t" or folder_option == "prep_t_train":
        z_nr =           folder_name.split("_")[-2][:]
        if z_nr not in z_list:
          z_list.append(z_nr)  

        t_nr =           folder_name.split("_")[-1][:]
        if t_nr not in t_list:
            t_list.append(t_nr)
  file_nr = os.listdir(folder_list[0])
  file_nr.sort()


  #create processed path
  from prepare_dataset_train_test_folders import make_folder_with_date
  processed_path = "/content/DAIN/MiddleBurySet/processed"
  if not os.path.exists(processed_path):
    os.mkdir(processed_path)
  reprocessed_folder = make_folder_with_date(processed_path, "experiment")

  #reorder the folder - put all the T or Z images together in one folder
  prep_folder_for_resconstruction(folder_option, img_list, fraction_list, z_list, t_list, file_nr, interpolate_location, reprocessed_folder)
  return reprocessed_folder


def save_interpolated_image(interpolate_location, Saving_path, log_path_file, folder_option, divisor, use_RGB):

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
      product_image_shape = divisor* multiplyer
  elif img_multiplyer == 4:
      multiplyer = 2
      product_image_shape = divisor * multiplyer
  elif img_multiplyer == 16:
      multiplyer = 4
      product_image_shape = divisor *multiplyer


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


  # Create folder where reconstructed images are stored (depending on the mode)
  if folder_option == "prep_predict_t" or folder_option == "upsample_t":
    save_location_image = make_folder_with_date(Saving_path, "t_interpolation")
  elif folder_option == "prep_predict_z" or folder_option == "upsample_z":
    save_location_image = make_folder_with_date(Saving_path, "z_interpolation")


  # Read log-file for naming the files correctly
  df_files = pd.read_csv(log_path_file)

  #----------------Save Image Stack as TIF file from h5py------------------------------#

  # The reconstructed files will be saved in a new folder in the provided source_path labelled with mode, date and time.
  file_count = 0 # necessarey in case the file is split because of a too big size - not implmelmented anymore
  for h5py_safe_location in tqdm(h5py_safe_location_list):
    with h5py.File(h5py_safe_location, 'r') as f:
        file_name = df_files.at[file_count, 'file_name']
        list_keys = list(f.keys())

        if use_RGB:
          tz_dim, xy_dim, xy_dim, channels = f[list_keys[0]].shape  
          temp_img = np.zeros((1 ,tz_dim, xy_dim, xy_dim, channels)).astype('uint8')
        else:
          tz_dim, xy_dim, xy_dim = f[list_keys[0]].shape  
          temp_img = np.zeros((1 ,tz_dim, xy_dim, xy_dim)).astype('uint8')

        # image_count = 0
        # slice_count = 0
        for image in f.values():
        # if asizeof.asizeof(temp_img) < available_ram*1000000000:
          if use_RGB:
            temp_img = np.append(temp_img,[image[:,:,:,:]],axis=0)
          else:
            temp_img = np.append(temp_img,[image[:,:,:]],axis=0)
            # print(asizeof.asized(temp_img, detail=1).format())

          # else:
            # save_image(temp_img, folder_option, save_location_image, file_name, use_RGB)
            # slice_count +=1
            # if use_RGB:
            #   temp_img = np.zeros((1 ,tz_dim, xy_dim, xy_dim, channels)).astype('uint8')
            #   temp_img = np.append(temp_img,[image[:,:,:,:]],axis=0)
            # else:
            #   temp_img = np.zeros((1 ,tz_dim, xy_dim, xy_dim)).astype('uint8')          
            #   temp_img = np.append(temp_img,[image[:,:,:]],axis=0)

        save_image(temp_img, folder_option, save_location_image, file_name, use_RGB)
        file_count += 1
  return save_location_image



