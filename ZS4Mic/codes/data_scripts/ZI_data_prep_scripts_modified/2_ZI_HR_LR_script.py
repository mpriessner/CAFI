import shutil
import os
import sys
import cv2
from tqdm import tqdm
import os.path as osp
import cv2
import numpy as np
import torch
import math

#### Load the necessary functions
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()

def create_folder_list_from_txt_guide(testlist_txt, trainlist_txt):
    print("Execute: create_folder_list_from_txt_guide")
    list_path_list = []
    with open(testlist_txt, "r") as f:
      for line in f:
        list_path_list.append(line)
    with open(trainlist_txt, "r") as f:
      for line in f:
        list_path_list.append(line)
    list_path_list.sort()

    with open(r"E:\master_sep_guide.txt", "w") as temp:
      for line in list_path_list:
        temp.write(line)

def prep_folder_structure(new_path):
  '''this function creates the same folder and subfolder structure as provided in the sequences folder in a 
  new given new_location path based on a master_sep_guide.txt file which recombined all folders from test and train'''
  print(f"Prepare Folder structure: {new_path}")
  with open(r"E:\master_sep_guide.txt", "r") as temp:
    for line in tqdm(temp):
        one = line[:-1].split("/")[0]
        two = line[:-1].split("/")[1]
        folder_1 = os.path.join(new_path, one)
        if not os.path.exists(folder_1):
          os.mkdir(folder_1)
          folder_2 = os.path.join(folder_1, two)
          os.mkdir(folder_2)
        else:
          folder_2 = os.path.join(folder_1, two)
          os.mkdir(folder_2)

def get_all_filepaths(input_path, N_frames):
    '''This function gets the paths based on the folder and the N_frames provided'''
    print("Execute: get_all_filepaths")
    flist = []
    with open(r"E:\master_sep_guide.txt", "r") as temp:
      for line in tqdm(temp):
        one = line[:-1].split("/")[0]
        two = line[:-1].split("/")[1]
        line = one + "\\" + two
        folder_path = os.path.join(input_path,line)
        for i in range(1,N_frames+1):
          file_name = f"im{i}.png"
          file_path = os.path.join(folder_path, file_name)
          flist.append(file_path)
    return flist

def generate_mod_LR(up_scale, sourcedir, savedir, train_guide, test_guide, continue_loading, N_frames):
    """This function generates the high and low resulution images in a given output folder"""

    create_folder_list_from_txt_guide(train_guide, test_guide)

    save_HR = os.path.join(savedir, 'HR')
    save_LR = os.path.join(savedir, 'LR')
 
    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(up_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
      
    # Create folder system
    if continue_loading == False:
        print("Restart loading")
        if os.path.isdir(savedir):
          shutil.rmtree(savedir)
          os.mkdir(savedir)
        else:
          os.mkdir(savedir)
          
        log_path = os.path.join(savedir, "HR_LR_log.txt")
        with open(log_path, "w") as f:
            f.write("start")
            
        os.mkdir(save_HR)
        os.mkdir(save_LR)
        
        os.mkdir(saveHRpath)
        prep_folder_structure(saveHRpath)

        os.mkdir(saveLRpath)
        prep_folder_structure(saveLRpath)

        # copy the set_guide text files in each folder (HR, LR)
        train_guide_HR = saveHRpath[:-3]+"/sep_trainlist.txt"
        train_guide_LR = saveLRpath[:-3]+"/sep_trainlist.txt"

        test_guide_HR = saveHRpath[:-3]+"/sep_testlist.txt"
        test_guide_LR = saveLRpath[:-3]+"/sep_testlist.txt"

        shutil.copy(train_guide, train_guide_HR)
        shutil.copy(train_guide, train_guide_LR)

        shutil.copy(test_guide, test_guide_HR)
        shutil.copy(test_guide, test_guide_LR)

        with open(log_path, "a") as f:
            f.write(f'Created new folders: {savedir} \n')
            f.write(f'Created new folders: {save_HR}\n')
            f.write(f'Created new folders: {save_LR}\n')
            f.write(f'Created new folders: {saveHRpath}\n')
            f.write(f'Created new file: {train_guide_HR}\n')
            f.write(f'Created new file: {test_guide_LR}\n')
    else:
        log_path = os.path.join(savedir, "HR_LR_log.txt")
        with open(log_path, "w") as f:
            f.write("start")

    filepaths = get_all_filepaths(sourcedir, N_frames)
    print(f"number of files: {len(filepaths)}")
    num_files = len(filepaths)

 # # prepare data with augementation
    for i in tqdm(range(num_files)):
        filename = filepaths[i]
        file_folder_path = filename[-18:]
        # check if file was already processed
        file_checker_path = r"{}\\{}".format(saveHRpath, file_folder_path)
        if os.path.exists(file_checker_path):
          with open(log_path, "a") as f:
            f.write(f"File already exists: {file_checker_path}\n")
          continue
        else: 
          try:
            with open(log_path, "a") as f:
              f.write('No.{} -- Processing {}\n'.format(i, filename))
            # read image
            image = cv2.imread(filename)

            width = int(np.floor(image.shape[1] / up_scale))
            height = int(np.floor(image.shape[0] / up_scale))
            # modcrop
            if len(image.shape) == 3:
                image_HR = image[0:up_scale * height, 0:up_scale * width, :]
            else:
                image_HR = image[0:up_scale * height, 0:up_scale * width]
            # LR
            image_LR = em_AG_D_sameas_preprint(image_HR, scale=up_scale, upsample=False) 

            image_LR = imresize_np(image_HR, 1 / up_scale, True)
            file_folder_path = filename[-18:]
            path_HR = r"{}\\{}".format(saveHRpath, file_folder_path)
            path_LR = r"{}\\{}".format(saveLRpath, file_folder_path)
            cv2.imwrite(path_HR, image_HR)
            cv2.imwrite(path_LR, image_LR)
          except:
            with open(log_path, "a") as f:
              f.write('No.{} -- failed {}\n'.format(i, filename))     

    return save_HR, save_LR

from scipy.ndimage.interpolation import zoom as npzoom
from skimage import filters
from skimage.util import random_noise, img_as_ubyte, img_as_float
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.transform import rescale
import PIL
from skimage import io
import numpy as np

def em_AG_D_sameas_preprint(x, scale, upsample=False):
     lvar = filters.gaussian(x, sigma=3)
    if len(x.shape) == 3:
        x_dim, y_dim, c = x.shape
        x1 = x[:,:,0]
        lvar = filters.gaussian(x1, sigma=3)

        x1 = random_noise(x1, mode='localvar', local_vars=(lvar+0.0001)*0.05)
        # x_down1 = npzoom(x1, 1/scale, order=1)
        
        img_temp = np.zeros((int(x_dim),int(y_dim),c))
        img_temp[:,:,0] = x1
        img_temp[:,:,1] = x1
        img_temp[:,:,2] = x1
    if len(x.shape) == 2:
        x_dim, y_dim = x.shape
        x1 = random_noise(x, mode='localvar', local_vars=(lvar+0.0001)*0.05)
        # x_down1 = npzoom(x1, 1/scale, order=1)
        
        img_temp = np.zeros((int(x_dim),int(y_dim),3))
        img_temp[:,:,0] = x1
        img_temp[:,:,1] = x1
        img_temp[:,:,2] = x1
 
    x_down = img_temp
    return x_down#, x_up


# for testing
# import cv2
# file = r"Z:\Martin Priessner\XXX__External_dataset\training_EM\trainsources\HR_1_stacks\sequences_Gauss_3\00001\0038\im1.png"
# image = cv2.imread(file)
# image.dtype
# # image = image[:,:,0]
# scale = 4
# # x = image
# image.shape
# image.min()
# image.max()
# x_down= em_AG_D_sameas_preprint(image, scale=4, upsample=False)
# x_down.shape
# io.imsave(r"Z:\Martin Priessner\XXX__External_dataset\training_EM\trainsources\HR_1_stacks\x_down2.png", x_down)
# lvar.min()



######################## SELECT SOURCE FOLDER ########################
### For "inPath" select the folder where the "sequences" folder and the two txt-files ("sep_trainlist.txt" and "sep_testlist.txt") are located.**
inPath = r'E:\Outsourced_Double\BF_data_for_training\SRFBN'#@param {type:"string"}

sequences_path = os.path.join(inPath, "sequences")
# test_or_train = "test"#@param ["test", "train"]

######################## SELECT SAVE LOCATION FOLDER ########################
outPath = r"E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024"

######################## SELECT SCALE FACTOR ########################
scale_factor = 4 
if not os.path.exists(outPath):
    os.makedirs(outPath)
train_guide = os.path.join(inPath, "sep_trainlist.txt")
test_guide = os.path.join(inPath, "sep_testlist.txt")

# in case if the process stopped for some reason you can select continue_loading = TRUE to continue the preparation
continue_loading = False 
N_frames = 7

save_HR, save_LR = generate_mod_LR(scale_factor, sequences_path, outPath, train_guide, test_guide,continue_loading, N_frames)

