import os
import os.path as osp
import sys
sys.path.insert(0,'/content/ZoomInterpolation/codes')
import cv2
import numpy as np
from data.util import imresize_np
import shutil
from skimage import io
from tqdm import tqdm

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

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

def split_test_train_sequences_data(inPath, outPath, guide):
  """This function splits the sequences folder into the test and train folder with the given format
  based on the guide txt files"""
  #if os.path.isdir(outPath):
  #  shutil.rmtree(outPath)
  f = open(guide, "r")
  lines = f.readlines()
  for l in tqdm(lines):
      line = l.replace('\n','')
      this_folder = os.path.join(inPath, line)
      dest_folder = os.path.join(outPath, line)
      if os.path.exists(dest_folder):
        print(f"Folder already moved: {dest_folder}")
      else:
        shutil.move(this_folder, dest_folder)
  print('Done')

def prep_folder_structure(new_path):
  '''this function creates the same folder and subfolder structure as provided in the sequences folder in a 
  new given new_location path based on a master_sep_guide.txt file which recombined all folders from test and train'''
  print(f"Prepare Folder structure: {new_path}")
  with open("/content/master_sep_guide.txt", "r") as temp:
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
    with open("/content/master_sep_guide.txt", "r") as temp:
      for line in tqdm(temp):
        folder_path = os.path.join(input_path,line[:-1])
        for i in range(1,N_frames+1):
          file_name = f"im{i}.png"
          file_path = os.path.join(folder_path, file_name)
          flist.append(file_path)
    return flist

def get_all_filepaths_in_folder(folder_path):
    '''This function gets the paths from each file in folder and subfolder of a given location'''
    flist = []
    for path, subdirs, files in tqdm(os.walk(folder_path)):
          for name in files:
            flist.append(os.path.join(path, name))
    return flist

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

    with open("/content/master_sep_guide.txt", "w") as temp:
      for line in list_path_list:
        temp.write(line)


def generate_mod_LR(up_scale, sourcedir, savedir, train_guide, test_guide, continue_loading, N_frames, log_path):
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
        with open(log_path, "w") as f:
            f.write("start")
        with open(log_path, "a") as f:
            f.write(f'Created new folders: {savedir} \n')
            f.write(f'Created new folders: {save_HR}\n')
            f.write(f'Created new folders: {save_LR}\n')
            f.write(f'Created new folders: {saveHRpath}\n')
            f.write(f'Created new file: {train_guide_HR}\n')
            f.write(f'Created new file: {test_guide_LR}\n')
    else:
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
        file_checker_path = os.path.join(saveHRpath, file_folder_path)
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
            image_LR = imresize_np(image_HR, 1 / up_scale, True)
            file_folder_path = filename[-18:]
            cv2.imwrite(os.path.join(saveHRpath, file_folder_path), image_HR)
            cv2.imwrite(os.path.join(saveLRpath, file_folder_path), image_LR)
          except:
            with open(log_path, "a") as f:
              f.write('No.{} -- failed {}\n'.format(i, filename))     

    return save_HR, save_LR


#############################Prepare LMBD data ##################################
import os,sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2
from tqdm import tqdm
sys.path.insert(0,'/content/ZoomInterpolation/codes')
import data.util as data_util
import utils.util as util


def reading_image_worker(path, key):
    '''worker for reading images'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)

def save_to_lmbd(img_folder, test_or_train, H_dst, W_dst, batch, mode, scale_factor):
    '''create lmdb for the Vimeo90K-7 frames dataset, each image with fixed size
    GT: [3, 256, 448]
        Only need the 4th frame currently, e.g., 00001_0001_4
    LR: [3, 64, 112]
        With 1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001
    '''
    #### configurations
    n_thread = 40

    # define the septest/trainlist & lmdb_save_path
    # path_parent = os.path.dirname(img_folder)

    if test_or_train == "test":
      txt_file = os.path.join(img_folder,"sep_testlist.txt")
      lmdb_save_path = os.path.join(img_folder, f"vimeo7_{test_or_train}_x{scale_factor}_{mode}.lmdb")
      img_folder_selected = os.path.join(img_folder, f"test_{scale_factor}")
      if os.path.isdir(lmdb_save_path):
        shutil.rmtree(lmdb_save_path)
    if test_or_train == "train":
      txt_file = os.path.join(img_folder,"sep_trainlist.txt")
      lmdb_save_path = os.path.join(img_folder, f"vimeo7_{test_or_train}_x{scale_factor}_{mode}.lmdb")
      img_folder_selected = os.path.join(img_folder, f"train_{scale_factor}")
      if os.path.isdir(lmdb_save_path):
        shutil.rmtree(lmdb_save_path)

    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in tqdm(train_l):
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        file_l = glob.glob(osp.join(img_folder_selected, folder, sub_folder) + '/*')
        all_img_list.extend(file_l)
        for j in range(7):
            keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'HR': 
        all_img_list = [v for v in all_img_list if v.endswith('.png')]
        keys = [v for v in keys]

    print('Calculating the total size of images...')
    data_size = sum(os.stat(v).st_size for v in all_img_list)

    #### read all images to memory (multiprocessing)
    print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
    
    #### create lmdb environment
    env = lmdb.open(lmdb_save_path, map_size=data_size * 30)
    txn = env.begin(write=True)  # txn is a Transaction object

    #### write data to lmdb
    #pbar = util.ProgressBar(len(all_img_list))

    i = 0
    for path, key in tqdm(zip(all_img_list, keys)):
        #pbar.update('Write {}'.format(key))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)        
        key_byte = key.encode('ascii')
        H, W, C = img.shape  # fixed shape
        assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, img)
        i += 1
        if  i % batch == 1:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish reading and writing {} images.'.format(len(all_img_list)))
            
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    if mode == 'HR':
        meta_info['name'] = 'Vimeo7_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'Vimeo7_train_LR7'
    meta_info['resolution'] = '{}_{}_{}'.format(3, H_dst, W_dst)
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = key_set
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'Vimeo7_train_keys.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

def change_Sakuya_arch(training_scale):
  """This function changes the network to perform a 4x 2x or no zoom magnification in: Sakuya_arch.py"""
  file_path_3 = "/content/ZoomInterpolation/codes/models/modules/Sakuya_arch.py"
  fh_3, abs_path_3 = mkstemp()
  with fdopen(fh_3,'w') as new_file:
    with open(file_path_3) as old_file:
      for counter, line in enumerate(old_file):
        if counter ==340:
          if training_scale == 4 or training_scale == 2:
            new_file.write("        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))\n")
          if training_scale ==1: 
            new_file.write("#        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))\n")
        elif counter ==341:
          if training_scale == 4:
            new_file.write("        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))\n")
          if training_scale == 1 or  training_scale == 2:
            new_file.write("#        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))\n")
        elif counter != 341 or counter != 340 :
          new_file.write(line)

  copymode(file_path_3, abs_path_3)
  #Remove original file
  remove(file_path_3)
  #Move new file
  move(abs_path_3, file_path_3) 

def change_dataset_file(HR_px, LR_px, training_scale, original_trainingset = False):
  """This function changes the resolution value in the file: Vimeo7_dataset.py"""
  if training_scale ==1:
    factor = 4
  elif training_scale ==2:
    factor = 2
  elif training_scale == 4:
    factor = 1
  file_path_2 = "/content/ZoomInterpolation/codes/data/Vimeo7_dataset.py"
  fh_2, abs_path_2 = mkstemp()
  with fdopen(fh_2,'w') as new_file:
    with open(file_path_2) as old_file:
      for counter, line in enumerate(old_file):
        if original_trainingset == False:
            if counter ==170:
              substi_1 = f"{HR_px}, {HR_px}"
              new_file.write("                img_GT = util.read_img(self.GT_env, key + '_{}'.format(v), (3, "+substi_1+"))\n")
            elif counter == 176:
              substi_2 = f"{LR_px}, {LR_px}"
              new_file.write("        LQ_size_tuple = (3, "+ substi_2 +") if self.LR_input else (3,"+ substi_1+")\n")
            else:
              new_file.write(line)
        elif original_trainingset == True:
            if counter ==170:
              substi_1 = f"{256}, {448}"
              new_file.write("                img_GT = util.read_img(self.GT_env, key + '_{}'.format(v), (3, "+substi_1+"))\n")
            elif counter == 176:
              substi_2 = f"{64*factor}, {112*factor}"
              new_file.write("        LQ_size_tuple = (3, "+ substi_2 +") if self.LR_input else (3,"+ substi_1+")\n")
            else:
              new_file.write(line)
  copymode(file_path_2, abs_path_2)
  #Remove original file
  remove(file_path_2)
  #Move new file
  move(abs_path_2, file_path_2) 


def change_train_yml(LMBD_HR, LMBD_LR, training_scale, cache_keys, niter, use_pretrained_model, pretrained_network_pth, pretrained_network_state, save_checkpoint_freq, warmup_iter, debug, learning_rate):
    """This function changes the parameters in the train_zml.yml files"""
    file_path = "/content/ZoomInterpolation/codes/options/train/train_zsm.yml"
    if training_scale == 4:
        GT_size = "128"
        LQ_size = "32" 
        scale = "4" 
        batch_size = "16" 
    elif training_scale == 2:
        GT_size = "128"
        LQ_size = "64" 
        scale = "2" 
        batch_size = "5" 
    elif training_scale == 1:
        GT_size = "64"
        LQ_size = "64" 
        scale = "1" 
        batch_size = "5"

    if use_pretrained_model == False:
      pretrained_1_sequence  = "  pretrain_model_G: ~\n"
      pretrained_2_sequence  = "  resume_state: ~\n"
    else:
      pretrained_1_sequence  = f"  pretrain_model_G: {pretrained_network_pth}\n" 
      pretrained_2_sequence  = f"  resume_state: {pretrained_network_state}\n"

    # change train_zsm.yml file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
      with open(file_path) as old_file:
        for counter, line in enumerate(old_file):
          if counter ==2:
            if debug == "True":
              new_file.write("name: LunaTokis_scratch_b16p32f5b40n7l1_600k_Vimeo_debug\n")
            else:
              new_file.write("name: LunaTokis_scratch_b16p32f5b40n7l1_600k_Vimeo\n")
          
          elif counter ==6:
            new_file.write(f"scale: {scale}\n") 
          elif counter ==17:
            new_file.write(f"    dataroot_GT: {LMBD_HR}\n")
          elif counter == 18:
            new_file.write(f"    dataroot_LQ: {LMBD_LR}\n")
          elif counter == 19:
            new_file.write(f"    cache_keys: {cache_keys}\n")
          elif counter == 24:
            new_file.write(f"    batch_size: {batch_size}\n")
          elif counter == 25:
            new_file.write(f"    GT_size: {GT_size}\n")
          elif counter == 26:
            new_file.write(f"    GT_size: {LQ_size}\n")
          elif counter == 44:
            new_file.write(pretrained_1_sequence)
          elif counter == 45:
            if scale !=4:
              new_file.write("  strict_load: false #true #\n") 
            else:       
              new_file.write("  strict_load: true #true #\n") 
          elif counter == 46:
            new_file.write(pretrained_2_sequence)
          elif counter == 50:
            new_file.write(f"  lr_G: {learning_rate}\n")
          elif counter == 54:
            new_file.write(f"  niter: {niter}\n")
          elif counter == 55:
            new_file.write(f"  warmup_iter: {warmup_iter} #4000  # -1: no warm up\n")
          elif counter == 70:
            new_file.write(f"  save_checkpoint_freq: {save_checkpoint_freq}")            
          else:
            new_file.write(line)
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)  
    
def change_train_file(backup_location):
  """This function changes the resolution value in the file: Vimeo7_dataset.py"""
  file_path_2 = "/content/ZoomInterpolation/codes/train.py"
  fh_2, abs_path_2 = mkstemp()
  with fdopen(fh_2,'w') as new_file:
    with open(file_path_2) as old_file:
      for counter, line in enumerate(old_file):
        if counter ==190:
          destination = f'                    destination = "{backup_location}/LunaTokis_scratch_b16p32f5b40n7l1_600k_Vimeo"\n'
          new_file.write(destination)
        else:
          new_file.write(line)
  copymode(file_path_2, abs_path_2)
  #Remove original file
  remove(file_path_2)
  #Move new file
  move(abs_path_2, file_path_2) 
  
  
  
  ############################################################
  
import shutil
import os
from tqdm import tqdm
import sys
sys.path.insert(0,'/content/ZoomInterpolation/load_functions')

  
def run_split_sequence(mode, scale_factor, save_HR, save_LR, test_or_train, outPath_test):
    if mode == "HR":
      sequences_path = os.path.join(save_HR, f"x{scale_factor}")
      train_guide = os.path.join(save_HR, "sep_trainlist.txt")
      test_guide = os.path.join(save_HR, "sep_testlist.txt")
      if test_or_train == "test":
          outPath_test = os.path.join(save_HR, f"test_{scale_factor}")
          split_test_train_sequences_data(sequences_path, outPath_test, test_guide)
      if test_or_train == "train":
          outPath_train = os.path.join(save_HR, f"train_{scale_factor}")
          split_test_train_sequences_data(sequences_path, outPath_train, train_guide)

    if mode == "LR":
      sequences_path = os.path.join(save_LR, f"x{scale_factor}")
      train_guide = os.path.join(save_LR, "sep_trainlist.txt")
      test_guide = os.path.join(save_LR, "sep_testlist.txt")
      if test_or_train == "test":
          outPath_test = os.path.join(save_LR, f"test_{scale_factor}")
          split_test_train_sequences_data(sequences_path, outPath_test, test_guide)
      if test_or_train == "train":
          outPath_train = os.path.join(save_LR, f"train_{scale_factor}")
          split_test_train_sequences_data(sequences_path, outPath_train, train_guide)
    return sequences_path, train_guide, test_guide, outPath_test

  
  
def prepare_lmbd(save_HR, save_LR, HR_input_dim, scale_factor, batch):
  
  LR_input_dim = HR_input_dim/scale_factor

  test_or_train = "train"
  mode = "HR"
  save_to_lmbd(save_HR, test_or_train, HR_input_dim, HR_input_dim, batch, mode, scale_factor)

  mode = "LR"
  save_to_lmbd(save_LR, test_or_train, LR_input_dim, LR_input_dim, batch, mode, scale_factor)
  
  test_or_train = "test"
  mode = "HR"
  save_to_lmbd(save_HR, test_or_train, HR_input_dim, HR_input_dim, batch, mode, scale_factor)
  
  mode = "LR"
  save_to_lmbd(save_LR, test_or_train, LR_input_dim, LR_input_dim, batch, mode, scale_factor)

  train_LMBD_HR = save_HR + "/vimeo7_train_x{}_HR.lmdb".format(scale_factor)
  train_LMBD_LR = save_LR + "/vimeo7_train_x{}_LR.lmdb".format(scale_factor)

  test_LMBD_HR = save_HR + "/vimeo7_test_x{}_HR.lmdb".format(scale_factor)
  test_LMBD_LR = save_LR + "/vimeo7_test_x{}_LR.lmdb".format(scale_factor)
  return train_LMBD_HR, train_LMBD_LR, test_LMBD_HR, test_LMBD_LR, LR_input_dim 

