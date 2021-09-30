'''create lmdb files for Vimeo90K-7 frames training dataset (multiprocessing)
Will read all the images to the memory
'''
import os,sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2
from tqdm import tqdm

#### Load the necessary functions
def reading_image_worker(path, key):
    '''worker for reading images'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)



######################## IMPORTANT ########################
##### CHANGE THE DIRECTORIES INT THIS FUNCTION ############
### img_folder needs to be the one created in in the script 3_ZI_Split_train_test named "train_x4 or x2"
### lmdb_save_path needs to end with "*.lmdb"
### txt_file needs to be selected as well (same as in script 3) either the sep_trainlist.txt or sep_testlist.txt
### H_dst, W_dst  needs to be selected as well - this are the input dimensions of the images used for training
### This needs to be done for HR and LR folder separately

def vimeo7_2(mode):
    '''create lmdb for the Vimeo90K-7 frames dataset, each image with fixed size
    GT: [3, 256, 448]
        Only need the 4th frame currently, e.g., 00001_0001_4
    LR: [3, 64, 112]
        With 1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001
    '''
        
    #### configurations
    # mode = 'LR'  # GT | LR
    batch = 3000 # TODO: depending on your mem size
    
    if mode == 'GT':
        img_folder = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\HR\train_x4'
        lmdb_save_path = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\HR\vimeo7_train_GT_2.lmdb'
        txt_file = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\HR\sep_trainlist.txt'
        H_dst, W_dst = 1024, 1024
    elif mode == 'LR':
        img_folder = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\LR\train_x4'
        lmdb_save_path = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\LR\vimeo7_train_GT_2.lmdb'
        txt_file = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\LR\sep_trainlist.txt'
        H_dst, W_dst = 256, 256
    n_thread = 40
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
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        file_l = glob.glob(osp.join(img_folder, folder, sub_folder) + '/*')
        all_img_list.extend(file_l)
        for j in range(7):
            keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'GT': 
        all_img_list = [v for v in all_img_list if v.endswith('.png')]
        keys = [v for v in keys]
    print('Calculating the total size of images...')
    data_size = sum(os.stat(v).st_size for v in all_img_list)

    #### read all images to memory (multiprocessing)
    print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
    
    #### create lmdb environment
    env = lmdb.open(lmdb_save_path, map_size=data_size * 3)  ### for big datasets if error occures change 3 to a higher number 
    txn = env.begin(write=True)  # txn is a Transaction object

    #### write data to lmdb
    # pbar = util.ProgressBar(len(all_img_list))

    i = 0
    for path, key in tqdm(zip(all_img_list, keys)):
        # pbar.update('Write {}'.format(key))
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
    if mode == 'GT':
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
    


# This function can be used for testing the functionality of the created lmdb dataset
def test_lmdb(dataroot, dataset='vimeo7'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(osp.join(dataroot, 'Vimeo7_train_keys.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    # read one image
    if dataset == 'vimeo7':
        key = '00033_0286_1'
    else:
        raise NameError('Please check the filename format.')
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
    img = img_flat.reshape(H, W, C)
    cv2.imwrite(r'G:\HR_LR\all\test.png', img)


# if __name__ == "__main__":
vimeo7_2("GT")
vimeo7_2("LR")
