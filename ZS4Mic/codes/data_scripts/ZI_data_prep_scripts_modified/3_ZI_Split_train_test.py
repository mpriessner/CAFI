import os, shutil
from tqdm import tqdm

######################## SELECT NECESSARY IN and OUTPUT FOLDERS ########################
### inPath folder is the one highlighting the magnification e.g. x4 or x2 within the HR or LR folder
### select a desired output path
### as guide select the sep_trainlist.txt or sep_testlist.txt located in that folder
### This process needs to be performed for both HR and LR folder 

inPath = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\HR\x4'
outPath = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\HR\train_x4'
guide = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\HR_LR_4x_ZI_BF_1024\HR\sep_trainlist.txt'

f = open(guide, "r")
lines = f.readlines()

if not os.path.isdir(outPath):
    os.mkdir(outPath)

for l in tqdm(lines):
    line = l.replace('\n','')
    this_folder = os.path.join(inPath, line)
    dest_folder = os.path.join(outPath, line)
    print(this_folder)
    shutil.copytree(this_folder, dest_folder)
print('Done')
