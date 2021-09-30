import os
import sys
import argparse
import json
import shutil
import tarfile

import torch


# Args
parser = argparse.ArgumentParser()
parser.add_argument('-ap', '--all_packages', type=str, choices=['False', 'True'], default='False', help='......')
parser.add_argument('-bt', '--build_type', type=str, choices=['install', 'develop', 'bdist_wheel'], default='install')
parser.add_argument('-cc', '--compute_compatibility', type=str, default='30,35,37,50,52,53,60,61,62,70,72,75')
parser.add_argument('-o', '--output', type=str, default='default', help='tar')
args = parser.parse_args()

dain_root = os.getcwd()

# Check PyTorch Version
python_executable = sys.executable
print(f'Building CUDAExtension for PyTorch in {python_executable}')
torch_version = torch.__version__
torch_version_split = torch_version.split('.')
prefix = 'You need torch>=1.0.0, <=1.4.0, you have torch=='
if torch_version_split[0] == '0':
    raise RuntimeError(prefix + torch_version + ' < 1.0.0')
elif int(torch_version_split[0]) > 1 or int(torch_version_split[1]) > 4:
    raise RuntimeError(prefix + torch_version + ' > 1.4.0')

if args.build_type == 'bdist_wheel':
    # Check output dir
    if args.output == 'default':
        tar = tarfile.open(f'{dain_root}/wheels.tar', 'w')
    else:
        os.makedirs('/'.join(args.output.split('/')[:-1]), exist_ok=True)
        tar = tarfile.open(args.output if args.output[-4:] == '.tar' else (args.output + '.tar'), 'w')
    # Check if wheel is installed
    try:
        import wheel
    except ImportError:
        os.system(f'{python_executable} -m pip install wheel')

# Write compiler args
nvcc_args = []
for cc in args.compute_compatibility.split(','):
    nvcc_args.append('-gencode')
    nvcc_args.append(f'arch=compute_{cc},code=sm_{cc}')
nvcc_args.append('-w')
with open('compiler_args.json', 'w') as f:
    json.dump({'nvcc': nvcc_args, 'cxx': ['-std=c++11', '-w']}, f)
print(f'Compiling for compute compatilility {args.compute_compatibility}')

# Compile
os.chdir('my_package')
folders = [folder for folder in sorted(os.listdir('.')) if os.path.isdir(folder)]
if args.all_packages == 'False':
    folders = [folder for folder in folders if folder in ['DepthFlowProjection', 'FilterInterpolation', 'FlowProjection']]
for folder in folders:
    os.chdir(f"{'' if folder == folders[0] else '../'}{folder}")
    os.system(f'{python_executable} setup.py {args.build_type}')
os.chdir('../../PWCNet/correlation_package_pytorch1_0')
os.system(f'{python_executable} setup.py {args.build_type}')

os.chdir(dain_root)
terms_to_delete = lambda path: ['dist', 'build', [fil for fil in os.listdir(path) if fil[-9:] == '.egg-info'][0]]

# Tar if set and clean temporary files
for folder in folders:
    if args.build_type == 'bdist_wheel':
        whl = os.listdir(f'my_package/{folder}/dist')[0]
        tar.add(f'my_package/{folder}/dist/' + whl, whl)
    for file_to_delete in terms_to_delete('my_package/' + folder):
        shutil.rmtree(f'my_package/{folder}/{file_to_delete}')
if args.build_type == 'bdist_wheel':
    whl = os.listdir(f'PWCNet/correlation_package_pytorch1_0/dist')[0]
    tar.add('PWCNet/correlation_package_pytorch1_0/dist/' + whl, whl)
for file_to_delete in terms_to_delete('PWCNet/correlation_package_pytorch1_0'):
    shutil.rmtree(f'PWCNet/correlation_package_pytorch1_0/{file_to_delete}')
os.remove('compiler_args.json')
if args.build_type == 'bdist_wheel':
    tar.close()
