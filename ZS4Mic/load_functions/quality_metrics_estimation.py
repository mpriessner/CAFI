from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import img_as_float32
import numpy as np


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):#dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def norm_minmse(gt, x, normalize_gt=True):
    """This function is adapted from Martin Weigert"""

    """
    normalizes and affinely scales an image pair such that the MSE is minimized  
     
    Parameters
    ----------
    gt: ndarray
        the ground truth image      
    x: ndarray
        the image that will be affinely scaled 
    normalize_gt: bool
        set to True of gt image should be normalized (default)
    Returns
    -------
    gt_scaled, x_scaled 
    """
    if normalize_gt:
        gt = normalize(gt, 0.1, 99.9, clip=False).astype(np.float32, copy = False)
    x = x.astype(np.float32, copy=False) - np.mean(x)
    #x = x - np.mean(x)
    gt = gt.astype(np.float32, copy=False) - np.mean(gt)
    #gt = gt - np.mean(gt)
    scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
    return gt, scale * x


#--------------------------------------------------------------
def get_full_file_paths(folder):
  import os
  list_files = []
  for root, dirs, files in os.walk(folder):
      for file in files:
          file_path = os.path.join(root, file)
          list_files.append(file_path)
  return list_files


def create_shift_image(img, y_shift, x_shift):
  """this function shifts a 2D image with a given shift to x and y dimension - used to find the shift introduced by the network (to correct it manually)"""
  
  y_dim = img.shape[-2]
  x_dim = img.shape[-1]

  if x_shift<0:
    line = img[:,-1:]
    for i in range(0, x_shift, -1):
      img = np.concatenate((img,line), axis = 1) #right
    img = img[:,abs(x_shift):] 

  if x_shift>0:
    line = img[:,:1]
    for i in range(0, x_shift):
        img = np.concatenate((line, img), axis = 1) #left
    img = img[:,:x_dim] 

  if y_shift<0:
    line = img[-1:,:]
    for i in range(0, y_shift, -1):
      img = np.concatenate((img,line), axis = 0) #bottom
    img = img[abs(y_shift):]
    
  if y_shift>0:
    line = img[:1,:]
    for i in range(0, y_shift):
      img = np.concatenate((line, img), axis = 0) #top
    img = img[:y_dim,:]  
  return img




