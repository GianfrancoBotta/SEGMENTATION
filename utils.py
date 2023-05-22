import os
import numpy as np
import skimage.io as skio


def count_images(folder):
  '''Counts the images in all the subfolders of folder or in folder if it does not have subfolders'''
  
  len=0
  os.chdir(folder)
  subfolders=[f.path for f in os.scandir(folder) if f.is_dir()]
  if subfolders == []:
    for file_name in os.listdir():
      if file_name.endswith('.xml'):
        len+=1
  else:
    for subfolder in os.listdir():
      os.chdir(subfolder)
      for file_name in os.listdir():
        if file_name.endswith('.xml'):
          len+=1
      parent_path = os.path.abspath('..')     # parent folder path
      os.chdir(parent_path)       # move to parent

  return len

def open_masks(masks_folder_path, image_shape):
  '''Finds all the masks regarding epithelial cells, lymphocytes, macrophages and neutrophils in MoNuSAC_masks'''

  mask_shape = list(image_shape)
  mask_shape[-1] = 1
  mask_shape = tuple(mask_shape)

  try:
    epithelial = skio.imread(os.path.join(masks_folder_path, 'Epithelial', os.listdir(masks_folder_path+'/Epithelial')[0]), plugin="tifffile")
    if(len(epithelial.shape) == 2):
      epithelial = np.expand_dims(epithelial, axis=-1)
  except:
    epithelial = np.zeros(mask_shape)

  try:
    lymphocyte = skio.imread(os.path.join(masks_folder_path, 'Lymphocyte', os.listdir(masks_folder_path+'/Lymphocyte')[0]), plugin="tifffile")
    if(len(lymphocyte.shape) == 2):
      lymphocyte = np.expand_dims(lymphocyte, axis=-1)
  except:
    lymphocyte = np.zeros(mask_shape)

  try:
    macrophage = skio.imread(os.path.join(masks_folder_path, 'Macrophage', os.listdir(masks_folder_path+'/Macrophage')[0]), plugin="tifffile")
    if(len(macrophage.shape) == 2):
      macrophage = np.expand_dims(macrophage, axis=-1)
  except:
    macrophage = np.zeros(mask_shape)

  try:
    neutrophil = skio.imread(os.path.join(masks_folder_path, 'Neutrophil', os.listdir(masks_folder_path+'/Neutrophil')[0]), plugin="tifffile")
    if(len(neutrophil.shape) == 2):
      neutrophil = np.expand_dims(neutrophil, axis=-1)
  except:
    neutrophil = np.zeros(mask_shape)

  if 'test' in masks_folder_path:
    try:
      ambiguous = skio.imread(os.path.join(masks_folder_path, 'Ambiguous', os.listdir(masks_folder_path+'/Ambiguous')[0]), plugin="tifffile")
      if(len(ambiguous.shape) == 2):
        ambiguous = np.expand_dims(ambiguous, axis=-1)
    except:
      ambiguous = np.zeros(mask_shape)
  
  os.chdir('/content')
    
  return ambiguous, epithelial, lymphocyte, macrophage, neutrophil

def rm_alpha(image):
  '''Removes alpha channel from images'''

  shape = image.shape
  if(shape[2] > 3):
    image = image[:, :, :3-shape[2]]
  return image