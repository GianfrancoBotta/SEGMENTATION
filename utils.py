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

def open_masks(masks_folder_path, image_shape, test=False):
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

  if not(test):
    #os.chdir('/content')
    return epithelial, lymphocyte, macrophage, neutrophil

  if test:
    try:
      ambiguous = skio.imread(os.path.join(masks_folder_path, 'Ambiguous', os.listdir(masks_folder_path+'/Ambiguous')[0]), plugin="tifffile")
      if(len(ambiguous.shape) == 2):
        ambiguous = np.expand_dims(ambiguous, axis=-1)
    except:
      ambiguous = np.zeros(mask_shape)

    #os.chdir('/content')
    return ambiguous, epithelial, lymphocyte, macrophage, neutrophil

def rm_alpha(image):
  '''Removes alpha channel from images'''

  shape = image.shape
  if(shape[2] > 3):
    image = image[:, :, :3-shape[2]]
  return image

def analyse_dataset(patches_dataset):
    layer1_count = 0
    layer2_count = 0
    layer3_count = 0
    layer4_count = 0
    # Iterate over the dataset
    for _, mask, _, _ in patches_dataset:
        # Convert the mask array to a numpy array if needed
        mask = np.array(mask[1:]) # doesn't take into account the bg
        
         #Count unique elements in each layer
        unique_layer1 = np.unique(mask[0])[1:]
        unique_layer2 = np.unique(mask[1])[1:]
        unique_layer3 = np.unique(mask[2])[1:]
        unique_layer4 = np.unique(mask[3])[1:]
        # Update counts for each layer
        layer1_count += len(unique_layer1)
        layer2_count += len(unique_layer2)
        layer3_count += len(unique_layer3)
        layer4_count += len(unique_layer4)

    # Calculate the total number of unique elements
    total_unique_elements = layer1_count + layer2_count + layer3_count + layer4_count

    # Calculate relative proportions
    layer1_proportion = layer1_count / total_unique_elements
    layer2_proportion = layer2_count / total_unique_elements
    layer3_proportion = layer3_count / total_unique_elements
    layer4_proportion = layer4_count / total_unique_elements

    # Print the results
    print("Epithelial - Count:", layer1_count)
    print("Epithelial - Proportion: {:.2%}".format(layer1_proportion))
    print("Lymphocyte - Count:", layer2_count)
    print("Lymphocyte - Proportion: {:.2%}".format(layer2_proportion))
    print("Macrophage - Count:", layer3_count)
    print("Macrophage - Proportion: {:.2%}".format(layer3_proportion))
    print("Neutrophil - Count:", layer4_count)
    print("Neutrophil - Proportion: {:.2%}".format(layer4_proportion))
