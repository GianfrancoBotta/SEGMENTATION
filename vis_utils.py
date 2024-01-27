import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

def masks_overlap(masks):
  '''Overlaps masks and creates a new total mask composed of four different masks, one for each cell type'''

  if(type(masks)==dict):
    mask_ep = np.copy(masks['mask_ep'])
    mask_lym = np.copy(masks['mask_lym'])
    mask_macro = np.copy(masks['mask_macro'])
    mask_neutr = np.copy(masks['mask_neutr'])

  else:
    masks_cp = np.copy(masks)
    mask_ep = masks_cp[:,:,-4]
    mask_ep = mask_ep[:,:,np.newaxis]
    mask_lym = masks_cp[:,:,-3]
    mask_lym = mask_lym[:,:,np.newaxis]
    mask_macro = masks_cp[:,:,-2]
    mask_macro = mask_macro[:,:,np.newaxis]
    mask_neutr = masks_cp[:,:,-1]
    mask_neutr = mask_neutr[:,:,np.newaxis]

  mask_ep_idx = np.nonzero(mask_ep)
  mask_lym_idx = np.nonzero(mask_lym)
  mask_macro_idx = np.nonzero(mask_macro)
  mask_neutr_idx = np.nonzero(mask_neutr)

  mask_ep[mask_ep_idx] = 1
  mask_lym[mask_lym_idx] = 2
  mask_macro[mask_macro_idx] = 3
  mask_neutr[mask_neutr_idx] = 4
  conc = np.concatenate((mask_ep, mask_lym, mask_macro, mask_neutr), axis=-1)
  total_mask = np.sum(conc, axis=-1)
  total_mask = total_mask[:,:,np.newaxis]
  total_mask[mask_macro_idx] = 3
  total_mask[mask_ep_idx] = 1
  total_mask[mask_neutr_idx] = 4
  total_mask[mask_lym_idx] = 2

  return total_mask

def mask_visualizer(mask):
  '''Visualizes total masks by coloring each cell type differently'''
  bounds = np.unique(mask)
  bounds = list(bounds)
  colors_dict = {0:'black', 1:'red', 2:'yellow', 3:'green', 4:'blue'}
  colors_list = [colors_dict[val] for val in bounds]

  cmap = colors.ListedColormap(colors_list)
  norm = colors.BoundaryNorm(bounds, cmap.N)
  total_mask = plt.imshow(mask, cmap=cmap)