from glob import glob
from SEGMENTATION.utils import count_images, open_masks, rm_alpha
import torch
import os
from PIL import Image 
import numpy as np
from torch.utils.data import Dataset
import slideio

class MonusacDataset(Dataset):
    '''MoNuSAC Dataset.'''

    def __init__(self, img_dir, masks_dir, transform=None):
        '''
        Arguments:
            img_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''

        self.img_dir = img_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return count_images(self.img_dir)

    def __getitem__(self, idx):
      
      #image extraction
      images = [y for x in os.walk(self.img_dir) for y in glob(os.path.join(x[0], '*.xml'))]

      img_path = images[idx]
      img_path = os.path.normpath(img_path)
      path_as_list = img_path.split('/')
      patient_code = path_as_list[-2]
      img_name = path_as_list[-1].split('.')[0]
      if (os.path.isfile(os.path.splitext(images[idx])[0] + '.tif')):
        image = Image.open(images[idx])
        image = np.asarray(image)
        image = rm_alpha(image)
      else:
        slide = slideio.open_slide(filename,"SVS")
        scene = slide.get_scene(0)
        image = scene.read_block()

      #mask extraction
      img_masks = []
      masks_folder_path = os.path.join(self.masks_dir, patient_code, img_name)

      if 'train' in self.masks_dir: 
          ep, lym, macro, neutr = open_masks(masks_folder_path, image.shape)
          sample = {'name': img_name, 'image': image, 'mask_ep': ep, 'mask_lym': lym, 'mask_macro': macro, 'mask_neutr': neutr}

      if 'test' in self.masks_dir:
          amb, ep, lym, macro, neutr = open_masks(masks_folder_path, image.shape)
          sample = {'name': img_name, 'image': image, 'mask_amb': amb, 'mask_ep': ep, 'mask_lym': lym, 'mask_macro': macro, 'mask_neutr': neutr}


      if self.transform:
        sample = self.transform(sample)

      return sample