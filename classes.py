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

    def __init__(self, img_dir, masks_dir, transform=None, test=False, blue_chan=False):
        '''
        Arguments:
            img_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            test (boolean): Indicating if the dataset is for train or test purpose
            blue_chan (boolean): Indicating if the images have three channels or only the blue one
        '''

        self.img_dir = img_dir
        self.masks_dir = masks_dir
        self.transform = transform
	      self.test = test
        self.blue_chan = blue_chan

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
      tif_file = os.path.splitext(images[idx])[0] + '.tif'
      svs_file = os.path.splitext(images[idx])[0]+'.svs'
      if (os.path.isfile(tif_file)):
        image = Image.open(tif_file)
        image = np.asarray(image)
        image = rm_alpha(image)
      else:
        slide = slideio.open_slide(svs_file,"SVS")
        scene = slide.get_scene(0)
        image = scene.read_block()
      

      #mask extraction
      img_masks = []
      masks_folder_path = os.path.join(self.masks_dir, patient_code, img_name)

      if not(self.test): 
          ep, lym, macro, neutr = open_masks(masks_folder_path, image.shape)
          sample = {'name': img_name, 'image': image, 'mask_ep': ep, 'mask_lym': lym, 'mask_macro': macro, 'mask_neutr': neutr}

      if self.test:
          amb, ep, lym, macro, neutr = open_masks(masks_folder_path, image.shape)
          sample = {'name': img_name, 'image': image, 'mask_amb': amb, 'mask_ep': ep, 'mask_lym': lym, 'mask_macro': macro, 'mask_neutr': neutr}

      if self.blue_chan:
         sample['image'] = np.expand_dims(image[:,:,-1], axis=-1)

      if self.transform:
        sample = self.transform(sample)

      return sample