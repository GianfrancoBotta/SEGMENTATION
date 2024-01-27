from albumentations.pytorch.transforms import ToTensorV2
from glob import glob
from SEGMENTATION.utils import count_images, open_masks, rm_alpha
from SEGMENTATION.hv_masks_generator import generate_hv_map
from SEGMENTATION.bin_masks_generator import convert_multiclass_mask_to_binary
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

class PatchesDataset(Dataset):
    '''Dataset of processed patches.'''

    def __init__(self, patch_dir, geom_transform = None, color_transform = None, tensor_transform = True, test = False, blue_chan = False):
        '''
        Arguments:
            patch_dir (string): Directory with all the patches.
            geom_transform (callable, optional): Optional geometric transformations to be applied
                on a sample.
            color_transform (callable, optional): Optional color transformations to be applied
                on a sample.
            tensor_transform (bool): set this parameter to False if you don't want to convert the sample to torch.Tensor
            test (bool): if true, it removes the ambiguous masks (4th channel)
        '''

        self.patch_dir = patch_dir
        self.blue_chan = blue_chan
        self.geom_transform = geom_transform
        self.color_transform = color_transform
        self.tensor_transform = tensor_transform
        self.test = test

    def __len__(self):
        return len(os.listdir(self.patch_dir))

    def __getitem__(self, idx):
      if self.blue_chan:
        img_channels = 1
      else:
        img_channels = 3
      img_name = os.listdir(self.patch_dir)[idx]
      image_and_masks = np.load(self.patch_dir + '/' + img_name)
      image = image_and_masks[:,:, :img_channels].astype(np.float32)
      masks = image_and_masks[:,:,img_channels:].astype(np.float32)

      if self.geom_transform:
        img_and_masks = self.geom_transform(image = image, mask = masks)
        image = img_and_masks['image']
        masks = img_and_masks['mask']
      if self.color_transform:
        img = self.color_transform(image = image)
        image = img['image']

      # add horizontal-vertical maps
      hv_maps = generate_hv_map(masks)
      hv_maps = np.transpose(hv_maps, (1, 2, 0))

      if self.tensor_transform:
        to_tensor = ToTensorV2()
        img_and_masks_T = to_tensor(image = image, mask = masks)
        image = img_and_masks_T['image'] # the numpy HWC image is converted to pytorch CHW tensor
        masks = img_and_masks_T['mask'].permute(2, 0, 1)  # the numpy HWC masks are converted to pytorch HWC tensors
        hv_maps = to_tensor(image=hv_maps)['image']

      # add cell types
      types = 'Epithelial', 'Lymphocyte', 'Macrophage', 'Neutrophil'

      # add background masks
      masks = masks[np.newaxis, ...]
      # add binary masks
      background_array = torch.logical_not(convert_multiclass_mask_to_binary(masks))
      masks_bg = torch.cat((background_array, masks[0]), dim=0)

      sample = image, masks_bg, hv_maps, types

      return sample
