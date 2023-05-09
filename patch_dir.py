def patch_dir(images_path: str, masks_path: str, dataset_name: str, out_dir_name: str, win_size:list, step_size: list, extract_type: str, type_classification: bool = True):
  '''Creates train and validation patch directory and set the output directory for the patches
  to the given name.
  
  Args:
    images_path: a string indicating the path of the training folder
    masks_path: a string indicating the path of the masks folder
    dataset_name: a string indicating the name of the user's dataset ('Kumar', 'CPM17' or 'CoNSeP')
    out_dir_name: a string indicating the name given to the output directory ('train' or 'valid')
    win_ size: a list containing the x and y size of the patch window
    step_size: a list containing the x and y size of the step
    extract_type: a string containing the type of patch extraction ('valid' or 'mirror')
    type_classification: boolean determining whether to extract type map (only applicable to datasets with class labels)
  '''

  if __name__ == '__main__':

      save_root = '/content/patches'
      # a dictionary to specify where the dataset path should be
      dataset_info = {
          'train': {
              'img': ('.tif', images_path),
              'ann': ('.tif', masks_path),
          },
          'valid': {
              'img': ('.tif', images_path),
              'ann': ('.tif', masks_path),
          },
      }

      # patterning = lambda x: re.sub('([\[\]])', '[\\1]', x) # anonymous function that replaces substring ([\[\]]) with substring [\\1] in string x
      parser = MonusacDataset
      xtractor = PatchExtractor(win_size, step_size)
      for split_name, split_desc in dataset_info.items():
          img_ext, img_dir = split_desc['img']
          ann_ext, ann_dir = split_desc['ann']

          out_dir = '%s/%s/%s/%dx%d_%dx%d/' % (
              save_root,
              dataset_name,
              out_dir_name,
              win_size[0],
              win_size[1],
              step_size[0],
              step_size[1],
          )
          # file_list = glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
          # file_list.sort()  # ensure same ordering across platform

          rm_n_mkdir(out_dir)

          pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
          pbarx = tqdm.tqdm(
              total=len(dataset_mon), bar_format=pbar_format, ascii=True, position=0
          )