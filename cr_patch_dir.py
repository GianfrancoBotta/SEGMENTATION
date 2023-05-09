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

      parser = MonusacDataset
      xtractor = PatchExtractor(win_size, step_size)

      out_dir = '%s/%s/%s/%dx%d_%dx%d/' % (
          save_root,
          dataset_name,
          out_dir_name,
          win_size[0],
          win_size[1],
          step_size[0],
          step_size[1],
      )

      print(out_dir)
      rm_n_mkdir(out_dir)
      print('ciao')
      pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
      pbarx = tqdm.tqdm(
          total=len(dataset_mon), bar_format=pbar_format, ascii=True, position=0
      )
