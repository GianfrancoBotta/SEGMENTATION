
import cv2
import logging
import numpy as np
from scipy.ndimage import label

def fix_mirror_padding(ann):
    'Deal with duplicated instances due to mirroring in interpolation during shape augmentation (scale, rotation etc.).'
    ann_copy = ann.copy()
    current_max_id = np.amax(ann_copy)
    inst_list = list(np.unique(ann_copy))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann_copy == inst_id, np.uint8)
        remapped_ids = label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += int(current_max_id)  # convert to int32
        ann_copy[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann_copy)
    return ann_copy

def compute_hv_map(mask):
    """
    Preprocessing step for HoVer-Net architecture.
    Compute center of mass for each nucleus, then compute distance of each nuclear pixel to its corresponding center
    of mass.
    Nuclear pixel distances are normalized to (-1, 1). Background pixels are left as 0.
    Operates on a single mask.
    Can be used in Dataset object to make Dataloader compatible with HoVer-Net.

    Based on https://github.com/vqdang/hover_net/blob/195ed9b6cc67b12f908285492796fb5c6c15a000/src/loader/augs.py#L192

    Args:
        mask (np.ndarray): Mask indicating individual nuclei. Array of shape (H, W),
            where each pixel is in {0, ..., n} with 0 indicating background pixels and {1, ..., n} indicating
            n unique nuclei.

    Returns:
        np.ndarray: array of hv maps of shape (2, H, W). First channel corresponds to horizontal and second vertical.
    """
    assert (
        mask.ndim == 2
    ), f"Input mask has shape {mask.shape}. Expecting a mask with 2 dimensions (H, W)"

    out = np.zeros((2, mask.shape[0], mask.shape[1]))
    # each individual nucleus is indexed with a different number
    inst_list = list(np.unique(mask))

    try:
        inst_list.remove(0)  # 0 is background
    except ValueError:
        logging.warning(
            "No pixels with 0 label. This means that there are no background pixels. This may indicate a problem. Ignore this warning if this is expected/intended."
        )

    for inst_id in inst_list:
        # get the mask for the nucleus
        inst_map = mask == inst_id
        inst_map = inst_map.astype(np.uint8)
        contours, _ = cv2.findContours(
            inst_map, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
        )

        # get center of mass coords
        mom = cv2.moments(contours[0])
        com_x = mom["m10"] / (mom["m00"] + 1e-6)
        com_y = mom["m01"] / (mom["m00"] + 1e-6)
        inst_com = (int(com_y), int(com_x))

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        # add to output mask
        # this works assuming background is 0, and each pixel is assigned to only one nucleus.
        out[0, :, :] += inst_x
        out[1, :, :] += inst_y
    return out

def generate_hv_map(annotation):
  '''Generates the hv map using '''

  # find the shortest axis
  fixed_annotation=fix_mirror_padding(annotation)
  shortest_axis = np.argmin(fixed_annotation.shape)

  # find the index of the maximum value along the shortest axis
  overlapped_mask = np.max(fixed_annotation, axis=shortest_axis)
  overlapped_mask = compute_hv_map(overlapped_mask)
  return overlapped_mask