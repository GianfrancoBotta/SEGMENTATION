import torch

def convert_multiclass_mask_to_binary(mask):
    '''Input mask of shape (B, n_classes, H, W) is converted to a mask of shape (B, 1, H, W).
    The last channel is assumed to be background, so the binary mask is computed by taking its inverse.'''
    mask_of_nonzero = mask != 0
    binary = torch.where(mask_of_nonzero, torch.ones_like(mask), mask)
    max_values, _ = torch.max(binary, dim=1)
    output_tensor = max_values

    return output_tensor