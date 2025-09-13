import numpy as np


def create_binary_mask(masks, image_shape):
    h, w = image_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    if masks is not None:
        for mask in masks:
            mask_data = mask[0].astype(np.uint8)
            binary_mask = np.logical_or(binary_mask, mask_data).astype(np.uint8)

    binary_mask = binary_mask * 255
    return binary_mask