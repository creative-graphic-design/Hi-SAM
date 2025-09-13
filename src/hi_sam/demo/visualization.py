import io

import matplotlib.pyplot as plt
import numpy as np


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = (
            color
            if color is not None
            else np.array([30 / 255, 144 / 255, 255 / 255, 0.5])
        )
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_masks(masks, filename, image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, mask in enumerate(masks):
        mask = mask[0].astype(np.uint8)
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


def create_masks_image(masks, image):
    """Create visualization image with masks overlaid, return as bytes buffer."""
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, mask in enumerate(masks):
        mask = mask[0].astype(np.uint8)
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis("off")

    # Save to bytes buffer instead of file
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    plt.close()

    return buffer