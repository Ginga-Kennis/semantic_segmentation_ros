import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def vis_img_mask(img: torch.tensor, mask: torch.tensor) -> None:
    img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC

    num_classes = mask.shape[0]
    hsv_colors = [(i / num_classes, 1, 1) for i in range(num_classes)]
    rgb_colors = list(map(lambda c: mcolors.hsv_to_rgb(c), hsv_colors))
    colors = (np.array(rgb_colors) * 255).astype(np.uint8)

    def mask_to_color(mask, colors):
        mask_color = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
        for i in range(mask.shape[0]):
            mask_color[mask.numpy()[i] > 0] = colors[i]
        return mask_color

    mask_combined = mask_to_color(mask, colors)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(img_np.astype(np.uint8))
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(mask_combined)
    axs[1].set_title("Mask")
    axs[1].axis("off")

    plt.show()
