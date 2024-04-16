import numpy as np
import matplotlib.pyplot as plt

def vis_img_mask(img, mask, aug_img, aug_mask):
    # 画像データを適切な形式に変換（CHW -> HWC）
    img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
    aug_img_np = aug_img.numpy().transpose(1, 2, 0)  # CHW -> HWC

    # クラス毎の色を定義
    colors = np.array([
        [255, 0, 0],    # 赤
        [0, 0, 255],    # 青
        [255, 255, 0],  # 黄
        [0, 255, 0]     # 緑
    ], dtype=np.uint8)
    
    # マスクをRGB画像に変換
    def mask_to_color(mask, colors):
        mask_color = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)  # height x width x 3
        for i in range(mask.shape[0]):
            mask_color[mask.numpy()[i] > 0] = colors[i]
        return mask_color
    
    # マスクを色付きの画像に変換
    mask_combined = mask_to_color(mask, colors)
    aug_mask_combined = mask_to_color(aug_mask, colors)

    # プロットの設定
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2x2のサブプロット

    # 元の画像
    axs[0, 0].imshow(img_np.astype(np.uint8))
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    # 元のマスク（色付き）
    axs[0, 1].imshow(mask_combined)
    axs[0, 1].set_title("Original Mask")
    axs[0, 1].axis('off')

    # オーグメンテーション後の画像
    axs[1, 0].imshow(aug_img_np.astype(np.uint8))
    axs[1, 0].set_title("Augmented Image")
    axs[1, 0].axis('off')

    # オーグメンテーション後のマスク（色付き）
    axs[1, 1].imshow(aug_mask_combined)
    axs[1, 1].set_title("Augmented Mask")
    axs[1, 1].axis('off')

    # プロットの表示
    plt.show()
