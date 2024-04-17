import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def vis_img_mask(img, mask):
    # 画像データを適切な形式に変換（CHW -> HWC）
    img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC

    # マスクのチャンネル数に応じて色相を自動生成
    num_classes = mask.shape[0]
    hsv_colors = [(i / num_classes, 1, 1) for i in range(num_classes)]  # 色相(Hue), 彩度(Saturation), 明度(Value)
    rgb_colors = list(map(lambda c: mcolors.hsv_to_rgb(c), hsv_colors))  # HSVからRGBに変換
    colors = (np.array(rgb_colors) * 255).astype(np.uint8)  # 0-255スケールへ変換

    # マスクをRGB画像に変換する関数
    def mask_to_color(mask, colors):
        mask_color = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)  # height x width x 3
        for i in range(mask.shape[0]):
            mask_color[mask.numpy()[i] > 0] = colors[i]
        return mask_color
    
    # マスクを色付きの画像に変換
    mask_combined = mask_to_color(mask, colors)

    # プロットの設定
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1x2のサブプロット

    # 元の画像の表示
    axs[0].imshow(img_np.astype(np.uint8))
    axs[0].set_title("Image")
    axs[0].axis('off')

    # マスク（色付き）の表示
    axs[1].imshow(mask_combined)
    axs[1].set_title("Mask")
    axs[1].axis('off')

    # プロットの表示
    plt.show()


