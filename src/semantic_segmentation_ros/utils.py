import numpy as np
import matplotlib.pyplot as plt

def vis_img_mask(img, mask, aug_img, aug_mask):
    # 画像とマスクのデータを適切な形式に変換
    img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
    aug_img_np = aug_img.numpy().transpose(1, 2, 0)  # CHW -> HWC

    # マスクの合成（すべてのクラスのマスクを重ね合わせる）
    mask_combined = np.zeros(mask.shape[1:], dtype=np.float32)  # height x width
    aug_mask_combined = np.zeros(mask.shape[1:], dtype=np.float32)

    for i in range(mask.shape[0]):  # 全クラスに対して
        mask_combined += (mask.numpy()[i] / mask.shape[0]) * (i+1)  # 各クラスにユニークな重みを付ける
        aug_mask_combined += (aug_mask.numpy()[i] / mask.shape[0]) * (i+1)

    # プロットの設定
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2x2のサブプロット

    # 元の画像
    axs[0, 0].imshow(img_np.astype(np.uint8))
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')  # 軸を非表示

    # 元のマスク（合成後）
    axs[0, 1].imshow(mask_combined, cmap='viridis')  # カラーマップを適用
    axs[0, 1].set_title("Original Mask")
    axs[0, 1].axis('off')  # 軸を非表示

    # オーグメンテーション後の画像
    axs[1, 0].imshow(aug_img_np.astype(np.uint8))
    axs[1, 0].set_title("Augmented Image")
    axs[1, 0].axis('off')  # 軸を非表示

    # オーグメンテーション後のマスク（合成後）
    axs[1, 1].imshow(aug_mask_combined, cmap='viridis')
    axs[1, 1].set_title("Augmented Mask")
    axs[1, 1].axis('off')  # 軸を非表示

    # プロットの表示
    plt.show()