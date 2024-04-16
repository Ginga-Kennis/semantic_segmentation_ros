import matplotlib.pyplot as plt
from torchvision.transforms import v2
from PIL import Image
import json
import numpy as np
import cv2

def show_two_images(img, mask):
    # サブプロットの設定
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 1つ目の画像の表示
    axes[0].imshow(img)
    axes[0].axis('off')  # 軸を非表示にする
    axes[0].set_title('Image')

    # マスクデータを視覚化可能な形式に変換
    if mask.ndim == 3 and mask.shape[0] > 1:  # マルチクラスマスクの場合
        mask_display = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)  # RGB画像を作成
        for i in range(mask.shape[0]-1):  # 最後のチャネルは背景として無視
            mask_display += (np.stack([mask[i]]*3, axis=-1) * (255 / mask.shape[0])).astype(np.uint8)

    # 2つ目の画像の表示
    axes[1].imshow(mask_display)
    axes[1].axis('off')  # 軸を非表示にする
    axes[1].set_title('Mask')

    # グラフを表示
    plt.show()

def get_image(img_path):
    img = cv2.imread(img_path).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (2, 0, 1))

def get_mask(img, mask_path, labels):
    with open(mask_path) as handle:
        data = json.load(handle)
    shape_dicts = data["shapes"]

    channels = []

    # class in mask
    cls = [x['label'] for x in shape_dicts]
    
    # points of each class
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]

    # dictionary where key:label, value:points
    label2poly = dict(zip(cls, poly))

    # define background
    # _, height, width = img.shape
    background = np.zeros(shape=(255, 255), dtype=np.float32)

    for label in labels:
        
        blank = np.zeros(shape=(255, 255), dtype=np.float32)
        
        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], 1)
            cv2.fillPoly(background, [label2poly[label]], 255)
            
        channels.append(blank)
    _, thresh = cv2.threshold(background, 127, 1, cv2.THRESH_BINARY_INV)
    channels.append(thresh)

    Y = np.stack(channels, axis=0)
    return Y

img = Image.open("assets/data/img/6.jpg")
# img = get_image("assets/data/img/6.jpg")
mask = get_mask(img, "assets/data/ann/6.json", ["circle", "square", "star", "triangle"])

tf = v2.Compose([
                v2.RandomAffine(
                    degrees=(-40, 40), translate=(0.30, 0.15), scale=(0.8, 1.2)
                ),
                v2.RandomHorizontalFlip(0.5),
                ])
tf_img, tf_mask = tf(img, mask)

show_two_images(img, mask)
show_two_images(tf_img, tf_mask)