import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from semantic_segmentation_ros.dataset import SegDataset  # Your dataset class
from semantic_segmentation_ros.networks import load_network  # Your function to load the network

def visualize_segmentation(true_mask, predicted_mask):
    """
    実際のマスクと予測マスクを並べて表示します。
    """
    plt.figure(figsize=(10, 4))
    
    if true_mask.ndim == 3:
        # 最も高い確率を持つクラスのインデックスを取得
        true_mask = np.argmax(true_mask, axis=0)
    
    if predicted_mask.ndim == 3:
        predicted_mask = np.argmax(predicted_mask, axis=0)

    plt.subplot(1, 2, 1)
    plt.imshow(true_mask, cmap='gray')
    plt.title('True Mask')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

def evaluate_and_visualize(model, data_loader, device, num_images=5):
    """
    Evaluates the model and visualizes the results.
    """
    model.eval()
    images_to_show = min(num_images, len(data_loader.dataset))

    with torch.no_grad():
        for i, (images, true_masks) in enumerate(data_loader):
            # if i >= images_to_show: break
            
            images = images.to(device).float()
            true_masks = true_masks.to(device)

            outputs = model(images)
            predicted_masks = torch.argmax(outputs, dim=1)
            
            # Display only the first data in the batch
            true_mask_np = true_masks[0].cpu().numpy()
            predicted_mask_np = predicted_masks[0].cpu().numpy()
            
            visualize_segmentation(true_mask_np, predicted_mask_np)

def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize segmentation results')
    parser.add_argument('--model-path', type=Path, required=True, help='Path to the saved model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for evaluation')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset and dataloader
    dataset = SegDataset(args.data_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the model
    model = load_network(args.model_path, device)

    # Execute evaluation and visualization
    evaluate_and_visualize(model, data_loader, device)

if __name__ == "__main__":
    main()