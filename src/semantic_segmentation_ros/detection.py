import numpy as np
import torch

from semantic_segmentation_ros.networks import load_model


class SemanticSegmentation:
    def __init__(self, model_name: str, encoder_name: str, encoder_weights: str, in_channels: int, classes: int, model_path: str, device: str):
        """
        Initializes the SemanticSegmentation class with the specified model and parameters.

        Input:
            model_name (str): Name of the segmentation model to be used.
            encoder_name (str): Name of the encoder for the segmentation model.
            encoder_weights (str): Pre-trained weights for the encoder.
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            classes (int): Number of classes for the segmentation task.
            model_path (str): Path to the trained model file.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = load_model(model_name, encoder_name, encoder_weights, in_channels, classes, model_path, self.device)
        self.model.eval()

    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Predict the segmentation mask for a given RGB image.

        Args:
            img (np.ndarray): A numpy array of the RGB image with shape (channel, height, width).

        Returns:
            np.ndarray: A numpy array containing the predicted segmentation mask.
        """
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_pred = self.model(img)
            y_pred = y_pred.argmax(1)

        return y_pred.cpu().squeeze(0).numpy()
