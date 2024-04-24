import torch
from semantic_segmentation_ros.networks import load_model

class SemanticSegmentation:
    def __init__(self, model_name, encoder_name, encoder_weights, in_channels, classes, model_path, device):
        self.device = device
        self.model = load_model(model_name, encoder_name, encoder_weights, in_channels, classes, model_path, self.device)
        self.model.eval()

    def predict(self, img):
        """
        Inputs a RGB image with shape (channel, height, width)
        """
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(img)
            y_pred = y_pred.argmax(1)
        
        return y_pred.cpu().squeeze(0).numpy()
