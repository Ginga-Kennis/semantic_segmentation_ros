import torch

from semantic_segmentation_ros.networks import load_model

class SemanticSegmentation:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(model_path)
        self.model = load_model(model_path, self.device)

    def predict(self, img):
        with torch.no_grad():
            y_pred = self.model(img)
        
        return y_pred.cpu().squeeze().numpy()