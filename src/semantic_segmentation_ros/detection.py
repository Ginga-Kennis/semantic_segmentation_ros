import torch
from semantic_segmentation_ros.networks import load_model

class SemanticSegmentation:
    def __init__(self, model_name, encoder_name, encoder_weights, in_channels, classes, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_name, encoder_name, encoder_weights, in_channels, classes, model_path,self.device)
        self.model.eval()

    def predict(self, img):
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(img)
            y_pred = y_pred.argmax(1)
        
        return y_pred.cpu().squeeze(0).numpy()
