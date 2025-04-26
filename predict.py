import torch
from PIL import Image
from model import SmileNet
from config import model_save_path
import sys
from torchvision import transforms

 
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def predict_smile(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = SmileNet().to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))  
    model.eval()

    try:
        
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).to(device)  

        
        with torch.no_grad():
            output = model(image)
            prediction = torch.round(output).item()

        return "Smiling 😄" if prediction == 1 else "Not Smiling 😐"

    except Exception as e:
        return f"Prediction failed: {e}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        image_path = sys.argv[1]
        result = predict_smile(image_path)
        print(result)
