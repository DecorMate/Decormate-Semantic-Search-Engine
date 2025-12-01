import sys
import os 
from PIL import Image
import torch
sys.path.append("ml-mobileclip")

from mobileclip import create_model_and_transforms, get_tokenizer
from dotenv import load_dotenv

load_dotenv()

class ModelCLIP:
    def __init__(self, model_name= 'mobileclip_s1', checkpoint=os.environ.get('CHECKPOINT'), device= 'cpu'):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.device = device

    def load_mobileclip_model(self):
            """
            Load MobileCLIP model and preprocessing transforms
            
            Args:
                model_name: mobileclip_s1
                device: Device to load model on ('cpu' or 'cuda')
            
            Returns:
                model: The MobileCLIP model
                preprocess: Image preprocessing transforms
                tokenizer: Text tokenizer
            """
            model, _, preprocess = create_model_and_transforms(
                model_name=self.model_name,
                pretrained= self.checkpoint, 
                device= self.device
            )
            
            tokenizer = get_tokenizer(self.model_name)
            
            return model, preprocess, tokenizer

    def encode_image(self,image_path, model ,preprocess):
        image = Image.open(image_path).convert('RGB')
        img = preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
             image_feat = model.encode_image(img)
             image_feat = image_feat / image_feat.norm(dim =-1, keepdim=True)

        return image_feat.squeeze().cpu().numpy()
    



if __name__ == '__main__':
    image_path = 'src/astro.png'
    
    # Set device
    print(torch.cuda.is_available())
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    
    clip = ModelCLIP(device='cuda')
    
    try:
        model, preprocess, tokenizer = clip.load_mobileclip_model()
        print("Model loaded successfully!")
        
        if os.path.exists(image_path):
            encoded_image = clip.encode_image(image_path, model, preprocess)
            print(f"Encoded image shape: {encoded_image.shape}")
        else:
            print(f"Warning: Image file '{image_path}' not found")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required packages and downloaded the model weights")