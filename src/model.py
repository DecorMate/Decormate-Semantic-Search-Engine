import sys
import os 
from PIL import Image
import torch
sys.path.append("ml-mobileclip")

from mobileclip import create_model_and_transforms, get_tokenizer
from dotenv import load_dotenv

load_dotenv()

class ModelCLIP:
    def __init__(self, model_name='mobileclip_s1', checkpoint=None, device='cpu'):
        self.model_name = model_name
        
        # Try multiple possible model paths for Railway deployment
        if checkpoint:
            self.checkpoint = checkpoint
        else:
            possible_paths = [
                os.environ.get('MODEL_PATH'),  # Railway environment variable
                '/app/models/mobileclip_s1.pt',  # Railway path
                os.path.join(os.getcwd(), 'models', 'mobileclip_s1.pt'),  # Local models folder
                os.environ.get('CHECKPOINT'),  # Legacy environment variable
            ]
            
            self.checkpoint = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    self.checkpoint = path
                    print(f"✅ Using model from: {path}")
                    break
            
            if not self.checkpoint:
                print("⚠️ No model checkpoint found, will use model without pretrained weights")
        
        self.device = device

    def load_mobileclip_model(self):
            """
            Load MobileCLIP model and preprocessing transforms
            Optimized for low memory environments like Railway
            """
            # Force CPU and low memory mode
            import torch
            torch.set_num_threads(1)  # Reduce CPU threads
            
            model, _, preprocess = create_model_and_transforms(
                model_name=self.model_name,
                pretrained= self.checkpoint, 
                device= 'cpu'  # Force CPU to avoid GPU memory issues
            )
            
            # Set to eval mode and optimize for inference
            model.eval()
            
            # Enable memory optimization
            for param in model.parameters():
                param.requires_grad = False
            
            tokenizer = get_tokenizer(self.model_name)
            
            return model, preprocess, tokenizer

    def encode_image(self,image_path, model ,preprocess):
        image = Image.open(image_path).convert('RGB')
        img = preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
             image_feat = model.encode_image(img)
             image_feat = image_feat / image_feat.norm(dim =-1, keepdim=True)

        return image_feat.squeeze().cpu().numpy()
    
    def encode_text(self, text, model, tokenizer):
         
         with torch.no_grad():
              tokens = tokenizer(text).to(self.device)
              text_feat = model.encode_text(tokens)
              text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
         
         return  text_feat.squeeze().cpu().numpy()


# if __name__ == '__main__':
#     image_path = 'src/astro.png'
#     text = "A photo of an astronaut riding a horse on mars."
#     # Set device

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # print(f"Using device: {device}")
    
#     clip = ModelCLIP(device=device)
    
#     try:
#         model, preprocess, tokenizer = clip.load_mobileclip_model()
#         print("Model loaded successfully!")
        
#         if os.path.exists(image_path):
#             encoded_image = clip.encode_image(image_path, model, preprocess)
#             print(f"Encoded image shape: {encoded_image.shape}")
#         if text:
#             encoded_text = clip.encode_text(text, model)
#             print(f"Encoded text shape: {encoded_text.shape}")
#             print(encoded_text)
#         else:
#             print(f"Warning: Image file '{image_path}' not found")
            
#     except Exception as e:
#         print(f"Error: {e}")
#         print("Make sure you have installed the required packages and downloaded the model weights")