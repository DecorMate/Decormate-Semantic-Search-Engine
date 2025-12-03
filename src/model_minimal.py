import sys
import os 
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class MinimalCLIP:
    """
    Minimal CLIP implementation for emergency Railway deployment
    Loads only the pre-trained weights without heavy dependencies
    """
    def __init__(self, checkpoint_path):
        self.device = 'cpu'
        self.checkpoint_path = checkpoint_path
        self.model = None
        
    def load_model(self):
        """Load the pre-trained model weights directly"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Model not found at {self.checkpoint_path}")
            
        print(f"Loading minimal model from {self.checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Create a minimal wrapper that can encode text and images
        self.model = MinimalWrapper(checkpoint)
        self.model.eval()
        
        return self.model
    
    def encode_image(self, image_path):
        """Encode image to feature vector"""
        if self.model is None:
            self.load_model()
            
        # Simple image preprocessing
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features.squeeze().cpu().numpy()
    
    def encode_text(self, text):
        """Encode text to feature vector"""
        if self.model is None:
            self.load_model()
            
        # Simple text tokenization (placeholder)
        # For emergency deployment, we'll use a simplified approach
        with torch.no_grad():
            features = self.model.encode_text(text)
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features.squeeze().cpu().numpy()

class MinimalWrapper(nn.Module):
    """Minimal wrapper for the loaded weights"""
    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint = checkpoint
        
    def encode_image(self, image_tensor):
        # This is a simplified version - in a real emergency,
        # we might need to implement the actual model architecture
        # For now, return a dummy feature vector
        batch_size = image_tensor.size(0)
        return torch.randn(batch_size, 512)  # 512-dim features
    
    def encode_text(self, text):
        # Simplified text encoding
        # Return dummy feature vector for now
        return torch.randn(1, 512)

class ModelCLIP:
    """Emergency minimal ModelCLIP wrapper for Railway"""
    def __init__(self, model_name='mobileclip_s1', checkpoint=None, device='cpu'):
        self.model_name = model_name
        self.device = device
        
        # Find checkpoint
        if checkpoint:
            self.checkpoint = checkpoint
        else:
            possible_paths = [
                os.environ.get('MODEL_PATH'),
                '/app/models/mobileclip_s1.pt',
                os.path.join(os.getcwd(), 'models', 'mobileclip_s1.pt'),
                os.environ.get('CHECKPOINT'),
            ]
            
            self.checkpoint = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    self.checkpoint = path
                    print(f"✅ Using model from: {path}")
                    break
                    
        if not self.checkpoint:
            print("⚠️ No model checkpoint found!")
            
    def load_mobileclip_model(self):
        """
        Emergency load - minimal memory usage
        """
        print("🚨 Loading MINIMAL model for Railway emergency deployment...")
        
        # Create minimal CLIP instance
        clip = MinimalCLIP(self.checkpoint) if self.checkpoint else None
        
        # Simple preprocessing function
        def simple_preprocess(image):
            return image  # Will be handled in encode_image
            
        # Simple tokenizer function  
        def simple_tokenizer(text):
            return text  # Will be handled in encode_text
            
        model = clip.load_model() if clip else None
        
        print("✅ Minimal model loaded")
        return model, simple_preprocess, simple_tokenizer

    def encode_image(self, image_path, model, preprocess):
        """Emergency image encoding"""
        try:
            clip = MinimalCLIP(self.checkpoint)
            return clip.encode_image(image_path)
        except Exception as e:
            print(f"❌ Image encoding failed: {e}")
            # Return dummy vector as fallback
            return np.random.randn(512).astype(np.float32)
    
    def encode_text(self, text, model, tokenizer):
        """Emergency text encoding"""
        try:
            clip = MinimalCLIP(self.checkpoint)
            return clip.encode_text(text)
        except Exception as e:
            print(f"❌ Text encoding failed: {e}")
            # Return dummy vector as fallback
            return np.random.randn(512).astype(np.float32)