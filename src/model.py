import sys
sys.path.append("ml-mobileclip")
import torch
import os
from PIL import Image

from mobileclip import create_model_and_transforms, get_tokenizer

# Example usage:
# Available model names: 'mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b'
def load_mobileclip_model(model_name='mobileclip_s1', device='cpu', use_pretrained=True):
    """
    Load MobileCLIP model and preprocessing transforms
    
    Args:
        model_name: One of ['mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b']
        device: Device to load model on ('cpu' or 'cuda')
        use_pretrained: Whether to load pretrained weights from HuggingFace cache
    
    Returns:
        model: The MobileCLIP model
        preprocess: Image preprocessing transforms
        tokenizer: Text tokenizer
    """
    pretrained_path = None
    
    if use_pretrained and model_name == 'mobileclip_s1':
        # Path to the downloaded pretrained weights
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--pcuenq--MobileCLIP-S1/snapshots/9e9005f93d5d8eb197aac25cf53af31364ccd489")
        pretrained_path = os.path.join(cache_dir, "mobileclip_s1.pt")
        
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Using model without pretrained weights. Download weights with:")
            print("huggingface-cli download pcuenq/MobileCLIP-S1")
            pretrained_path = None
    
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained_path,
        device=device
    )
    
    tokenizer = get_tokenizer(model_name)
    
    return model, preprocess, tokenizer

def encode_image(model, preprocess, image_path, device='cpu'):
    """
    Encode an image using MobileCLIP
    
    Args:
        model: MobileCLIP model
        preprocess: Preprocessing transforms
        image_path: Path to image file
        device: Device to run inference on
    
    Returns:
        Image embedding tensor
    """
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features

def encode_text(model, tokenizer, text, device='cpu'):
    """
    Encode text using MobileCLIP
    
    Args:
        model: MobileCLIP model
        tokenizer: Text tokenizer
        text: Text string or list of text strings
        device: Device to run inference on
    
    Returns:
        Text embedding tensor
    """
    if isinstance(text, str):
        text = [text]
    
    text_tokens = tokenizer(text).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features

# Example usage:
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model with pretrained weights
    model, preprocess, tokenizer = load_mobileclip_model('mobileclip_s1', device, use_pretrained=True)
    print("Model loaded successfully!")
    
    # Example text encoding
    text = "a photo of a cat"
    text_features = encode_text(model, tokenizer, text, device)
    print(f"Text feature shape: {text_features.shape}")
    
    # Example image encoding (uncomment and provide image path)
    # image_features = encode_image(model, preprocess, "path/to/your/image.jpg", device)
    # print(f"Image feature shape: {image_features.shape}")
    
    # Calculate similarity (uncomment when you have both image and text features)
    # similarity = torch.cosine_similarity(image_features, text_features)
    # print(f"Similarity: {similarity.item()}")

