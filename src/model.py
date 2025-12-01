import sys
sys.path.append("ml-mobileclip")

from mobileclip import create_model_and_transforms, get_tokenizer


def load_mobileclip_model(model_name='mobileclip_s1', device='cpu'):
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
        model_name=model_name,
        pretrained=None, 
        device=device
    )
    
    tokenizer = get_tokenizer(model_name)
    
    return model, preprocess, tokenizer


