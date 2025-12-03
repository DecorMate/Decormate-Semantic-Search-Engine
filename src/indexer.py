import os
import uuid
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from model import ModelCLIP

load_dotenv()

class SimpleIndexer:
    def __init__(self):
        """Initialize Pinecone only - defer model loading to save memory"""
        # Setup Pinecone
        self.pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        self.index = self.pc.Index('decormate')
        
        # Model components - load on demand
        self.clip = None
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        print("✅ Pinecone ready - Model will load on first use")

    def _get_model(self):
        """Load model only when needed"""
        if self.model is None:
            print("🔄 Loading model...")
            self.clip = ModelCLIP(device='cpu')
            self.model, self.preprocess, self.tokenizer = self.clip.load_mobileclip_model()
            print("✅ Model loaded")
        return self.model, self.preprocess, self.tokenizer

    def add_image(self, image_path, description=None, custom_id=None):
        """Add an image to the database"""
        try:
            model, preprocess, _ = self._get_model()
            # Create embedding
            vector = self.clip.encode_image(image_path, model, preprocess)
            
            # Use custom ID or generate one
            item_id = custom_id or str(uuid.uuid4())
            metadata = {
                'type': 'image',
                'name': os.path.basename(image_path),
                'description': description or '',
                'path': image_path
            }
            
            # Save to database
            self.index.upsert([{
                "id": item_id,
                "values": vector.tolist(),
                "metadata": metadata
            }])
            
            print(f"✅ Added image: {item_id}")
            return item_id
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def add_text(self, text, category=None, custom_id=None):
        """Add text to the database"""
        try:
            model, _, tokenizer = self._get_model()
            # Create embedding
            vector = self.clip.encode_text(text, model, tokenizer)
            
            # Use custom ID or generate one
            item_id = custom_id or str(uuid.uuid4())
            metadata = {
                'type': 'text',
                'content': text,
                'category': category or ''
            }
            
            # Save to database
            self.index.upsert([{
                "id": item_id,
                "values": vector.tolist(),
                "metadata": metadata
            }])
            
            print(f"✅ Added text: {item_id}")
            return item_id
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def search(self, query, limit=5):
        """Search for similar items"""
        try:
            model, preprocess, tokenizer = self._get_model()
            
            # Check if query is an image file
            if os.path.exists(query) and query.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Search with image
                vector = self.clip.encode_image(query, model, preprocess)
                print(f"🔍 Searching with image: {os.path.basename(query)}")
            else:
                # Search with text
                vector = self.clip.encode_text(query, model, tokenizer)
                print(f"🔍 Searching for: {query}")
            
            # Find similar items
            results = self.index.query(
                vector=vector.tolist(),
                top_k=limit,
                include_metadata=True
            )
            
            # Show results
            print(f"Found {len(results.matches)} results:")
            for i, match in enumerate(results.matches, 1):
                score = match.score
                metadata = match.metadata
                if metadata.get('type') == 'image':
                    print(f"{i}. 📷 {metadata.get('name', 'Unknown')} (score: {score:.3f})")
                else:
                    content = metadata.get('content', metadata.get('name', 'Unknown'))
                    print(f"{i}. 📝 {content[:50]}... (score: {score:.3f})")
            
            return results.matches
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

# Simple usage examples
if __name__ == "__main__":
    # Create indexer
    indexer = SimpleIndexer()
    
    # image = 'src/images/furniture_1d37592b-0902-461e-b45d-daeb82e38d3e.jpg'
    image = 'src/images/furniture_5ec90457-1501-435b-b80f-fe9b4a1a9eae.jpg'

    # indexer.add_image(image)
    indexer.search(image)
    # Search with image
    # indexer.search("src/astro2.png")
    
    print("📚 Indexer is ready! Uncomment examples above to start using it.")