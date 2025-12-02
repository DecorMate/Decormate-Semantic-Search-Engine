import os
import uuid
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from model import ModelCLIP

load_dotenv()

class SimpleIndexer:
    def __init__(self):
        """Initialize with minimal setup"""
        # Setup Pinecone
        self.pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        self.index = self.pc.Index('decormate')
        
        # Setup model (CPU only for memory efficiency)
        self.clip = ModelCLIP(device='cpu')
        self.model, self.preprocess, self.tokenizer = self.clip.load_mobileclip_model()
        print("✅ Ready!")

    def add_image(self, image_path, description=None, custom_id=None):
        """Add image to database"""
        vector = self.clip.encode_image(image_path, self.model, self.preprocess)
        item_id = custom_id or str(uuid.uuid4())
        
        self.index.upsert(vectors=[(item_id, vector.tolist(), {
            'type': 'image',
            'description': description or ''
        })])
        
        return item_id

    def add_text(self, text, category=None, custom_id=None):
        """Add text to database"""
        vector = self.clip.encode_text(text, self.model, self.tokenizer)
        item_id = custom_id or str(uuid.uuid4())
        
        self.index.upsert(vectors=[(item_id, vector.tolist(), {
            'type': 'text',
            'category': category or ''
        })])
        
        return item_id

    def search(self, query, limit=5):
        """Search for similar content"""
        # Create query vector
        if os.path.exists(query):  # File path
            vector = self.clip.encode_image(query, self.model, self.preprocess)
        else:  # Text query
            vector = self.clip.encode_text(query, self.model, self.tokenizer)
        
        # Search
        results = self.index.query(vector=vector.tolist(), top_k=limit, include_metadata=True)
        return results.matches

    def add_image(self, image_path, description=None, custom_id=None):
        """Add an image to the database"""
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Create embedding
            vector = self.clip.encode_image(image_path, self.model, self.preprocess)
            
            # Use custom ID or generate one
            item_id = custom_id or str(uuid.uuid4())
            metadata = {
                'type': 'image',
                'name': os.path.basename(image_path),
                'description': description or ''
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
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Create embedding
            vector = self.clip.encode_text(text, self.model, self.tokenizer)
            
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
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Check if query is an image file
            if os.path.exists(query) and query.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Search with image
                vector = self.clip.encode_image(query, self.model, self.preprocess)
                print(f"🔍 Searching with image: {os.path.basename(query)}")
            else:
                # Search with text
                vector = self.clip.encode_text(query, self.model, self.tokenizer)
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
                if metadata['type'] == 'image':
                    print(f"{i}. 📷 {metadata['name']} (score: {score:.3f})")
                else:
                    print(f"{i}. 📝 {metadata['content'][:50]}... (score: {score:.3f})")
            
            return results.matches
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

# Simple usage examples
if __name__ == "__main__":
    # Create indexer
    indexer = SimpleIndexer()
    
    
    # Search with image
    # indexer.search("src/astro2.png")
    
    print("📚 Indexer is ready! Uncomment examples above to start using it.")