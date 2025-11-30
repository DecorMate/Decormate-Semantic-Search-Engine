import pinecone
import torch
from pinecone import Pinecone , ServerlessSpec
import os 
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoProcessor, AutoModel

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pc = Pinecone(api_key= os.environ.get('PINECONE_API_KEY'), enviroment=os.environ.get('PINECONE_ENV'))

def delete_exiting_index(index_name):
    if index_name in [index.name for index in pc.list_indexes()]:
        pc.delete_index(index_name)
        print(f"[info] {index_name} Deleted Succsessfully")
    else : 
        print(f'[info] {index_name} not in the list')

MODEL_NAME = 'BAAI/EVA02-CLIP-B-16'

model = AutoModel.from_pretrained(MODEL_NAME, cache_dir='./').to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir='./')

# dimensions 
EMBED_dim = model.visual_projection.out_features

def create_new_index(index_name, dim = EMBED_dim, metrice="dotproduct"):
    if index_name  not in [index.name for index in pc.list_indexes()]:
        # create index 
        pc.create_index(
            name= index_name,
            dimension= dim,
            spec= ServerlessSpec(
                cloud= 'aws',
                region='us-east-1'
            )
        )
        print(f"{index_name} created Sucssefully")
    else: 
        print(f'[info] {index_name}  the index already exist')

if __name__ == "__main__":
    index_name = 'Decormate'
    create_new_index(index_name)

