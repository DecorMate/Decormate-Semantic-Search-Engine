#!/usr/bin/env python3
import time
import os
from huggingface_hub import hf_hub_download

def download_with_retry(repo_id, filename, local_dir, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f'Attempt {attempt + 1}/{max_retries}: Downloading {filename}...')
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print('✅ Download successful!')
            return True
        except Exception as e:
            print(f'❌ Attempt {attempt + 1} failed: {e}')
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # Exponential backoff
                print(f'Waiting {wait_time} seconds before retry...')
                time.sleep(wait_time)
            else:
                print('❌ All download attempts failed!')
                raise e

if __name__ == "__main__":
    # Try to download the model
    download_with_retry('pcuenq/MobileCLIP-S1', 'mobileclip_s1.pt', 'models')