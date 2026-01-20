import requests
import os
from pathlib import Path

#Configuration
BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
OUTPUT_DIR = "tennis_data"

# Files to download
FILES_TO_DOWNLOAD = [
    "2025.csv",
    "2026.csv"
]

# Optional: Add rankings files if needed
RANKINGS_FILES = [
    #None for TML Database
]

def download_file(filename, output_dir):
    """Download a single file from GitHub"""
    url = f"{BASE_URL}/{filename}"
    output_path = Path(output_dir) / filename
    
    print(f"Downloading {filename}...", end=" ")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_size = len(response.content) / (1024 * 1024)  # MB
        print(f"✓ ({file_size:.2f} MB)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        return False

def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"Downloading files to '{OUTPUT_DIR}/'...\n")
    
    # Download main files
    success_count = 0
    for filename in FILES_TO_DOWNLOAD:
        if download_file(filename, OUTPUT_DIR):
            success_count += 1
    
    # Optional: Download rankings
    download_rankings = input("\nDownload rankings files? (y/n): ").lower() == 'y'
    if download_rankings:
        for filename in RANKINGS_FILES:
            if download_file(filename, OUTPUT_DIR):
                success_count += 1
    
    print(f"\n✓ Successfully downloaded {success_count} files to '{OUTPUT_DIR}/'")
    print(f"\nNext steps:")
    print(f"1. cd {OUTPUT_DIR}")
    print(f"2. python -c \"import pandas as pd; df = pd.read_csv('atp_matches_2024.csv'); print(df.head())\"")

if __name__ == "__main__":
    main()