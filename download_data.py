"""
Download script for BCI Competition IV Dataset 2a
Note: You need to register at http://www.bbci.de/competition/iv/ to download
"""

import os
import urllib.request
import sys
import config

def download_subject(subject, url_template=None):
    """Download a single subject file"""
    if url_template is None:
        # Replace with actual URL after registration
        url_template = "http://www.bbci.de/competition/iv/download/{}.gdf"
    
    url = url_template.format(subject)
    filename = os.path.join(config.DATA_PATH, f"{subject}.gdf")
    
    if os.path.exists(filename):
        print(f"  {subject}.gdf already exists, skipping")
        return True
    
    try:
        print(f"  Downloading {subject}.gdf...")
        urllib.request.urlretrieve(url, filename)
        print(f"    Saved to {filename}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False

def main():
    """Main download function"""
    print("=" * 60)
    print("BCI Competition IV Dataset 2a - Download Helper")
    print("=" * 60)
    print("\nIMPORTANT: You need to register at http://www.bbci.de/competition/iv/")
    print("to download these files. After registration, you'll receive download links.")
    print("\nManual download alternative:")
    print("1. Visit: http://www.bbci.de/competition/iv/")
    print("2. Register and log in")
    print("3. Navigate to 'Dataset 2a'")
    print("4. Download all .gdf files (approximately 2.5 GB)")
    print("5. Place them in the 'data' folder")
    print("\n" + "=" * 60)
    
    response = input("\nDo you want to attempt automatic download? (y/n): ")
    if response.lower() != 'y':
        print("\nExiting. Please download files manually.")
        return
    
    # Create data directory
    os.makedirs(config.DATA_PATH, exist_ok=True)
    
    # Ask for URL template
    print("\nEnter the base URL for downloads (e.g., http://example.com/{}.gdf)")
    print("Or press Enter to use the default (likely won't work without registration)")
    url_template = input("URL template: ").strip()
    
    if not url_template:
        url_template = "http://www.bbci.de/competition/iv/download/{}.gdf"
    
    # Download all subjects
    print("\nDownloading training subjects...")
    successful = []
    for subject in config.SUBJECTS_TRAIN:
        if download_subject(subject, url_template):
            successful.append(subject)
    
    print(f"\nDownloaded {len(successful)}/{len(config.SUBJECTS_TRAIN)} subjects")
    
    if len(successful) < len(config.SUBJECTS_TRAIN):
        print("\nSome downloads failed. Please consider downloading manually.")
        print("You can also test the analysis with just A09T.gdf which you already have.")

if __name__ == "__main__":
    main()