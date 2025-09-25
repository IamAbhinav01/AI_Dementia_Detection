
"""
Simple script to move captured canvas images from Downloads to project folder
Run this script after capturing images to organize them in your project
"""

import os
import shutil
from pathlib import Path
import time

def move_captured_images():
    # Get the current project directory
    project_dir = Path(__file__).parent
    downloads_dir = Path.home() / "Downloads"
    
    # Create a captures folder in your project if it doesn't exist
    captures_dir = project_dir / "captures"
    captures_dir.mkdir(exist_ok=True)
    
    # Look for captured images
    captured_files = list(downloads_dir.glob("ai-dementia-canvas-*.png"))
    
    if not captured_files:
        print("No captured canvas images found in Downloads folder.")
        return
    
    print(f"Found {len(captured_files)} captured image(s):")
    
    for file_path in captured_files:
        # Move the file to the captures folder
        destination = captures_dir / file_path.name
        shutil.move(str(file_path), str(destination))
        print(f"Moved: {file_path.name} -> captures/{file_path.name}")
    
    print(f"\nAll captured images moved to: {captures_dir}")
    print("You can now find your captured images in the 'captures' folder of your project.")

if __name__ == "__main__":
    move_captured_images()
