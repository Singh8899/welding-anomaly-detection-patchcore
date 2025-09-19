#!/usr/bin/env python3
"""
Welding Region Extractor
Extracts welding regions from images using XML annotations.
Saves cropped and resized patches for PatchCore training.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

class WeldExtractor:
    """Extract welding regions from images using XML annotations"""
    
    def __init__(self, output_size=(224, 224)):
        """
        Initialize the extractor
        
        Args:
            output_size (tuple): Size to resize extracted patches to
        """
        self.output_size = output_size
        
    def extract_from_xml(self, image_path, xml_path):
        """
        Extract welding regions using XML annotations
        
        Args:
            image_path (str): Path to input image
            xml_path (str): Path to XML annotation file
            
        Returns:
            list: List of (patch, class_name, bbox) tuples
        """
        if not os.path.exists(xml_path):
            return []
            
        # Parse XML annotation
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return []
            
        patches = []
        
        # Extract each object
        for obj in root.findall('object'):
            # Get class name
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            class_name = name_elem.text
            
            # Get bounding box
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Extract and resize patch
            patch = image[ymin:ymax, xmin:xmax]
            if patch.size > 0:
                # Make patch square by padding
                patch_square = self.make_square(patch)
                # Resize to target size
                patch_resized = cv2.resize(patch_square, self.output_size)
                patches.append((patch_resized, class_name, (xmin, ymin, xmax, ymax)))
                
        return patches
    
    def make_square(self, image):
        """
        Make image square by padding with zeros
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Square image
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        # Create square canvas
        if len(image.shape) == 3:
            square = np.zeros((max_dim, max_dim, image.shape[2]), dtype=image.dtype)
        else:
            square = np.zeros((max_dim, max_dim), dtype=image.dtype)
            
        # Calculate padding
        top = (max_dim - h) // 2
        left = (max_dim - w) // 2
        
        # Place image in center
        square[top:top+h, left:left+w] = image
        
        return square
    
    def process_dataset(self, input_dir, output_dir):
        """
        Process entire dataset and extract welding patches using XML annotations
        
        Args:
            input_dir (str): Directory containing images and XML files
            output_dir (str): Directory to save extracted patches
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        good_dir = output_path / 'good'
        bad_dir = output_path / 'bad'
        all_dir = output_path / 'all'  # For single class detection
        
        for dir_path in [good_dir, bad_dir, all_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process all images
        image_files = list(input_path.glob('*.jpeg')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        total_patches = 0
        good_patches = 0
        bad_patches = 0
        
        for img_file in image_files:
            print(f"Processing: {img_file.name}")
            
            # Use XML annotations
            xml_file = img_file.with_suffix('.xml')
            patches = self.extract_from_xml(str(img_file), str(xml_file))
            
            for i, (patch, class_name, bbox) in enumerate(patches):
                patch_name = f"{img_file.stem}_patch_{i}.jpg"
                
                # Save to appropriate directory
                if class_name == 'good_weld':
                    cv2.imwrite(str(good_dir / patch_name), patch)
                    good_patches += 1
                elif class_name == 'bad_weld':
                    cv2.imwrite(str(bad_dir / patch_name), patch)
                    bad_patches += 1
                
                # Also save to 'all' directory for single class training
                cv2.imwrite(str(all_dir / patch_name), patch)
                total_patches += 1
        
        print(f"\nExtraction completed!")
        print(f"Total patches extracted: {total_patches}")
        print(f"Good welds: {good_patches}")
        print(f"Bad welds: {bad_patches}")
        print(f"Patches saved to: {output_path}")

def main():
    """Main extraction pipeline"""
    
    # Configuration
    base_dir = Path(__file__).parent.parent
    train_dir = base_dir / 'downloaded_photos' / 'train'
    val_dir = base_dir / 'downloaded_photos' / 'val'
    output_dir = base_dir / 'extracted_welds'
    
    print("=== Welding Region Extraction ===")
    
    # Initialize extractor
    extractor = WeldExtractor(output_size=(224, 224))
    
    # Process training set
    if train_dir.exists():
        print(f"\nProcessing training set: {train_dir}")
        train_output = output_dir / 'train'
        extractor.process_dataset(
            input_dir=str(train_dir),
            output_dir=str(train_output)
        )
    
    # Process validation set
    if val_dir.exists():
        print(f"\nProcessing validation set: {val_dir}")
        val_output = output_dir / 'val'
        extractor.process_dataset(
            input_dir=str(val_dir),
            output_dir=str(val_output)
        )

if __name__ == "__main__":
    main()
