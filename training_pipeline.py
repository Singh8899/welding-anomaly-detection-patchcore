#!/usr/bin/env python3
"""
Complete Training Pipeline for Welding Anomaly Detection

This script implements a two-stage training pipeline:
1. YOLO training for welding detection (single class)
2. PatchCore training for anomaly detection (good vs bad welds)

Usage:
    python training_pipeline.py [--skip-yolo] [--skip-extraction] [--skip-patchcore]
"""

import os
import sys
import argparse
from pathlib import Path

# Add project directories to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'yolo'))
sys.path.append(str(project_root / 'patchcore'))

def stage1_format_dataset():
    """Stage 1: Format dataset for YOLO training"""
    print("\n" + "="*60)
    print("STAGE 1: FORMATTING DATASET FOR YOLO")
    print("="*60)
    
    try:
        from dataset_formatter import main as format_main
        format_main()
        print("✓ Dataset formatting completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error in dataset formatting: {e}")
        return False

def stage2_train_yolo():
    """Stage 2: Train YOLO model for welding detection"""
    print("\n" + "="*60)
    print("STAGE 2: TRAINING YOLO MODEL")
    print("="*60)
    
    try:
        from yolo_train import main as yolo_main
        yolo_main()
        print("✓ YOLO training completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error in YOLO training: {e}")
        return False

def stage3_extract_welds():
    """Stage 3: Extract welding patches from images"""
    print("\n" + "="*60)
    print("STAGE 3: EXTRACTING WELDING PATCHES")
    print("="*60)
    
    try:
        from extract_welds import main as extract_main
        extract_main()
        print("✓ Welding patch extraction completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error in patch extraction: {e}")
        return False

def stage4_train_patchcore():
    """Stage 4: Train PatchCore for anomaly detection"""
    print("\n" + "="*60)
    print("STAGE 4: TRAINING PATCHCORE FOR ANOMALY DETECTION")
    print("="*60)
    
    try:
        from welding_patchcore import main as patchcore_main
        patchcore_main()
        print("✓ PatchCore training completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error in PatchCore training: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import torchvision
        print(f"✓ Torchvision {torchvision.__version__}")
    except ImportError:
        missing_deps.append("torchvision")
        
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
        
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        from sklearn import metrics
        print(f"✓ Scikit-learn")
    except ImportError:
        missing_deps.append("scikit-learn")
        
    try:
        from ultralytics import YOLO
        print(f"✓ Ultralytics")
    except ImportError:
        missing_deps.append("ultralytics")
        
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        missing_deps.append("matplotlib")
        
    try:
        from PIL import Image
        print(f"✓ Pillow")
    except ImportError:
        missing_deps.append("Pillow")
        
    try:
        import yaml
        print(f"✓ PyYAML")
    except ImportError:
        missing_deps.append("PyYAML")
    
    if missing_deps:
        print(f"\n✗ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print("✓ All dependencies are installed")
    return True

def check_data_structure():
    """Check if the required data structure exists"""
    print("\nChecking data structure...")
    
    base_dir = Path(__file__).parent
    train_dir = base_dir / 'downloaded_photos' / 'train'
    val_dir = base_dir / 'downloaded_photos' / 'val'
    
    issues = []
    
    if not train_dir.exists():
        issues.append(f"Training directory not found: {train_dir}")
    else:
        # Check for images and XML files
        images = list(train_dir.glob('*.jpeg')) + list(train_dir.glob('*.jpg'))
        xmls = list(train_dir.glob('*.xml'))
        
        if len(images) == 0:
            issues.append(f"No JPEG images found in {train_dir}")
        if len(xmls) == 0:
            issues.append(f"No XML annotation files found in {train_dir}")
        
        print(f"✓ Found {len(images)} images and {len(xmls)} annotations in training set")
    
    if val_dir.exists():
        images = list(val_dir.glob('*.jpeg')) + list(val_dir.glob('*.jpg'))
        xmls = list(val_dir.glob('*.xml'))
        print(f"✓ Found {len(images)} images and {len(xmls)} annotations in validation set")
    else:
        print("! Validation directory not found (optional)")
    
    if issues:
        print("\n✗ Data structure issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✓ Data structure looks good")
    return True

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description='Complete Welding Anomaly Detection Training Pipeline')
    parser.add_argument('--skip-yolo', action='store_true', help='Skip YOLO training stage')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip patch extraction stage')
    parser.add_argument('--skip-patchcore', action='store_true', help='Skip PatchCore training stage')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and data checks')
    
    args = parser.parse_args()
    
    print("🔧 WELDING ANOMALY DETECTION TRAINING PIPELINE 🔧")
    print("="*60)
    print("This pipeline will:")
    print("1. Format dataset for YOLO training")
    print("2. Train YOLO model for welding detection")
    print("3. Extract welding patches from images")
    print("4. Train PatchCore for anomaly detection")
    print("="*60)
    
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            print("\\nPlease install missing dependencies and try again.")
            return 1
        
        # Check data structure
        if not check_data_structure():
            print("\\nPlease fix data structure issues and try again.")
            return 1
    
    # Track success of each stage
    success_stages = []
    failed_stages = []
    
    # Stage 1: Format dataset
    print("\\n🚀 Starting pipeline execution...")
    if stage1_format_dataset():
        success_stages.append("Dataset Formatting")
    else:
        failed_stages.append("Dataset Formatting")
        print("\\n❌ Pipeline stopped due to formatting failure")
        return 1
    
    # Stage 2: Train YOLO (optional skip)
    if not args.skip_yolo:
        if stage2_train_yolo():
            success_stages.append("YOLO Training")
        else:
            failed_stages.append("YOLO Training")
            print("\\n⚠️  YOLO training failed, but continuing with XML-based extraction...")
    else:
        print("\\n⏭️  Skipping YOLO training (using XML annotations)")
    
    # Stage 3: Extract welding patches (optional skip)
    if not args.skip_extraction:
        if stage3_extract_welds():
            success_stages.append("Patch Extraction")
        else:
            failed_stages.append("Patch Extraction")
            print("\\n❌ Pipeline stopped due to extraction failure")
            return 1
    else:
        print("\\n⏭️  Skipping patch extraction")
    
    # Stage 4: Train PatchCore (optional skip)
    if not args.skip_patchcore:
        if stage4_train_patchcore():
            success_stages.append("PatchCore Training")
        else:
            failed_stages.append("PatchCore Training")
    else:
        print("\\n⏭️  Skipping PatchCore training")
    
    # Final summary
    print("\\n" + "="*60)
    print("🏁 PIPELINE EXECUTION COMPLETED")
    print("="*60)
    
    if success_stages:
        print("✅ Successful stages:")
        for stage in success_stages:
            print(f"   • {stage}")
    
    if failed_stages:
        print("\\n❌ Failed stages:")
        for stage in failed_stages:
            print(f"   • {stage}")
    
    print("\\n📁 Output locations:")
    base_dir = Path(__file__).parent
    
    # YOLO outputs
    yolo_output = base_dir / 'yolo' / 'runs' / 'detect' / 'welding_detection'
    if yolo_output.exists():
        print(f"   • YOLO model: {yolo_output / 'weights' / 'best.pt'}")
    
    # Extracted patches
    patches_output = base_dir / 'extracted_welds'
    if patches_output.exists():
        print(f"   • Extracted patches: {patches_output}")
    
    # PatchCore results
    patchcore_output = base_dir / 'patchcore_results'
    if patchcore_output.exists():
        print(f"   • PatchCore results: {patchcore_output}")
    
    if len(failed_stages) == 0:
        print("\\n🎉 All stages completed successfully!")
        return 0
    else:
        print(f"\\n⚠️  Pipeline completed with {len(failed_stages)} failed stages")
        return 1

if __name__ == "__main__":
    sys.exit(main())
