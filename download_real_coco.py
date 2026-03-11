"""
Download a small real COCO dataset for testing
Downloads actual COCO 2017 images (not synthetic)
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def download_file(url, filepath, filename):
    """Download a single file"""
    try:
        print(f"  ⬇️  Downloading {filename}...", end='\r')
        urllib.request.urlretrieve(url, filepath)
        print(f"  ✓ Downloaded {filename:<40}")
        return True
    except Exception as e:
        print(f"  ✗ Failed {filename}: {e}")
        return False


def download_coco_annotations(output_dir):
    """Download COCO annotation files"""
    print("\n📥 Downloading COCO annotations...")
    
    anno_dir = Path(output_dir) / 'annotations'
    anno_dir.mkdir(parents=True, exist_ok=True)
    
    # Official COCO URLs
    train_anno_url = "http://images.cocodataset.org/annotations/instances_train2017.json"
    val_anno_url = "http://images.cocodataset.org/annotations/instances_val2017.json"
    
    train_anno_path = anno_dir / 'instances_train2017.json'
    val_anno_path = anno_dir / 'instances_val2017.json'
    
    print("  Train annotations...")
    if not train_anno_path.exists():
        download_file(train_anno_url, str(train_anno_path), 'train2017.json')
    else:
        print(f"  ✓ Train annotations already exist")
    
    print("  Val annotations...")
    if not val_anno_path.exists():
        download_file(val_anno_url, str(val_anno_path), 'val2017.json')
    else:
        print(f"  ✓ Val annotations already exist")
    
    return train_anno_path, val_anno_path


def filter_coco_subset(anno_path, output_path, num_images=100, seed=42):
    """
    Create a subset of COCO with only specified number of images
    """
    print(f"\n  Filtering COCO to {num_images} images...")
    
    with open(anno_path, 'r') as f:
        coco = json.load(f)
    
    # Seed for reproducibility
    import random
    random.seed(seed)
    
    # Sample images
    selected_image_ids = random.sample(
        [img['id'] for img in coco['images']], 
        min(num_images, len(coco['images']))
    )
    selected_image_ids_set = set(selected_image_ids)
    
    # Filter images and annotations
    filtered_images = [img for img in coco['images'] if img['id'] in selected_image_ids_set]
    filtered_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in selected_image_ids_set]
    
    # Create filtered COCO
    filtered_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco.get('categories', [])
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(filtered_coco, f)
    
    print(f"  ✓ Created subset: {len(filtered_images)} images, {len(filtered_annotations)} annotations")
    
    return selected_image_ids


def download_coco_images(image_ids, output_dir, coco_type='train2017', num_workers=4):
    """
    Download specific COCO images
    
    Args:
        image_ids: List of image IDs to download
        output_dir: Output directory
        coco_type: 'train2017' or 'val2017'
        num_workers: Number of parallel downloads
    """
    
    print(f"\n📥 Downloading {len(image_ids)} real COCO {coco_type} images...")
    
    img_dir = Path(output_dir) / coco_type
    img_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = f"http://images.cocodataset.org/zips/{coco_type}".replace('2017', '')
    
    # COCO image URL pattern
    def get_image_url(image_id):
        return f"http://images.cocodataset.org/{coco_type}/{image_id:012d}.jpg"
    
    # Download with threading
    downloaded = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for image_id in image_ids:
            filepath = img_dir / f"{image_id:012d}.jpg"
            
            # Skip if already exists
            if filepath.exists():
                downloaded += 1
                continue
            
            url = get_image_url(image_id)
            future = executor.submit(download_file, url, str(filepath), f"{image_id:012d}.jpg")
            futures[future] = image_id
        
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                downloaded += 1
            else:
                failed += 1
            
            if (i) % max(1, len(futures) // 10) == 0:
                print(f"  Progress: {i}/{len(futures)} images")
    
    print(f"  ✓ Downloaded {downloaded} images, Failed: {failed}")
    
    return downloaded


def setup_coco_dataset(output_dir, num_train=100, num_val=20):
    """
    Setup a small real COCO dataset
    
    Args:
        output_dir: Where to save COCO
        num_train: Number of training images
        num_val: Number of validation images
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("🎯 Setting up Real COCO Dataset")
    print("="*70)
    print(f"Location: {output_dir}")
    print(f"Train images: {num_train}")
    print(f"Val images: {num_val}")
    
    # Step 1: Download annotations
    train_anno_path, val_anno_path = download_coco_annotations(str(output_dir))
    
    # Step 2: Filter to subset
    print("\n📊 Creating dataset subsets...")
    
    train_anno_filtered = output_dir / 'annotations' / 'instances_train2017.json'
    val_anno_filtered = output_dir / 'annotations' / 'instances_val2017.json'
    
    train_ids = filter_coco_subset(train_anno_path, train_anno_filtered, num_train)
    val_ids = filter_coco_subset(val_anno_path, val_anno_filtered, num_val)
    
    # Step 3: Download images
    print("\n⏳ This may take a few minutes depending on connection speed...")
    print("   (Will download ~50MB for 100 images)\n")
    
    download_coco_images(train_ids, str(output_dir), 'train2017', num_workers=4)
    download_coco_images(val_ids, str(output_dir), 'val2017', num_workers=4)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ Real COCO Dataset Setup Complete!")
    print("="*70)
    print(f"\nLocation: {output_dir}")
    print(f"Structure:")
    print(f"  ├── train2017/        ({num_train} real images)")
    print(f"  ├── val2017/          ({num_val} real images)")
    print(f"  └── annotations/")
    print(f"      ├── instances_train2017.json")
    print(f"      └── instances_val2017.json")
    
    print(f"\nUsage:")
    print(f"  python train.py --data-path {output_dir} --task detection \\")
    print(f"                  --model vca_deit_tiny --epochs 5 --batch-size 4")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download real COCO dataset subset')
    parser.add_argument('--output-dir', type=str, default='/tmp/coco_real',
                       help='Output directory for dataset')
    parser.add_argument('--num-train', type=int, default=100,
                       help='Number of training images (default: 100)')
    parser.add_argument('--num-val', type=int, default=20,
                       help='Number of validation images (default: 20)')
    
    args = parser.parse_args()
    
    try:
        setup_coco_dataset(args.output_dir, args.num_train, args.num_val)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("You can resume by running the same command again")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Make sure you have internet connection and enough disk space")
