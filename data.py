"""Data Loading and Augmentation"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from config import DataConfig, AugmentationConfig
try:
    from pycocotools.coco import COCO
    HAS_COCO = True
except ImportError:
    HAS_COCO = False


# ============================================================================
# Augmentation Transforms
# ============================================================================

class RandomCutout:
    """Random cutout augmentation"""
    def __init__(self, size=56, prob=0.5, value=0):
        self.size = size
        self.prob = prob
        self.value = value
    
    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        
        h, w = img.size
        mask = np.ones((h, w), dtype=np.float32) * self.value
        
        # Random position
        y = np.random.randint(0, h - self.size)
        x = np.random.randint(0, w - self.size)
        
        mask[y:y+self.size, x:x+self.size] = 1.0
        
        return img * transforms.ToTensor()(mask)


class Mixup:
    """Mixup data augmentation"""
    def __init__(self, alpha=0.8, num_classes=1000):
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(self, batch):
        images, targets = batch
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        target_a, target_b = targets, targets[index]
        
        return mixed_images, target_a, target_b, lam
    
    def __repr__(self):
        return f"Mixup(alpha={self.alpha})"


# ============================================================================
# Data Transform Builders
# ============================================================================

def build_train_transform(img_size=224):
    """Build training augmentation pipeline"""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])


def build_test_transform(img_size=224):
    """Build testing/validation transform"""
    # Resize to 256, then center crop to 224
    resize_size = int((256 / 224) * img_size)
    
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def build_detection_train_transform(img_size=512, min_size=480, max_size=1333):
    """Build training augmentation for detection"""
    return DetectionTransform(
        resize_size=(min_size, max_size),
        train=True,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225]
    )


def build_detection_eval_transform(img_size=512, min_size=480, max_size=1333):
    """Build eval augmentation for detection"""
    return DetectionTransform(
        resize_size=(min_size, max_size),
        train=False,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225]
    )


# ============================================================================
# Detection-specific Transforms
# ============================================================================

class DetectionTransform:
    """Transform for object detection with bounding boxes"""
    
    def __init__(self, resize_size=(480, 1333), train=True, 
                 normalize_mean=None, normalize_std=None):
        self.min_size, self.max_size = resize_size
        self.train = train
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
    
    def __call__(self, image, targets=None):
        """
        Args:
            image: PIL Image
            targets: dict with 'boxes' [N, 4] and 'labels' [N]
        
        Returns:
            image: [C, H, W] normalized tensor
            targets: dict with transformed boxes and labels
        """
        if targets is None:
            targets = {}
        
        image = np.array(image)
        h, w = image.shape[:2]
        
        # Augmentation
        if self.train:
            if np.random.rand() > 0.5:
                image = image[:, ::-1]  # Horizontal flip
                if 'boxes' in targets:
                    boxes = targets['boxes']
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    targets['boxes'] = boxes
        
        # Resize
        scale = min(self.max_size / max(h, w), self.min_size / min(h, w))
        new_h, new_w = int(h * scale), int(w * scale)
        image = transforms.functional.resize(
            transforms.functional.pil_to_tensor(image),
            (new_h, new_w)
        ) / 255.0  # Normalize to [0, 1]
        
        # Update boxes for resize
        if 'boxes' in targets:
            targets['boxes'] = targets['boxes'] * scale
        
        # Pad to square
        max_dim = max(new_h, new_w)
        pad_h = max_dim - new_h
        pad_w = max_dim - new_w
        image = torch.nn.functional.pad(
            image.unsqueeze(0),
            (0, pad_w, 0, pad_h),
            value=0.0
        ).squeeze(0)
        
        # Normalize
        image = transforms.functional.normalize(
            image,
            mean=self.normalize_mean,
            std=self.normalize_std
        )
        
        targets['image_size'] = (new_h, new_w)
        targets['pad'] = (pad_h, pad_w)
        targets['scale'] = scale
        
        return image, targets
    
    def __repr__(self):
        return f"DetectionTransform(min_size={self.min_size}, max_size={self.max_size})"


# ============================================================================
# Custom COCO Detection Dataset
# ============================================================================

class COCODetection(Dataset):
    """Custom COCO detection dataset"""
    
    def __init__(self, coco_root, subset='train2017', transform=None, num_classes=80, annotation_path=None):
        """
        Args:
            coco_root: Path to COCO dataset root
            subset: 'train2017', 'val2017', etc.
            transform: Transform to apply
            num_classes: Number of classes (80 for COCO)
            annotation_path: Custom annotation directory (default: {coco_root}/annotations/)
        """
        self.coco_root = coco_root
        self.subset = subset
        self.transform = transform
        self.num_classes = num_classes
        
        # Initialize COCO API
        if not HAS_COCO:
            raise ImportError("pycocotools is required for COCO dataset. Install with: pip install pycocotools")
        
        if annotation_path:
            anno_file = os.path.join(annotation_path, f'instances_{subset}.json')
        else:
            anno_file = os.path.join(coco_root, f'annotations/instances_{subset}.json')
        if not os.path.exists(anno_file):
            raise FileNotFoundError(f"COCO annotation file not found: {anno_file}")
        
        self.coco = COCO(anno_file)
        self.ids = list(self.coco.imgToAnns.keys())
        
        # Filter images with at least one annotation
        self.ids = [id for id in self.ids if len(self.coco.imgToAnns[id]) > 0]
        
        print(f"COCODetection: Loaded {len(self.ids)} images from {subset}")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.coco_root, self.subset, img_info['file_name'])
        image = self._load_image(img_path)
        h, w = image.shape[:2]
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        for ann in anns:
            if ann['iscrowd']:
                continue
            
            x, y, bw, bh = ann['bbox']
            
            # Convert to [x1, y1, x2, y2] format
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'] - 1)  # COCO categories are 1-indexed
        
        if len(boxes) == 0:
            # Return dummy target if no valid boxes
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        
        targets = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'orig_size': (h, w)
        }
        
        # Apply transform
        if self.transform:
            image, targets = self.transform(image, targets)
        else:
            image = transforms.ToTensor()(image)
        
        return image, targets
    
    @staticmethod
    def _load_image(path):
        """Load image using PIL"""
        from PIL import Image
        return np.array(Image.open(path).convert('RGB'))


# ============================================================================
# Custom Collate Function for Detection
# ============================================================================

def detection_collate_fn(batch):
    """Collate function for detection batches with variable number of objects"""
    images = []
    targets_list = []
    
    for image, targets in batch:
        images.append(image)
        targets_list.append(targets)
    
    images = torch.stack(images, dim=0)
    
    return images, targets_list


# ============================================================================
# Dataset Builders
# ============================================================================

def build_dataset(data_path, is_train=True, img_size=224):
    """Build ImageFolder dataset"""
    
    if is_train:
        transform = build_train_transform(img_size)
        subset = 'train'
    else:
        transform = build_test_transform(img_size)
        subset = 'val'
    
    dataset_path = os.path.join(data_path, subset)
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path not found: {dataset_path}")
    
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return dataset


def build_fake_dataset(num_samples=1000, img_size=224, num_classes=1000):
    """Build fake dataset for testing"""
    return datasets.FakeData(
        size=num_samples,
        image_size=(3, img_size, img_size),
        num_classes=num_classes,
        transform=transforms.ToTensor()
    )


# ============================================================================
# DataLoader Builder
# ============================================================================

def build_dataloader(config: DataConfig, is_train=True, use_fake=False, 
                    distributed=False, rank=0, world_size=1):
    """
    Build DataLoader for training or validation
    
    Args:
        config: DataConfig object
        is_train: If True, build training set; else validation set
        use_fake: If True, use FakeData for debugging
        distributed: If True, use DistributedSampler
        rank: Rank of current process (for distributed training)
        world_size: Number of processes (for distributed training)
    
    Returns:
        DataLoader, Dataset
    """
    
    if config.dataset == 'coco':
        return build_detection_dataloader(
            config, is_train=is_train, distributed=distributed,
            rank=rank, world_size=world_size
        )
    
    # Build dataset
    if use_fake:
        dataset = build_fake_dataset(
            num_samples=1000,
            img_size=config.img_size,
            num_classes=1000
        )
    else:
        dataset = build_dataset(
            config.data_path,
            is_train=is_train,
            img_size=config.img_size
        )
    
    # Build sampler
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=is_train
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = is_train
    
    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=is_train,
        persistent_workers=(config.num_workers > 0)
    )
    
    return dataloader, dataset


def build_detection_dataloader(config: DataConfig, is_train=True, 
                              distributed=False, rank=0, world_size=1):
    """
    Build DataLoader for object detection (COCO)
    
    Args:
        config: DataConfig object with dataset='coco'
        is_train: If True, build training set; else validation set
        distributed: If True, use DistributedSampler
        rank: Rank of current process
        world_size: Number of processes
    
    Returns:
        DataLoader, Dataset
    """
    
    # Build transform
    if is_train:
        transform = build_detection_train_transform(
            img_size=config.img_size,
            min_size=config.min_size if hasattr(config, 'min_size') else 480,
            max_size=config.max_size if hasattr(config, 'max_size') else 1333
        )
        subset = 'train2017'
    else:
        transform = build_detection_eval_transform(
            img_size=config.img_size,
            min_size=config.min_size if hasattr(config, 'min_size') else 480,
            max_size=config.max_size if hasattr(config, 'max_size') else 1333
        )
        subset = 'val2017'
    
    # Build dataset
    dataset = COCODetection(
        coco_root=config.data_path,
        subset=subset,
        transform=transform,
        num_classes=80,  # COCO has 80 classes
        annotation_path=getattr(config, 'annotation_path', None)
    )
    
    # Build sampler
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=is_train
        )
        shuffle = False
    else:
        sampler = None
        shuffle = is_train
    
    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=is_train,
        persistent_workers=(config.num_workers > 0),
        collate_fn=detection_collate_fn
    )
    
    return dataloader, dataset


# ============================================================================
# Utility Functions
# ============================================================================

def mixup_batch(batch, alpha=0.8, num_classes=1000):
    """Apply mixup to a batch"""
    images, targets = batch
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    target_a = targets
    target_b = targets[index]
    
    return mixed_images, target_a, target_b, lam


def cutmix_batch(batch, alpha=1.0):
    """Apply cutmix to a batch"""
    images, targets = batch
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = images.size(0)
    _, _, h, w = images.shape
    
    index = torch.randperm(batch_size)
    
    # Random box size
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    # Random position
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
    
    target_a = targets
    target_b = targets[index]
    
    return images, target_a, target_b, lam
