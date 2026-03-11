# VCA-DeiT Baseline - Visual Contrast Attention Vision Transformer

Một implementation từ scratch của **DeiT-tiny với Visual Contrast Attention (VCA)** - một cơ chế chú ý tuyến tính nhằm giảm độ phức tạp tính toán trong Vision Transformers.

## 📁 Cấu Trúc Dự Án

```
VCA_DeiT_baseline/
├── config.py               # Configuration classes
├── models.py               # VCA-DeiT architecture (from scratch)
├── data.py                 # Data loading & augmentation
├── train.py                # Training script
├── utils.py                # Utility functions (logger, checkpoint, etc.)
├── download_real_coco.py   # Download real COCO dataset
├── __init__.py             # Package init
└── README.md               # This file
```

## 🎯 Tính Năng Chính

### Tasks Supported
- ✅ **Image Classification**: ImageNet classification with VCA-DeiT
- ✅ **Object Detection**: Multi-object detection on COCO dataset

### Visual Contrast Attention (VCA)
- **2-stage attention mechanism** giảm độ phức tạp từ O(N²) → O(Nn)
- **Stage I**: Global contrast sử dụng visual contrast tokens (t_+, t_-)
- **Stage II**: Patch-wise differential attention
- **RMSNorm**: Normalization hiệu quả hơn LayerNorm
- **Learnable lambda parameters**: Điều chỉnh trọng số attention

### Model Variants
- **vca_deit_tiny**: 12 layers, 192 embed_dim, 3 heads (5.7M params)
- **vca_deit_small**: 12 layers, 384 embed_dim, 6 heads (22.5M params)
- **vca_deit_base**: 12 layers, 768 embed_dim, 12 heads (86.6M params)

### Training Features
- ✅ Multi-GPU training (DistributedDataParallel)
- ✅ Mixed Precision Training (AMP)
- ✅ Mixup/Cutmix augmentation
- ✅ Cosine annealing scheduler
- ✅ Gradient clipping
- ✅ Checkpoint management

## 🚀 Nhanh Chóng Bắt Đầu

### 1. Installation

```bash
# Tạo virtual environment (tuỳ chọn)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy
```

### 2. Quick Test

```bash
cd /media/tan/F/RESEARCH/VCA_DeiT_baseline

# Test model
python -c "
from models import create_model
import torch
model = create_model('vca_deit_tiny')
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f'Output: {out.shape}')  # [2, 1000]
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print('OK!')
"
```

### 3. Training on ImageNet

#### Single GPU
```bash
python train.py \
    --data-path /path/to/imagenet \
    --model vca_deit_tiny \
    --batch-size 128 \
    --epochs 300 \
    --lr 5e-4 \
    --output-dir output/vca_deit_tiny
```

#### Multi-GPU (8 GPUs)
```bash
torchrun --nproc_per_node=8 train.py \
    --data-path /path/to/imagenet \
    --model vca_deit_tiny \
    --batch-size 128 \
    --epochs 300 \
    --lr 5e-4 \
    --output-dir output/vca_deit_tiny
```

#### With Fake Data (Testing)
```bash
python train.py \
    --data-path /path/to/imagenet \
    --model vca_deit_tiny \
    --fake-data \
    --epochs 5 \
    --batch-size 32 \
    --output-dir output/test
```

### 4. Evaluation

```bash
python train.py \
    --data-path /path/to/imagenet \
    --model vca_deit_tiny \
    --resume output/vca_deit_tiny/ckpt_best.pth \
    --eval
```

## 📋 Training Arguments

### Data Arguments
- `--data-path`: Đường dẫn đến ImageNet dataset
- `--img-size`: Kích thước ảnh (default: 224)
- `--num-workers`: Số workers cho data loading (default: 8)
- `--fake-data`: Sử dụng fake data cho testing

### Model Arguments
- `--model`: Loại model (vca_deit_tiny, vca_deit_small, vca_deit_base)
- `--drop-path`: Drop path rate (default: 0.1)

### Training Arguments
- `--batch-size`: Batch size mỗi GPU (default: 128)
- `--epochs`: Số epochs (default: 300)
- `--warmup-epochs`: Warmup epochs (default: 20)
- `--lr`: Base learning rate (default: 5e-4)
- `--min-lr`: Min learning rate (default: 5e-6)
- `--weight-decay`: Weight decay (default: 0.05)
- `--clip-grad`: Gradient clipping (default: 5.0)

### Augmentation Arguments
- `--mixup`: Mixup alpha (default: 0.8)
- `--cutmix`: Cutmix alpha (default: 1.0)

### Checkpoint Arguments
- `--resume`: Resume từ checkpoint
- `--pretrained`: Load pretrained weights
- `--output-dir`: Output directory

### Other Arguments
- `--amp`: Sử dụng Automatic Mixed Precision
- `--eval`: Evaluation only mode
- `--print-freq`: Print frequency (default: 100)
- `--seed`: Random seed (default: 0)

## 📊 Cấu Trúc Code

### config.py
```python
@dataclass
class Config:
    data: DataConfig          # Data settings
    model: ModelConfig        # Model settings
    aug: AugmentationConfig   # Augmentation settings
    train: TrainConfig        # Training settings
```

### models.py - Kiến Trúc VCA-DeiT

**Main Classes:**
- `RMSNorm`: Root Mean Square Normalization
- `VisualContrastAttention`: VCA mechanism (2 stages)
- `VisualContrastBlock`: Transformer block with VCA
- `TransformerBlock`: Standard transformer block (for non-VCA layers)
- `PatchEmbed`: Image to patch embedding
- `VisionTransformer`: Complete model

**Key Components:**
```python
# Visual Contrast Token Generation
t_tilde = AvgPool2d(q_spatial)
t_pos = e_pos + t_tilde      # Positive tokens
t_neg = e_neg + t_tilde      # Negative tokens

# Stage I: Global Contrast
lambda_1 = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init
v_hat = v_hat_+ - lambda_1 * v_hat_-
v_hat = RMSNorm(v_hat) * (1 - lambda_1)

# Stage II: Patch-wise Differential Attention
lambda_2 = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init
A = A_1 - lambda_2 * A_2
x = RMSNorm(A @ v_hat) * (1 - lambda_2)
```

### data.py - Data Loading
- `build_train_transform()`: Training augmentation pipeline
- `build_test_transform()`: Validation transform
- `build_dataset()`: Load ImageNet dataset
- `build_dataloader()`: Create DataLoader with distributed support
- `mixup_batch()`: Mixup augmentation
- `cutmix_batch()`: Cutmix augmentation

### train.py - Training Loop
```
main()
├── Parse arguments
├── Setup distributed training
├── Build config & model
├── Build optimizer & scheduler
├── Load dataloaders
├── Training loop (300 epochs)
│   ├── train_one_epoch()
│   │   ├── Forward pass
│   │   ├── Loss computation
│   │   ├── Backward pass
│   │   └── Optimizer step
│   └── validate()
│       ├── Compute top-1 & top-5 accuracy
│       └── Save best checkpoint
└── End training
```

### utils.py - Utilities
- `create_logger()`: Setup logging
- `save_checkpoint()`: Save model checkpoint
- `load_checkpoint()`: Load checkpoint
- `AverageMeter`: Track metrics
- `accuracy()`: Compute top-k accuracy
- `build_optimizer()`: Create optimizer
- `build_scheduler()`: Create LR scheduler
- `CosineLRScheduler`: Cosine annealing

## 🔧 Configuration Example

**Default Configuration (vca_deit_tiny_300ep):**

```python
# Data
batch_size = 128
img_size = 224
num_workers = 8

# Model
embed_dim = 192
depth = 12
num_heads = 3
drop_path_rate = 0.1
vct_num = [49, 49, 49, 49]  # Visual contrast tokens

# Training
epochs = 300
warmup_epochs = 20
base_lr = 5e-4
min_lr = 5e-6
weight_decay = 0.05
optimizer = adamw

# Scheduler
scheduler = cosine
warmup_lr = 5e-7

# Augmentation
mixup = 0.8
cutmix = 1.0
color_jitter = 0.4
```

## 📈 Expected Results

**VCA-DeiT-Tiny @ 224x224:**
- ImageNet-1K Top-1: ~72-74% (with proper training)
- Training time: ~36 hours on 8 V100 GPUs
- Memory: ~11GB per GPU

## 🔍 Code Explanation

### Visual Contrast Attention - Detailed

```python
# Input: [B, N, C] where N = 196 + 1 (patches + cls token)

# 1. Generate visual contrast tokens via adaptive pooling
q_spatial = q[:, 1:, :]  # Exclude cls token
t_tilde = AdaptiveAvgPool2d(q_spatial)  # [B, 49, C]

# 2. Add learnable positional embeddings
t_pos = e_pos + t_tilde  # [B, 49, C] - positive tokens
t_neg = e_neg + t_tilde  # [B, 49, C] - negative tokens

# 3. Stage I: Global Contrast
# Both t_+ and t_- attend to all k,v
t_all = cat(t_pos, t_neg)  # [B, M, 98, d]
v_hat_all = softmax(t_all @ k^T) @ v  # [B, M, 98, d]

# Differential combining with learnable lambda
lambda_1 = exp(...) - exp(...) + init
v_hat = v_hat_+ - lambda_1 * v_hat_-

# 4. Stage II: Patch-wise Differential Attention
A_1 = softmax(q @ t_+^T)  # [B, M, N, 49]
A_2 = softmax(q @ t_-^T)  # [B, M, N, 49]
lambda_2 = exp(...) - exp(...) + init
A = A_1 - lambda_2 * A_2

# Final output
x = A @ v_hat  # [B, M, N, d]
x = RMSNorm(x) * (1 - lambda_2)
```

### Complexity Analysis

Standard Attention: O(N²d)
- N=196 patches: 196² = 38,416 operations/head

VCA Attention: O((N+2n)nd)
- n=49 tokens: (196 + 98) * 49 * d ≈ 14,406 operations/head
- **~62% reduction in attention complexity**

## 🐛 Debugging

### Test Forward Pass
```python
python test.py
```

### Training with Fake Data (Small Dataset)
```bash
python train.py \
    --data-path dummy \
    --fake-data \
    --epochs 5 \
    --batch-size 32
```

### Check Model Architecture
```python
from models import create_model
model = create_model('vca_deit_tiny')
print(model)
```

## 🎯 Object Detection Mode

VCA-DeiT hỗ trợ **multi-object detection** trên COCO dataset:

### Detection Model Creation

```python
from models import create_model
import torch

# Tạo detection model
model = create_model(
    model_type='vca_deit_tiny',
    task='detection',
    img_size=512,
    num_classes=80,      # COCO classes
    num_queries=100      # Detection queries
)

# Forward pass
images = torch.randn(2, 3, 512, 512)
class_logits, bbox_pred = model(images)

print(f"Class logits: {class_logits.shape}")  # [2, 100, 81]
print(f"Bbox predictions: {bbox_pred.shape}")  # [2, 100, 4]
```

### Detection Training

#### COCO Dataset Setup

**Option 1: Script tự động (subset nhỏ để test)**
```bash
# Download 100 train + 20 val images (~50MB)
python download_real_coco.py --output-dir /tmp/coco_data --num-train 100 --num-val 20

# Download lớn hơn
python download_real_coco.py --output-dir /tmp/coco_data --num-train 1000 --num-val 200
```

**Option 2: Download thủ công (full dataset)**

Vào https://cocodataset.org/#download và tải:
- 2017 Train images (13GB)
- 2017 Val images (6.4GB)  
- 2017 Train/Val annotations (307MB)

```bash
# Extract and organize
unzip train2017.zip -d /path/to/coco/
unzip val2017.zip -d /path/to/coco/
unzip annotations_trainval2017.zip -d /path/to/coco/
```

#### Training Detection Model
```bash
# Single GPU
python train.py \
    --data-path /path/to/coco \
    --task detection \
    --model vca_deit_tiny \
    --batch-size 16 \
    --img-size 512 \
    --epochs 100 \
    --lr 1e-4 \
    --output-dir output/detection

# Multi-GPU
torchrun --nproc_per_node=4 train.py \
    --data-path /path/to/coco \
    --task detection \
    --model vca_deit_tiny \
    --batch-size 16 \
    --epochs 100 \
    --output-dir output/detection
```

#### Evaluation
```bash
python train.py \
    --data-path /path/to/coco \
    --task detection \
    --resume output/detection/ckpt_best.pth \
    --eval
```

### Testing Detection

```python
from models import create_model
import torch

model = create_model('vca_deit_tiny', task='detection', img_size=512, num_classes=80, num_queries=100)
images = torch.randn(2, 3, 512, 512)
cls_logits, bbox_pred = model(images)
print(f"Classes: {cls_logits.shape}")  # [2, 100, 81]
print(f"Boxes: {bbox_pred.shape}")     # [2, 100, 4]
```

### Detection Features

**Architecture:**
- Query-based detection with 100 detection slots
- Classification head: Predicts 80 object classes + background
- Bounding box head: Regresses normalized coordinates [0, 1]
- Multi-scale training support (480-1333 resolution)

**Loss Function:**
```
Total Loss = L_cls + 5.0 * L_bbox + 2.0 * L_GIoU
```

**Data Augmentation:**
- Multi-scale resizing (480-1333)
- Random horizontal flip
- Image normalization
- Bounding box transform tracking

**Supported Detection Models:**
- vca_deit_tiny: 6.27M parameters
- vca_deit_small: 23.74M parameters

## 📚 References

- DeiT: Data-efficient Image Transformers (https://arxiv.org/abs/2012.12877)
- Vision Transformer (ViT): https://arxiv.org/abs/2010.11929
- Visual Contrast Attention: Linear complexity attention mechanism

## 📝 License

MIT License

## 🤝 Contributing

Feel free to modify and extend this baseline for your research!

---

**Created from scratch without using image_classification library**
