"""Configuration for VCA-DeiT Training"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Data configurations"""
    dataset: str = 'coco'  # 'coco' for object detection
    data_path: str = '/path/to/coco'
    annotation_path: str = None  # Custom annotation dir (default: {data_path}/annotations/)
    img_size: int = 512  # Larger for detection
    batch_size: int = 16  # Smaller for detection
    num_workers: int = 8
    pin_memory: bool = True
    interpolation: str = 'bicubic'
    min_size: int = 480
    max_size: int = 1333


@dataclass
class ModelConfig:
    """Model configurations"""
    model_type: str = 'vca_deit_tiny'
    task: str = 'detection'  # 'classification' or 'detection'
    
    # Architecture
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    
    # Visual Contrast Attention
    vct_num: List[int] = None  # Number of visual contrast tokens per stage
    vct_layer: int = 12  # Which layers use VCA (all if == depth)
    
    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    label_smoothing: float = 0.1
    
    # Detection specific
    num_classes: int = 80  # COCO: 80 classes
    num_queries: int = 100  # Number of detection queries
    num_decoder_layers: int = 6  # Decoder layers for detection
    
    def __post_init__(self):
        if self.vct_num is None:
            self.vct_num = [49, 49, 49, 49]


@dataclass
class AugmentationConfig:
    """Augmentation configurations"""
    color_jitter: float = 0.4
    auto_augment: str = 'rand-m9-mstd0.5-inc1'
    reprob: float = 0.25  # Random erase probability
    remode: str = 'pixel'  # Random erase mode
    recount: int = 1
    
    # Mixup
    mixup: float = 0.8
    cutmix: float = 1.0
    cutmix_minmax: Optional[List[float]] = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = 'batch'


@dataclass
class TrainConfig:
    """Training configurations"""
    epochs: int = 300
    warmup_epochs: int = 20
    start_epoch: int = 0
    
    # Optimizer
    optimizer: str = 'adamw'
    base_lr: float = 5e-4
    warmup_lr: float = 5e-7
    min_lr: float = 5e-6
    weight_decay: float = 0.05
    
    # Optimizer specific
    eps: float = 1e-8
    betas: tuple = (0.9, 0.999)  # For AdamW
    momentum: float = 0.9  # For SGD
    
    # LR Scheduler
    lr_scheduler: str = 'cosine'  # 'cosine', 'linear', 'step'
    
    # Gradient
    clip_grad: float = 5.0
    
    # Checkpointing
    save_freq: int = 1
    print_freq: int = 100
    output_dir: str = 'output'
    
    # Distributed
    distributed: bool = True
    seed: int = 0


@dataclass
class Config:
    """Complete configuration"""
    data: DataConfig
    model: ModelConfig
    aug: AugmentationConfig
    train: TrainConfig
    
    # Misc
    amp: bool = False
    eval_mode: bool = False
    resume_path: Optional[str] = None
    pretrained_path: Optional[str] = None
    
    @classmethod
    def get_default(cls):
        """Get default configuration"""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            aug=AugmentationConfig(),
            train=TrainConfig(),
        )
