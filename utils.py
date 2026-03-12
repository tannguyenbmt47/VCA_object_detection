"""Utility Functions for Training"""

import os
import torch
import json
import logging
import torch.distributed as dist
from datetime import datetime


# ============================================================================
# Logging Setup
# ============================================================================

def create_logger(output_dir, rank=0, prefix=''):
    """Create logger"""
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if rank == 0:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, f'log_rank{rank}.txt'), mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(output_dir, epoch, model, optimizer, lr_scheduler, 
                   best_acc, logger, name=''):
    """Save checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict() if lr_scheduler and hasattr(lr_scheduler, 'state_dict') else None,
        'best_acc': best_acc,
    }
    
    if name:
        ckpt_path = os.path.join(output_dir, f'ckpt_{name}.pth')
    else:
        ckpt_path = os.path.join(output_dir, f'ckpt_epoch_{epoch:03d}.pth')
    
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")
    
    return ckpt_path


def load_checkpoint(ckpt_path, model, optimizer=None, lr_scheduler=None, logger=None):
    """Load checkpoint"""
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Load model
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
    
    # Load optimizer
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler
    if lr_scheduler and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    if logger:
        logger.info(f"Loaded checkpoint from {ckpt_path} (epoch {epoch})")
    
    return epoch, best_acc


# ============================================================================
# Metrics
# ============================================================================

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self, name='', fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)


def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the top-k predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc)
        return res


# ============================================================================
# Detection Loss Functions
# ============================================================================

def detection_loss(class_logits, bbox_pred, targets, device='cpu', 
                  num_classes=80, alpha=0.25, gamma=2.0):
    """
    Compute detection loss with Hungarian matching
    
    Args:
        class_logits: [B, num_queries, num_classes+1]
        bbox_pred: [B, num_queries, 4] (raw, before sigmoid)
        targets: list of dicts with 'boxes' and 'labels'
        device: device to compute on
        num_classes: number of classes
    
    Returns:
        loss: scalar tensor
        loss_dict: dict of loss components
    """
    B, num_queries = class_logits.shape[:2]
    
    # Apply sigmoid to get predicted boxes in [0, 1] as cxcywh
    pred_boxes_all = bbox_pred.sigmoid()  # [B, num_queries, 4]
    
    # Initialize loss accumulators
    loss_cls_total = torch.tensor(0.0, device=device)
    loss_bbox_total = torch.tensor(0.0, device=device)
    loss_giou_total = torch.tensor(0.0, device=device)
    num_boxes = 0
    
    for i in range(B):
        cls_logit = class_logits[i]  # [num_queries, num_classes+1]
        pred_box = pred_boxes_all[i]  # [num_queries, 4] cxcywh in [0, 1]
        target = targets[i]
        
        target_boxes = target.get('boxes', torch.tensor([], device=device).reshape(0, 4))
        target_labels = target.get('labels', torch.tensor([], device=device, dtype=torch.long))
        if not isinstance(target_boxes, torch.Tensor):
            target_boxes = torch.as_tensor(target_boxes, dtype=torch.float32, device=device)
        if not isinstance(target_labels, torch.Tensor):
            target_labels = torch.as_tensor(target_labels, dtype=torch.long, device=device)
        
        n_targets = len(target_labels)
        
        if n_targets == 0:
            # No targets — all queries should predict background
            cls_loss = torch.nn.functional.cross_entropy(
                cls_logit, 
                torch.full((num_queries,), num_classes, dtype=torch.long, device=device)
            )
            loss_cls_total = loss_cls_total + cls_loss
            continue
        
        # Normalize target boxes to [0, 1] and convert to cxcywh
        img_size = target.get('image_size', None)
        pad = target.get('pad', None)
        if img_size is not None and pad is not None:
            total_h = img_size[0] + pad[0]
            total_w = img_size[1] + pad[1]
        else:
            total_h, total_w = target.get('orig_size', (1, 1))
        
        tgt_boxes_norm = target_boxes / torch.tensor([total_w, total_h, total_w, total_h], dtype=torch.float32, device=device)
        tgt_boxes_norm = tgt_boxes_norm.clamp(0, 1)
        
        # Convert target from xyxy to cxcywh
        tgt_cxcywh = torch.stack([
            (tgt_boxes_norm[:, 0] + tgt_boxes_norm[:, 2]) / 2,
            (tgt_boxes_norm[:, 1] + tgt_boxes_norm[:, 3]) / 2,
            (tgt_boxes_norm[:, 2] - tgt_boxes_norm[:, 0]).clamp(min=0),
            (tgt_boxes_norm[:, 3] - tgt_boxes_norm[:, 1]).clamp(min=0),
        ], dim=-1)  # [n_targets, 4]
        
        # === Hungarian Matching ===
        with torch.no_grad():
            # Cost: classification
            cls_probs = cls_logit.softmax(dim=-1)  # [num_queries, num_classes+1]
            cost_cls = -cls_probs[:, target_labels]  # [num_queries, n_targets]
            
            # Cost: L1 bbox
            cost_bbox = torch.cdist(pred_box, tgt_cxcywh, p=1)  # [num_queries, n_targets]
            
            # Cost: GIoU
            pred_xyxy = _cxcywh_to_xyxy(pred_box)  # [num_queries, 4]
            tgt_xyxy = tgt_boxes_norm  # already xyxy [n_targets, 4]
            cost_giou = -_compute_pairwise_giou(pred_xyxy, tgt_xyxy)  # [num_queries, n_targets]
            
            # Total cost
            C = 1.0 * cost_cls + 5.0 * cost_bbox + 2.0 * cost_giou
            
            # Greedy matching (approximate Hungarian for efficiency)
            indices = _greedy_match(C)
        
        query_idx, tgt_idx = indices
        
        # Classification loss
        targets_cls = torch.full((num_queries,), num_classes, dtype=torch.long, device=device)
        targets_cls[query_idx] = target_labels[tgt_idx]
        cls_loss = torch.nn.functional.cross_entropy(cls_logit, targets_cls)
        loss_cls_total = loss_cls_total + cls_loss
        
        # Bbox losses (only for matched queries)
        matched_pred = pred_box[query_idx]  # [n_matched, 4] cxcywh
        matched_tgt = tgt_cxcywh[tgt_idx]   # [n_matched, 4] cxcywh
        
        # L1 loss
        loss_l1 = torch.nn.functional.l1_loss(matched_pred, matched_tgt)
        loss_bbox_total = loss_bbox_total + loss_l1
        
        # GIoU loss
        matched_pred_xyxy = _cxcywh_to_xyxy(matched_pred).clamp(0, 1)
        matched_tgt_xyxy = tgt_boxes_norm[tgt_idx]
        loss_giou = compute_giou_loss(matched_pred_xyxy, matched_tgt_xyxy)
        loss_giou_total = loss_giou_total + loss_giou
        
        num_boxes += len(query_idx)
    
    # Average losses
    loss_cls = loss_cls_total / max(B, 1)
    loss_bbox = loss_bbox_total / max(num_boxes, 1)
    loss_giou = loss_giou_total / max(num_boxes, 1)
    
    # Total loss
    total_loss = loss_cls + 5.0 * loss_bbox + 2.0 * loss_giou
    
    loss_dict = {
        'loss_cls': loss_cls.item(),
        'loss_bbox': loss_bbox.item(),
        'loss_giou': loss_giou.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict


def _cxcywh_to_xyxy(boxes):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def _compute_pairwise_giou(boxes1, boxes2, eps=1e-7):
    """Compute pairwise GIoU between boxes1 [N, 4] and boxes2 [M, 4] in xyxy format"""
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + eps)
    
    gx1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    gy1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    gx2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    gy2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    g_area = (gx2 - gx1) * (gy2 - gy1)
    
    giou = iou - (g_area - union) / (g_area + eps)
    return giou


def _greedy_match(cost_matrix):
    """Greedy matching: for each target, pick the best unmatched query"""
    num_queries, n_targets = cost_matrix.shape
    n_match = min(num_queries, n_targets)
    
    query_indices = []
    tgt_indices = []
    used_queries = set()
    
    for _ in range(n_match):
        # Mask used queries
        mask = cost_matrix.clone()
        for q in used_queries:
            mask[q, :] = float('inf')
        for t in tgt_indices:
            mask[:, t] = float('inf')
        
        # Find min cost
        flat_idx = mask.argmin()
        q_idx = (flat_idx // n_targets).item()
        t_idx = (flat_idx % n_targets).item()
        
        query_indices.append(q_idx)
        tgt_indices.append(t_idx)
        used_queries.add(q_idx)
    
    return torch.tensor(query_indices, dtype=torch.long, device=cost_matrix.device), \
           torch.tensor(tgt_indices, dtype=torch.long, device=cost_matrix.device)


def compute_giou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Compute GIoU loss
    
    Args:
        pred_boxes: [N, 4] in format [x1, y1, x2, y2] normalized to [0, 1]
        target_boxes: [N, 4] in format [x1, y1, x2, y2] normalized to [0, 1]
    
    Returns:
        loss: scalar tensor
    """
    # Intersection area
    x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
    
    # Pred area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    
    # Target area
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    
    # Union area
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + eps)
    
    # GIoU
    x1_gbox = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_gbox = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_gbox = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_gbox = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    g_area = (x2_gbox - x1_gbox) * (y2_gbox - y1_gbox)
    giou = iou - (g_area - union_area) / (g_area + eps)
    
    # GIoU loss
    loss = 1 - giou
    
    return loss.mean()


# ============================================================================
# Detection Metrics
# ============================================================================

def compute_detection_metrics(pred_class, pred_box, targets, num_classes=80, 
                            iou_threshold=0.5):
    """
    Compute detection metrics (simplified AP, AR)
    
    Args:
        pred_class: [B, num_queries, num_classes+1]
        pred_box: [B, num_queries, 4]
        targets: list of target dicts
        num_classes: number of classes
        iou_threshold: IOU threshold for positive prediction
    
    Returns:
        metrics_dict: dict with 'ap', 'ar', etc.
    
    Note: This is a simplified version. For full COCO evaluation, use pycocotools
    """
    metrics = {
        'ap': 0.0,
        'ar': 0.0,
        'ap50': 0.0,
    }
    
    # This is a placeholder. Full evaluation requires:
    # 1. Post-process predictions (NMS, confidence threshold)
    # 2. Match predictions to targets using Hungarian algorithm
    # 3. Compute AP using COCO-style evaluation
    # 4. Use pycocotools for standard COCO metrics
    
    # For now, return dummy metrics
    return metrics


def get_grad_norm(parameters):
    """Calculate gradient norm"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


# ============================================================================
# Learning Rate & Optimization
# ============================================================================

class CosineLRScheduler:
    """Cosine annealing learning rate scheduler"""
    def __init__(self, optimizer, t_initial, lr_min, warmup_t, warmup_lr_init, 
                 warmup_prefix=False):
        self.optimizer = optimizer
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.cycle_count = 0
        self.t = 0
        self.base_values = [group['lr'] for group in optimizer.param_groups]

    def _get_lr(self, t):
        if t < self.warmup_t:
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * t / self.warmup_t 
                    for base_lr in self.base_values]
        else:
            t_post_warmup = t - self.warmup_t
            t_total = self.t_initial - self.warmup_t
            return [self.lr_min + (base_lr - self.lr_min) * 0.5 * 
                   (1 + math.cos(math.pi * t_post_warmup / t_total)) 
                   for base_lr in self.base_values]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.cycle_count
            self.cycle_count += 1

        lrs = self._get_lr(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

        return lrs

    def step_update(self, num_updates):
        self.t = num_updates / self.t_initial
        lrs = self._get_lr(num_updates)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


import math

def build_optimizer(model, config):
    """Build optimizer"""
    params = model.parameters()
    
    if config.train.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=config.train.base_lr,
            betas=config.train.betas,
            eps=config.train.eps,
            weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=config.train.base_lr,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.train.optimizer}")
    
    return optimizer


def build_scheduler(optimizer, config, num_iters_per_epoch):
    """Build learning rate scheduler"""
    total_steps = config.train.epochs * num_iters_per_epoch
    warmup_steps = config.train.warmup_epochs * num_iters_per_epoch
    
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps,
        lr_min=config.train.min_lr,
        warmup_t=warmup_steps,
        warmup_lr_init=config.train.warmup_lr,
    )
    
    return scheduler


# ============================================================================
# Distributed Training
# ============================================================================

def reduce_tensor(tensor, world_size=1):
    """Reduce tensor across all processes"""
    if world_size == 1:
        return tensor
    
    dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor


def all_reduce_dict(metrics_dict, average=True):
    """All reduce a dictionary of metrics"""
    if not dist.is_available() or not dist.is_initialized():
        return metrics_dict
    
    for k, v in metrics_dict.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            if average:
                v = v / dist.get_world_size()
            metrics_dict[k] = v
    
    return metrics_dict


# ============================================================================
# Config Save
# ============================================================================

def save_config(config, output_dir):
    """Save configuration to JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    config_dict = {
        'data': vars(config.data),
        'model': vars(config.model),
        'aug': vars(config.aug),
        'train': vars(config.train),
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
