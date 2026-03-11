"""Training Script for VCA-DeiT"""

import os
import sys
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from config import Config, DataConfig, ModelConfig, AugmentationConfig, TrainConfig
from models import create_model
from data import build_dataloader, mixup_batch, cutmix_batch
from utils import (
    create_logger, save_checkpoint, load_checkpoint, AverageMeter, 
    accuracy, get_grad_norm, build_optimizer, build_scheduler,
    save_config, reduce_tensor, detection_loss, compute_detection_metrics
)


# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train VCA-DeiT')
    
    # Paths
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet')
    parser.add_argument('--annotation-path', type=str, default='', help='Path to annotation directory (default: {data-path}/annotations/)')
    parser.add_argument('--image-path', type=str, default='', help='Path to train image directory containing train2017/ (default: {data-path}/)')
    parser.add_argument('--val-image-path', type=str, default='', help='Path to val image directory containing val2017/ (default: same as --image-path)')
    parser.add_argument('--max-train-samples', type=int, default=0, help='Max number of training images (0 = use all)')
    parser.add_argument('--max-val-samples', type=int, default=0, help='Max number of validation images (0 = use all)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained weights')
    
    # Model
    parser.add_argument('--model', type=str, default='vca_deit_tiny', help='Model name')
    parser.add_argument('--task', type=str, default='classification', 
                       choices=['classification', 'detection'], help='Task type')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--drop-path', type=float, default=0.1, help='Drop path rate')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=20, help='Warmup epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Base learning rate')
    parser.add_argument('--min-lr', type=float, default=5e-6, help='Min learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    
    # Augmentation
    parser.add_argument('--mixup', type=float, default=0.8, help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=1.0, help='Cutmix alpha')
    
    # Misc
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Use AMP')
    parser.add_argument('--eval', action='store_true', help='Evaluation only')
    parser.add_argument('--fake-data', action='store_true', help='Use fake data for testing')
    parser.add_argument('--print-freq', type=int, default=100, help='Print frequency')
    
    # Distributed
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist-url', type=str, default='env://')
    
    return parser.parse_args()


# ============================================================================
# Main Training Functions
# ============================================================================

def main():
    args = parse_args()
    
    # Setup distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = True
    
    if args.distributed:
        torch.cuda.set_device(args.gpu if hasattr(args, 'gpu') else args.rank % torch.cuda.device_count())
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                               world_size=args.world_size, rank=args.rank)
    
    # Set random seed
    seed = args.seed + args.rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup logger
    logger = create_logger(args.output_dir, rank=args.rank, prefix='VCA-DeiT')
    
    if args.rank == 0:
        logger.info(f"Arguments: {args}")
    
    # Build config
    config = Config(
        data=DataConfig(
            data_path=args.data_path,
            annotation_path=args.annotation_path if args.annotation_path else None,
            image_path=args.image_path if args.image_path else None,
            val_image_path=args.val_image_path if args.val_image_path else None,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
        ),
        model=ModelConfig(
            model_type=args.model,
            drop_path_rate=args.drop_path,
        ),
        aug=AugmentationConfig(
            mixup=args.mixup,
            cutmix=args.cutmix,
        ),
        train=TrainConfig(
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            output_dir=args.output_dir,
            distributed=args.distributed,
        ),
        amp=args.amp,
        eval_mode=args.eval,
        resume_path=args.resume if args.resume else None,
        pretrained_path=args.pretrained if args.pretrained else None,
    )
    
    if args.rank == 0:
        save_config(config, args.output_dir)
    
    # Build model
    if args.rank == 0:
        logger.info(f"Creating model: {args.model} for task: {args.task}")
    
    model = create_model(
        model_type=config.model.model_type,
        task=args.task,
        img_size=config.data.img_size,
        drop_path_rate=config.model.drop_path_rate,
        agent_num=config.model.vct_num,
        num_classes=config.model.num_classes,
    )
    
    model = model.to(args.device)
    
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.rank], find_unused_parameters=False
        )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank == 0:
        logger.info(f"Number of parameters: {n_params / 1e6:.2f}M")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    
    # Build dataloaders
    train_loader, train_dataset = build_dataloader(
        config.data, is_train=True, use_fake=args.fake_data,
        distributed=args.distributed, rank=args.rank, world_size=args.world_size
    )
    
    val_loader, val_dataset = build_dataloader(
        config.data, is_train=False, use_fake=args.fake_data,
        distributed=args.distributed, rank=args.rank, world_size=args.world_size
    )
    
    if args.rank == 0:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")
        logger.info(f"Batch size: {config.data.batch_size}")
        logger.info(f"Num iters per epoch: {len(train_loader)}")
    
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    
    # Loss function
    if args.task == 'detection':
        criterion = None  # Will use custom detection loss
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Setup AMP
    scaler = GradScaler() if config.amp and torch.cuda.is_available() else None
    
    # Resume or load pretrained
    start_epoch = 0
    best_acc = 0.0
    
    if config.resume_path and os.path.exists(config.resume_path):
        start_epoch, best_acc = load_checkpoint(
            config.resume_path, model, optimizer, scheduler, logger
        )
    elif config.pretrained_path and os.path.exists(config.pretrained_path):
        load_checkpoint(config.pretrained_path, model, logger=logger)
    
    # Evaluation only
    if args.eval:
        if args.task == 'detection':
            metrics = validate_detection(val_loader, model, args.device, args.rank)
            if args.rank == 0:
                logger.info(f"Detection Metrics: {metrics}")
        else:
            acc1, acc5, loss = validate(val_loader, model, criterion, args.device, args.rank)
            if args.rank == 0:
                logger.info(f"Top-1 Accuracy: {acc1:.2f}%, Top-5 Accuracy: {acc5:.2f}%")
        return
    
    # Training loop
    if args.rank == 0:
        logger.info("Starting training...")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config.train.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train one epoch
        if args.task == 'detection':
            train_loss, train_meter = train_one_epoch_detection(
                train_loader, model, optimizer, scheduler, 
                epoch, config, args.device, scaler, logger, args.rank,
                args.print_freq
            )
        else:
            train_loss, train_meter = train_one_epoch(
                train_loader, model, criterion, optimizer, scheduler, 
                epoch, config, args.device, scaler, logger, args.rank,
                args.print_freq
            )
        
        # Validation
        if args.task == 'detection':
            val_metrics = validate_detection(val_loader, model, args.device, args.rank)
            val_ap = val_metrics.get('ap', 0.0)
            val_loss = val_metrics.get('val_loss', 0.0)
            
            if args.rank == 0:
                logger.info(
                    f"Epoch {epoch+1}/{config.train.epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val AP@50: {val_ap:.4f}"
                )
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0 or epoch == config.train.epochs - 1:
                    save_checkpoint(
                        args.output_dir, epoch + 1, model, optimizer, scheduler, 
                        val_ap, logger
                    )
                
                # Save best checkpoint
                if val_ap > best_acc:
                    best_acc = val_ap
                    save_checkpoint(
                        args.output_dir, epoch + 1, model, optimizer, scheduler,
                        best_acc, logger, name='best'
                    )
        else:
            val_acc1, val_acc5, val_loss = validate(
                val_loader, model, criterion, args.device, args.rank
            )
            
            if args.rank == 0:
                logger.info(
                    f"Epoch {epoch+1}/{config.train.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_meter:.2f}% - "
                    f"Val Loss: {val_loss:.4f}, Val Acc1: {val_acc1:.2f}%, Val Acc5: {val_acc5:.2f}%"
                )
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0 or epoch == config.train.epochs - 1:
                    save_checkpoint(
                        args.output_dir, epoch + 1, model, optimizer, scheduler, 
                        best_acc, logger
                    )
                
                # Save best checkpoint
                if val_acc1 > best_acc:
                    best_acc = val_acc1
                    save_checkpoint(
                        args.output_dir, epoch + 1, model, optimizer, scheduler,
                        best_acc, logger, name='best'
                    )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    if args.rank == 0:
        logger.info(f"Training time: {total_time_str}")
        logger.info(f"Best Top-1 Accuracy: {best_acc:.2f}%")


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, 
                   epoch, config, device, scaler, logger, rank, print_freq):
    """Train one epoch for classification"""
    model.train()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc@1')
    batch_time = AverageMeter('Time')
    
    end = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixup
        if config.aug.mixup > 0:
            images, targets_a, targets_b, lam = mixup_batch(
                (images, targets), alpha=config.aug.mixup, num_classes=config.model.num_classes
            )
            
            if config.amp and scaler:
                with autocast():
                    outputs = model(images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if config.train.clip_grad:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
                optimizer.zero_grad()
                loss.backward()
                if config.train.clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                optimizer.step()
        else:
            if config.amp and scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if config.train.clip_grad:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                if config.train.clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                optimizer.step()
            
            # For accuracy metric
            targets_a = targets
        
        # Update metrics
        acc = accuracy(outputs, targets_a)[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        
        # Update learning rate
        scheduler.step_update(epoch * len(train_loader) + batch_idx)
        
        # Time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log
        if rank == 0 and (batch_idx + 1) % print_freq == 0:
            logger.info(
                f"Epoch [{epoch+1}][{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"Acc: {acc_meter.val:.2f}% ({acc_meter.avg:.2f}%) "
                f"Time: {batch_time.val:.3f}s"
            )
    
    return loss_meter.avg, acc_meter.avg


def train_one_epoch_detection(train_loader, model, optimizer, scheduler, 
                             epoch, config, device, scaler, logger, rank, print_freq):
    """Train one epoch for object detection"""
    model.train()
    
    loss_meter = AverageMeter('Loss')
    loss_cls_meter = AverageMeter('Loss_cls')
    loss_bbox_meter = AverageMeter('Loss_bbox')
    loss_giou_meter = AverageMeter('Loss_giou')
    batch_time = AverageMeter('Time')
    
    end = time.time()
    
    for batch_idx, (images, targets_list) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        
        # Move targets to device
        for targets in targets_list:
            if 'boxes' in targets:
                if not isinstance(targets['boxes'], torch.Tensor):
                    targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32)
                targets['boxes'] = targets['boxes'].to(device)
            if 'labels' in targets:
                if not isinstance(targets['labels'], torch.Tensor):
                    targets['labels'] = torch.as_tensor(targets['labels'], dtype=torch.int64)
                targets['labels'] = targets['labels'].to(device)
            if 'orig_size' in targets and isinstance(targets['orig_size'], tuple):
                targets['orig_size'] = (targets['orig_size'][0], targets['orig_size'][1])
        
        if config.amp and scaler:
            with autocast():
                class_logits, bbox_pred = model(images)
                loss, loss_dict = detection_loss(
                    class_logits, bbox_pred, targets_list, 
                    device=device, num_classes=config.model.num_classes
                )
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.train.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            class_logits, bbox_pred = model(images)
            loss, loss_dict = detection_loss(
                class_logits, bbox_pred, targets_list, 
                device=device, num_classes=config.model.num_classes
            )
            
            optimizer.zero_grad()
            loss.backward()
            if config.train.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
            optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), images.size(0))
        loss_cls_meter.update(loss_dict['loss_cls'], images.size(0))
        loss_bbox_meter.update(loss_dict['loss_bbox'], images.size(0))
        loss_giou_meter.update(loss_dict['loss_giou'], images.size(0))
        
        # Update learning rate
        scheduler.step_update(epoch * len(train_loader) + batch_idx)
        
        # Time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log
        if rank == 0 and (batch_idx + 1) % print_freq == 0:
            logger.info(
                f"Epoch [{epoch+1}][{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"Loss_cls: {loss_cls_meter.avg:.4f} "
                f"Loss_bbox: {loss_bbox_meter.avg:.4f} "
                f"Loss_giou: {loss_giou_meter.avg:.4f} "
                f"Time: {batch_time.val:.3f}s"
            )
    
    return loss_meter.avg, loss_meter.avg


def _compute_iou(box1, box2):
    """Compute IoU between two sets of boxes [N,4] and [M,4] in xyxy format"""
    x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def _compute_ap(all_pred_boxes, all_pred_scores, all_pred_labels,
                all_gt_boxes, all_gt_labels, iou_threshold=0.5, score_threshold=0.05):
    """Compute mean AP@iou_threshold across all images"""
    # Gather per-class predictions and ground truths
    num_images = len(all_pred_boxes)
    
    # Collect all unique classes from ground truth
    all_classes = set()
    for gt_labels in all_gt_labels:
        all_classes.update(gt_labels.tolist())
    
    if len(all_classes) == 0:
        return 0.0
    
    aps = []
    for cls_id in all_classes:
        # Collect all predictions and GTs for this class
        pred_scores_cls = []
        pred_boxes_cls = []
        pred_img_ids = []
        num_gt_total = 0
        gt_matched = {}  # img_id -> bool array
        
        for img_id in range(num_images):
            # GT for this class
            gt_mask = all_gt_labels[img_id] == cls_id
            n_gt = gt_mask.sum().item()
            num_gt_total += n_gt
            gt_matched[img_id] = torch.zeros(n_gt, dtype=torch.bool)
            
            # Predictions for this class above score threshold
            pred_mask = (all_pred_labels[img_id] == cls_id) & (all_pred_scores[img_id] > score_threshold)
            if pred_mask.any():
                pred_scores_cls.append(all_pred_scores[img_id][pred_mask])
                pred_boxes_cls.append(all_pred_boxes[img_id][pred_mask])
                pred_img_ids.extend([img_id] * pred_mask.sum().item())
        
        if num_gt_total == 0:
            continue
        
        if len(pred_scores_cls) == 0:
            aps.append(0.0)
            continue
        
        pred_scores_cls = torch.cat(pred_scores_cls)
        pred_boxes_cls = torch.cat(pred_boxes_cls)
        
        # Sort by score descending
        sorted_idx = pred_scores_cls.argsort(descending=True)
        pred_scores_cls = pred_scores_cls[sorted_idx]
        pred_boxes_cls = pred_boxes_cls[sorted_idx]
        pred_img_ids = [pred_img_ids[i] for i in sorted_idx.tolist()]
        
        # Compute TP/FP
        tp = torch.zeros(len(pred_scores_cls))
        fp = torch.zeros(len(pred_scores_cls))
        
        for det_idx in range(len(pred_scores_cls)):
            img_id = pred_img_ids[det_idx]
            gt_mask = all_gt_labels[img_id] == cls_id
            gt_boxes = all_gt_boxes[img_id][gt_mask]
            
            if len(gt_boxes) == 0:
                fp[det_idx] = 1
                continue
            
            iou = _compute_iou(pred_boxes_cls[det_idx:det_idx+1], gt_boxes)  # [1, M]
            max_iou, max_idx = iou[0].max(dim=0)
            
            if max_iou >= iou_threshold and not gt_matched[img_id][max_idx]:
                tp[det_idx] = 1
                gt_matched[img_id][max_idx] = True
            else:
                fp[det_idx] = 1
        
        # Compute precision-recall
        tp_cumsum = tp.cumsum(dim=0)
        fp_cumsum = fp.cumsum(dim=0)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
        recall = tp_cumsum / (num_gt_total + 1e-7)
        
        # AP using all-point interpolation
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
        
        # Make precision monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # Find points where recall changes
        change_points = torch.where(recall[1:] != recall[:-1])[0]
        ap = ((recall[change_points + 1] - recall[change_points]) * precision[change_points + 1]).sum().item()
        aps.append(ap)
    
    return sum(aps) / len(aps) if aps else 0.0


def validate_detection(val_loader, model, device, rank):
    """Validation for object detection"""
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    
    with torch.no_grad():
        for images, targets_list in val_loader:
            images = images.to(device, non_blocking=True)
            
            # Move targets to device
            for targets in targets_list:
                if 'boxes' in targets:
                    if not isinstance(targets['boxes'], torch.Tensor):
                        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32)
                    targets['boxes'] = targets['boxes'].to(device)
                if 'labels' in targets:
                    if not isinstance(targets['labels'], torch.Tensor):
                        targets['labels'] = torch.as_tensor(targets['labels'], dtype=torch.int64)
                    targets['labels'] = targets['labels'].to(device)
            
            class_logits, bbox_pred = model(images)
            loss, loss_dict = detection_loss(
                class_logits, bbox_pred, targets_list, device=device
            )
            
            loss_meter.update(loss.item(), images.size(0))
            
            # Collect predictions and ground truths for AP computation
            B = class_logits.shape[0]
            for i in range(B):
                # Predicted scores and labels (exclude background class = last)
                scores = torch.softmax(class_logits[i], dim=-1)[:, :-1]  # [num_queries, num_classes]
                max_scores, pred_cls = scores.max(dim=-1)  # [num_queries]
                
                # Predicted boxes: sigmoid + cxcywh -> xyxy
                pb = bbox_pred[i].sigmoid()
                pred_xyxy = torch.stack([
                    pb[:, 0] - pb[:, 2] / 2,
                    pb[:, 1] - pb[:, 3] / 2,
                    pb[:, 0] + pb[:, 2] / 2,
                    pb[:, 1] + pb[:, 3] / 2,
                ], dim=-1).clamp(0, 1)
                
                all_pred_boxes.append(pred_xyxy.cpu())
                all_pred_scores.append(max_scores.cpu())
                all_pred_labels.append(pred_cls.cpu())
                
                # Ground truth boxes (already scaled by transform)
                gt = targets_list[i]
                gt_boxes = gt.get('boxes', torch.zeros(0, 4))
                gt_labels = gt.get('labels', torch.zeros(0, dtype=torch.long))
                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu()
                    gt_labels = gt_labels.cpu()
                else:
                    gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
                    gt_labels = torch.as_tensor(gt_labels, dtype=torch.int64)
                
                # Normalize gt_boxes to [0,1] same as pred
                img_size = gt.get('image_size', None)
                pad = gt.get('pad', None)
                if img_size is not None and pad is not None:
                    total_h = img_size[0] + pad[0]
                    total_w = img_size[1] + pad[1]
                    gt_boxes = gt_boxes / torch.tensor([total_w, total_h, total_w, total_h], dtype=torch.float32)
                    gt_boxes = gt_boxes.clamp(0, 1)
                
                all_gt_boxes.append(gt_boxes)
                all_gt_labels.append(gt_labels)
    
    # Compute AP@50
    ap50 = _compute_ap(all_pred_boxes, all_pred_scores, all_pred_labels,
                       all_gt_boxes, all_gt_labels, iou_threshold=0.5)
    
    metrics = {
        'ap': ap50,
        'ap50': ap50,
        'val_loss': loss_meter.avg
    }
    
    return metrics


def validate(val_loader, model, criterion, device, rank):
    """Validation for classification"""
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    acc1_meter = AverageMeter('Acc@1')
    acc5_meter = AverageMeter('Acc@5')
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))
    
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    main()
