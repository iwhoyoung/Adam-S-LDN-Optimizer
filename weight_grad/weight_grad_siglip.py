import torch
from torch.serialization import add_safe_globals
import numpy as np
import open_clip
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from tqdm import tqdm
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F  # Add this import
from open_clip import create_model  # Add this import

# def load_siglip_model(model_name, pretrained_path, gpu_id):
#     device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
#     # Check if file exists
#     if not os.path.exists(pretrained_path):
#         print(f"Pretrained weight file not found: {pretrained_path}")
#         return None
#     model, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-B-16-SigLIP-i18n-256',
#         pretrained="/home/yangtao/siglip/ViT-B-16-SigLIP-i18n-256/open_clip_model.safetensors",
#         device=device
#     )
#     return model

# Add configuration at the top
SIGLIP_CONTEXT_LENGTH = 64  # SigLIP sequence length
CLIP_CONTEXT_LENGTH = 77    # CLIP sequence length

def load_model_manually(model_name, model_path, device):
    # Check if it is a CLIP model
    if 'clip' in model_name.lower():
        # Load CLIP architecture without pretrained weights
        # model, _, preprocess = open_clip.create_model_and_transforms(
        #     'ViT-B-32',
        #     pretrained=None,
        #     device=device
        # )
        model, _, preprocess = open_clip.create_model_and_transforms(
            'RN50',              # ResNet-50 as visual encoder
            pretrained=None,     # Do not load pretrained weights
            device=device
        )
    elif 'siglip' in model_name.lower():
        # Load SigLIP architecture without pretrained weights
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16-SigLIP-i18n-256',
            pretrained=None,
            device=device
        )
    # safetensors_path = os.path.join(model_path, "open_clip_model.safetensors")
    # pytorch_path = os.path.join(model_path, "open_clip_pytorch_model.bin")
    # if os.path.exists(safetensors_path):
    #     from safetensors.torch import load_file
    #     state_dict = load_file(safetensors_path)
    # elif os.path.exists(pytorch_path):
    #     state_dict = torch.load(pytorch_path, map_location=device)
    # else:
    #     raise FileNotFoundError("Weight file not found")
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # print(f"missing_keys: {missing_keys[:5]}")
    # print(f"unexpected_keys: {unexpected_keys[:5]}")
    # print("✅ Manually loaded weights successfully")
    return model, preprocess


# Add loss function definitions
def contrastive_loss(logits):
    """CLIP contrastive loss"""
    targets = torch.arange(logits.size(0)).to(logits.device)
    loss_images = F.cross_entropy(logits, targets)
    loss_texts = F.cross_entropy(logits.t(), targets)
    return (loss_images + loss_texts) / 2

def siglip_loss(logits):
    """SigLIP loss"""
    n = logits.size(0)
    # -1 for off-diagonals and 1 for diagonals
    labels = 2 * torch.eye(n, device=logits.device) - 1
    # pairwise sigmoid loss
    return -torch.sum(F.logsigmoid(labels * logits)) / n

def compute_loss(image_features, text_features, loss_type='siglip', temperature=0.07):
    """
    Compute loss function
    Args:
        image_features: image features [batch_size, feature_dim]
        text_features: text features [batch_size, feature_dim]
        loss_type: loss type, 'siglip' or 'clip'
        temperature: temperature parameter (used for CLIP only)
    """
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix
    logits = image_features @ text_features.t()
    
    if loss_type == 'clip':
        # CLIP uses temperature scaling
        logits = logits / temperature
        return contrastive_loss(logits)
    elif loss_type == 'siglip':
        # SigLIP does not use temperature scaling
        return siglip_loss(logits)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

# COCO dataset class
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, images_path, annotations_path, transform=None, max_samples=None):
        self.images_path = images_path
        self.transform = transform
        
        # Load COCO annotations
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        self.annotations = coco_data['annotations']
        self.images = {img['id']: img for img in coco_data['images']}
        
        # Limit number of samples
        if max_samples:
            self.annotations = self.annotations[:max_samples]
        
        print(f"Loaded {len(self.annotations)} image-text pairs")
        self.default_size = 256
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_info = self.images[ann['image_id']]
        
        # Load image
        image_path = os.path.join(self.images_path, image_info['file_name'])
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # If image loading fails, return a default black image
            print(f"Warning: Failed to load image {image_path}: {e}")
            if hasattr(self, 'default_size'):
                image = torch.zeros(3, self.default_size, self.default_size)
            else:
                image = torch.zeros(3, 256, 256)  # Default size
        
        # Get text caption
        caption = ann['caption']
        
        return image, caption


def image_transform(image_size, is_train=False):
    """Image transformation"""
    # Handle the case where image_size may be a tuple or an integer
    if isinstance(image_size, (tuple, list)):
        size = image_size
    else:
        size = (image_size, image_size)
    
    if is_train:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def print_stage(stage_name, details=""):
    """Print stage information"""
    print(f"\n{'='*60}")
    print(f"🔄 {stage_name}")
    if details:
        print(f"   {details}")
    print(f"{'='*60}")

# Configuration
# model_path = "checkpoints/lr5e-03_sgd/final_model_lr5e-03_sgd.pth"  # SigLIP model path
target_batch_size = 32  # Target large batch size
actual_batch_size = 32  # Actual batch size per step
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure divisibility
if target_batch_size % actual_batch_size != 0:
    print(f"Warning: target_batch_size ({target_batch_size}) is not divisible by actual_batch_size ({actual_batch_size})")
    target_batch_size = (target_batch_size // actual_batch_size) * actual_batch_size
    print(f"Adjusted target_batch_size to: {target_batch_size}")

accumulation_steps = target_batch_size // actual_batch_size

# Function to check if a parameter is a weight parameter
def is_weight_parameter(name):
    """
    Check if a parameter is a weight parameter (excluding bias)
    """
    # Only consider parameters whose name contains "weight", excluding "bias"
    return "weight" in name.lower() and "bias" not in name.lower()

# Function to compute mean absolute value of weights (weight parameters only)
def compute_weight_stats(model):
    weight_stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None and is_weight_parameter(name):
            weight_abs_mean = torch.abs(param.data).mean().item()
            weight_stats[name] = {
                'weight_abs_mean': weight_abs_mean,
                'shape': list(param.shape),
                'num_params': param.numel()
            }
    
    return weight_stats

# Function to compute SGD gradient statistics using gradient accumulation (processes the start_batch-th batch)
def compute_sgd_gradient_stats_accumulated(model, tokenizer, dataloader, target_batch_size, actual_batch_size, start_batch=10):
    model.train()
    accumulation_steps = target_batch_size // actual_batch_size
    
    print(f"Computing SGD gradient stats for batch {start_batch} with {target_batch_size} samples...")
    print(f"Accumulation steps: {accumulation_steps}, batch size per step: {actual_batch_size}")
    print("Optimizer: SGD (using raw gradients)")
    
    # Zero gradients
    model.zero_grad()
    
    total_loss = 0.0
    dataloader_iter = iter(dataloader)
    
    # Skip the first (start_batch - 1) full accumulation cycles
    skip_steps = (start_batch - 1) * accumulation_steps
    print(f"Skipping first {skip_steps} mini-batches (first {start_batch-1} large batches)...")
    
    for skip_step in tqdm(range(skip_steps), desc="Skipping batches"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
    
    print(f"Processing batch {start_batch} (SGD gradient stats)...")
    
    for step in tqdm(range(accumulation_steps), desc=f"Accumulating gradients for batch {start_batch} (SGD)"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        images, captions = batch
        images = images.to(device)
        
        # Tokenize text
        text_tokens = tokenizer(captions).to(device)
        
        # Print batch info
        if step == 0:  # Print details only on the first step
            print(f"Batch {start_batch} Step {step}: images shape = {images.shape}")
            print(f"Batch {start_batch} Step {step}: text_tokens shape = {text_tokens.shape}")
        
        # Forward pass - compute loss
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)
        
        # Compute loss (automatically select SigLIP or CLIP loss)
        # Determine which loss to use based on model name
        if 'siglip' in args.model_name.lower():
            loss = compute_loss(image_features, text_features, loss_type='siglip')
            loss_name = "SigLIP"
        else:
            loss = compute_loss(image_features, text_features, loss_type='clip', temperature=0.07)
            loss_name = "CLIP"
        
        if step == 0:
            print(f"Batch {start_batch} Step {step}: Contrastive loss = {loss.item():.6f}")
        
        # Scale loss (important: simulate average loss of large batch)
        scaled_loss = loss / accumulation_steps
        
        # Backward pass (gradients accumulate; raw gradient values are accumulated, not absolute values)
        scaled_loss.backward()
        
        total_loss += loss.item()
    
    # At this point, model gradients have accumulated over target_batch_size samples from batch start_batch
    # For SGD, directly use the accumulated gradients (this is the SGD update direction)
    # Compute mean absolute gradient values here (weight parameters only)
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and is_weight_parameter(name):
            # SGD uses raw accumulated gradients; take absolute value and compute mean here
            grad_abs_mean = torch.abs(param.grad).mean().item()
            gradient_stats[name] = {
                'grad_abs_mean': grad_abs_mean,
                'shape': list(param.shape),
                'num_params': param.numel()
            }
    
    avg_loss = total_loss / accumulation_steps
    return gradient_stats, avg_loss

# Function to compute Adam gradient statistics using gradient accumulation
def compute_adam_dynamics_stats_accumulated(model, tokenizer, dataloader, target_batch_size, actual_batch_size, 
                                            exp_avgs, exp_avg_sqs, state_steps, start_batch=10, beta1=0.9, beta2=0.999, eps=1e-8):
    model.train()
    accumulation_steps = target_batch_size // actual_batch_size
    
    print(f"Computing Adam gradient stats for batch {start_batch} with {target_batch_size} samples...")
    print(f"Accumulation steps: {accumulation_steps}, batch size per step: {actual_batch_size}")
    print("Optimizer: Adam")
    
    # Zero gradients
    model.zero_grad()
    
    total_loss = 0.0
    dataloader_iter = iter(dataloader)
    
    # Skip the first (start_batch - 1) full accumulation cycles and update Adam state
    skip_steps = (start_batch - 1) * accumulation_steps
    print(f"Skipping first {skip_steps} mini-batches (first {start_batch-1} large batches)...")
    
    for skip_batch in tqdm(range((start_batch - 1)), desc="Skipping batches"):
        # Zero gradients
        model.zero_grad()
        for skip_step in range(accumulation_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
    
            images, captions = batch
            images = images.to(device)
            text_tokens = tokenizer(captions).to(device)
            
            # Forward pass - compute loss
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)
            
            # Compute loss (automatically select SigLIP or CLIP loss)
            # Determine which loss to use based on model name
            if 'siglip' in args.model_name.lower():
                loss = compute_loss(image_features, text_features, loss_type='siglip')
                loss_name = "SigLIP"
            else:
                loss = compute_loss(image_features, text_features, loss_type='clip', temperature=0.07)
                loss_name = "CLIP"
            
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
        # Update Adam state
        weight_param_index = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and is_weight_parameter(name):
                grad = param.grad
                exp_avg = exp_avgs[weight_param_index]
                exp_avg_sq = exp_avg_sqs[weight_param_index]
                # step = state_steps[weight_param_index]
                state_steps[weight_param_index] += 1
                
                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                
                weight_param_index += 1
    
    print(f"Processing batch {start_batch} (Adam gradient stats)...")
    
    for step in tqdm(range(accumulation_steps), desc=f"Accumulating gradients for batch {start_batch} (Adam)"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        images, captions = batch
        images = images.to(device)
        text_tokens = tokenizer(captions).to(device)
        
        if step == 0:
            print(f"Batch {start_batch} Step {step}: images shape = {images.shape}")
            print(f"Batch {start_batch} Step {step}: text_tokens shape = {text_tokens.shape}")
        
        # Forward pass
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)
        
        # Compute loss (automatically select SigLIP or CLIP loss)
        if 'siglip' in args.model_name.lower():
            loss = compute_loss(image_features, text_features, loss_type='siglip')
        else:
            loss = compute_loss(image_features, text_features, loss_type='clip', temperature=0.07)
        
        if step == 0:
            print(f"Batch {start_batch} Step {step}: Contrastive loss = {loss.item():.6f}")
        
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        
        total_loss += loss.item()
    
    # Compute mean absolute value of Adam update step sizes
    gradient_stats = {}
    
    weight_param_index = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and is_weight_parameter(name):
            grad = param.grad
            exp_avg = exp_avgs[weight_param_index]
            exp_avg_sq = exp_avg_sqs[weight_param_index]
            step = state_steps[weight_param_index]
    
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # Adam update
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            
            # Compute Adam update step size, then take mean absolute value
            adam_update = (exp_avg / bias_correction1) / denom
            grad_abs_mean = torch.abs(adam_update).mean().item()
            
            gradient_stats[name] = {
                'grad_abs_mean': grad_abs_mean,
                'shape': list(param.shape),
                'num_params': param.numel()
            }
            
            weight_param_index += 1
    
    avg_loss = total_loss / accumulation_steps
    return gradient_stats, avg_loss

# Function to compute AdamS gradient statistics using gradient accumulation
def compute_adams_dynamics_stats_accumulated(model, tokenizer, dataloader, target_batch_size, actual_batch_size, 
                                            exp_avgs, exp_avg_sqs, init_avg_abs_params, state_steps, start_batch=10, beta1=0.9, beta2=0.999, eps=1e-8):
    model.train()
    accumulation_steps = target_batch_size // actual_batch_size
    
    print(f"Computing AdamS gradient stats for batch {start_batch} with {target_batch_size} samples...")
    print(f"Accumulation steps: {accumulation_steps}, batch size per step: {actual_batch_size}")
    print("Optimizer: AdamS")
    
    # Zero gradients
    model.zero_grad()
    
    total_loss = 0.0
    dataloader_iter = iter(dataloader)
    
    # Skip the first (start_batch - 1) full accumulation cycles and update Adam state
    skip_steps = (start_batch - 1) * accumulation_steps
    print(f"Skipping first {skip_steps} mini-batches (first {start_batch-1} large batches)...")
    
    for skip_batch in tqdm(range((start_batch - 1)), desc="Skipping batches"):
        # Zero gradients
        model.zero_grad()
        for skip_step in range(accumulation_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
    
            images, captions = batch
            images = images.to(device)
            text_tokens = tokenizer(captions).to(device)
            
            # Forward pass - compute loss
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)
            
            # Compute loss (automatically select SigLIP or CLIP loss)
            # Determine which loss to use based on model name
            if 'siglip' in args.model_name.lower():
                loss = compute_loss(image_features, text_features, loss_type='siglip')
                loss_name = "SigLIP"
            else:
                loss = compute_loss(image_features, text_features, loss_type='clip', temperature=0.07)
                loss_name = "CLIP"
            
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
        # Update Adam state
        weight_param_index = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and is_weight_parameter(name):
                grad = param.grad
                exp_avg = exp_avgs[weight_param_index]
                exp_avg_sq = exp_avg_sqs[weight_param_index]
                # step = state_steps[weight_param_index]
                state_steps[weight_param_index] += 1
                
                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                
                weight_param_index += 1
    
    print(f"Processing batch {start_batch} (AdamS gradient stats)...")
    
    for step in tqdm(range(accumulation_steps), desc=f"Accumulating gradients for batch {start_batch} (AdamS)"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        images, captions = batch
        images = images.to(device)
        text_tokens = tokenizer(captions).to(device)
        
        if step == 0:
            print(f"Batch {start_batch} Step {step}: images shape = {images.shape}")
            print(f"Batch {start_batch} Step {step}: text_tokens shape = {text_tokens.shape}")
        
        # Forward pass
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)
        
        # Compute loss (automatically select SigLIP or CLIP loss)
        if 'siglip' in args.model_name.lower():
            loss = compute_loss(image_features, text_features, loss_type='siglip')
        else:
            loss = compute_loss(image_features, text_features, loss_type='clip', temperature=0.07)
        
        if step == 0:
            print(f"Batch {start_batch} Step {step}: Contrastive loss = {loss.item():.6f}")
        
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        
        total_loss += loss.item()
    
    # Compute mean absolute value of Adam update step sizes
    gradient_stats = {}
    
    weight_param_index = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and is_weight_parameter(name):
            grad = param.grad
            exp_avg = exp_avgs[weight_param_index]
            exp_avg_sq = exp_avg_sqs[weight_param_index]
            step = state_steps[weight_param_index]
            init_avg_abs_param = init_avg_abs_params[weight_param_index]
    
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # Adam update
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            
            # Compute Adam update step size, then take mean absolute value
            adam_update = (exp_avg / bias_correction1) / denom
            grad_abs_mean = (torch.abs(adam_update).mean()*init_avg_abs_param).item()
            
            gradient_stats[name] = {
                'grad_abs_mean': grad_abs_mean,
                'shape': list(param.shape),
                'num_params': param.numel()
            }
            
            weight_param_index += 1
    
    avg_loss = total_loss / accumulation_steps
    return gradient_stats, avg_loss

# Function to compute weight-to-gradient ratios
def compute_weight_gradient_ratios(weight_stats, gradient_stats):
    ratios = {}
    
    for name in weight_stats.keys():
        if name in gradient_stats:
            weight_mean = weight_stats[name]['weight_abs_mean']
            grad_mean = gradient_stats[name]['grad_abs_mean']
            
            # Avoid division by zero
            if grad_mean != 0:
                ratio = weight_mean / grad_mean
            else:
                ratio = float('inf')
            
            ratios[name] = {
                'weight_abs_mean': weight_mean,
                'grad_abs_mean': grad_mean,
                'weight_grad_ratio': ratio,
                'shape': weight_stats[name]['shape'],
                'num_params': weight_stats[name]['num_params']
            }
    
    return ratios

# Function to compute overall summary statistics
def compute_overall_statistics(ratios):
    """
    Compute overall summary statistics
    """
    valid_ratios = []
    valid_params = []
    total_weighted_sum = 0
    total_params = 0
    
    for name, stats in ratios.items():
        ratio = stats['weight_grad_ratio']
        num_params = stats['num_params']
        
        # Only consider finite ratios
        if ratio != float('inf') and not np.isnan(ratio) and ratio > 1e-6:
            valid_ratios.append(ratio)
            valid_params.append(num_params)
            
            # Compute weighted sum
            total_weighted_sum += ratio * num_params
            total_params += num_params
    
    # 1. Simple mean: arithmetic mean of all layer ratios
    simple_mean = np.mean(valid_ratios) if valid_ratios else 0
    
    # 2. Weighted mean: mean weighted by number of parameters
    weighted_mean = total_weighted_sum / total_params if total_params > 0 else 0
    
    return {
        'simple_mean': simple_mean,
        'weighted_mean': weighted_mean,
        'total_valid_layers': len(valid_ratios),
        'total_params': total_params,
        'valid_ratios': valid_ratios,
        'valid_params': valid_params
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--model_name', type=str, default="ViT-B-16-SigLIP-i18n-256", help='SigLIP model name')
    parser.add_argument('--pretrained_path', type=str, default="/home/yangtao/siglip/ViT-B-16-SigLIP-i18n-256/open_clip_pytorch_model.bin", help='Path to pretrained weights')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of data samples to use')
    parser.add_argument('--start_batch', type=int, default=10, help='Batch index to start analysis from')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adams'], default='adams', help='Optimizer type')
    args = parser.parse_args()

    if 'siglip' in args.model_name.lower():
        loss_type = 'siglip'
        loss_name = 'SigLIP'
        print(f"✅ Detected SigLIP model, using SigLIP loss function")
    else:
        loss_type = 'clip'
        loss_name = 'CLIP'
        print(f"✅ Detected CLIP model, using CLIP contrastive loss function")

    print_stage("Program starting", f"Model: SigLIP")

    # Load SigLIP model
    print("Loading SigLIP model...")
    model, preprocess = load_model_manually(loss_type, "/home/yangtao/siglip/ViT-B-16-SigLIP-i18n-256", device)
    
    model.to(device)
    print("Model loaded successfully")

    # Get tokenizer
    # _, _, preprocess = open_clip.create_model_and_transforms(args.model_name)
    if 'clip' in args.model_name.lower():
        tokenizer = open_clip.get_tokenizer(args.model_name)
    else:
    ### siglip tokenizer
        tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP-i18n-256')
    
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("/home/yangtao/siglip/ViT-B-16-SigLIP-i18n-256")
    
    # Get image transform
    try:
        image_size = model.visual.image_size
        transform = image_transform(image_size, is_train=False)
        if isinstance(image_size, (tuple, list)):
            print(f"✅ Using model image size: {image_size[0]}x{image_size[1]}")
        else:
            print(f"✅ Using model image size: {image_size}")
    except Exception as e:
        print(f"Failed to get model image size: {e}")
        transform = image_transform(256, is_train=False)
        print("✅ Using default image size: 256")
    
    # Data paths
    images_path = "/home/yangtao/train2017/train2017"
    annotations_path = "/home/yangtao/annotations/annotations/captions_train2017.json"
    
    print_stage("Loading COCO Dataset")
    dataset = COCODataset(images_path, annotations_path, transform=transform, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, num_workers=4)
    print(f"✅ Loaded {len(dataset)} images")

    # Initialize Adam state (if Adam statistics are needed)
    exp_avgs = []
    exp_avg_sqs = []
    state_steps = []
    init_avg_abs_params = []
    init_avg_abs_param_buffer = 1
    
    for name, param in model.named_parameters():
        if param.requires_grad and is_weight_parameter(name):
            exp_avgs.append(torch.zeros_like(param.data))
            exp_avg_sqs.append(torch.zeros_like(param.data))
            init_avg_abs_param = torch.mean(torch.abs(param.data))
            if init_avg_abs_param < 1e-6:
                init_avg_abs_param = 0.1*init_avg_abs_param_buffer*torch.ones_like(init_avg_abs_param, memory_format=torch.preserve_format)
            init_avg_abs_param_buffer = init_avg_abs_param
            init_avg_abs_params.append(init_avg_abs_param_buffer)
            state_steps.append(1)  # Adam step starts from 1

    # Test dataloader
    print("Testing dataloader...")
    test_batch = next(iter(dataloader))
    images, captions = test_batch
    print(f"Test batch - images shape: {images.shape}")
    print(f"Test batch - captions: {captions[:2]}")  # Show first 2 captions

    print("Starting weight and gradient analysis (SigLIP version, weight parameters only, excluding bias)...")
    print(f"Target batch size: {target_batch_size}")
    print(f"Actual batch size: {actual_batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Analysis batch: batch {args.start_batch}")
    print("Optimizer: SGD/Adam")
    print("Note: Using contrastive learning loss; raw gradient values are accumulated during gradient accumulation, absolute mean is computed at the end")
    
    # 1. Compute weight statistics
    print("Computing mean absolute values of weights...")
    weight_stats = compute_weight_stats(model)
    print(f"Found {len(weight_stats)} weight parameter layers")
    
    # 2. Compute gradient statistics (SGD or Adam selectable)
    # use_adam = False  # Set to True to use Adam, False to use SGD
    
    if args.optimizer == 'adam':
        print(f"Computing Adam gradient stats for batch {args.start_batch}...")
        gradient_stats, loss = compute_adam_dynamics_stats_accumulated(
            model, tokenizer, dataloader, target_batch_size, actual_batch_size, 
            exp_avgs, exp_avg_sqs, state_steps, start_batch=args.start_batch
        )
        optimizer_name = "Adam"
    elif args.optimizer == 'sgd':
        print(f"Computing SGD gradient stats for batch {args.start_batch}...")
        gradient_stats, loss = compute_sgd_gradient_stats_accumulated(
            model, tokenizer, dataloader, target_batch_size, actual_batch_size, start_batch=args.start_batch
        )
        optimizer_name = "SGD"
    else:
        print(f"Computing AdamS gradient stats for batch {args.start_batch}...")
        gradient_stats, loss = compute_adams_dynamics_stats_accumulated(
            model, tokenizer, dataloader, target_batch_size, actual_batch_size, 
            exp_avgs, exp_avg_sqs, init_avg_abs_params, state_steps, start_batch=args.start_batch
        )
        optimizer_name = "AdamS"

    print(f"Computed {optimizer_name} gradient stats for {len(gradient_stats)} weight parameter layers")
    
    # 3. Compute ratios
    print(f"Computing weight / {optimizer_name} gradient ratios...")
    ratios = compute_weight_gradient_ratios(weight_stats, gradient_stats)
    
    # 4. Compute overall statistics
    print("Computing overall summary statistics...")
    overall_stats = compute_overall_statistics(ratios)
    
    # Count all parameters (for comparison)
    total_all_params = sum(p.numel() for p in model.parameters())
    total_weight_params = sum(p.numel() for name, p in model.named_parameters() if is_weight_parameter(name))
    
    # Save results to text file
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    model_name_short = args.model_name.replace('/', '_')
    results_file = f"{results_dir}/fei0{args.start_batch}_weight_gradient_analysis_{optimizer_name}_weights_only_contrastive_accumulated_batch{target_batch_size}_batch{args.start_batch}_{model_name_short}.txt"
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SigLIP Weight vs {optimizer_name} Gradient Analysis Report (Weight Parameters Only, Contrastive Learning Loss, Batch {args.start_batch}, Gradient Accumulation batch={target_batch_size})\n")
        f.write(f"Model name: {args.model_name}\n")
        f.write(f"Pretrained weights: {args.pretrained_path}\n")
        f.write(f"Analysis device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Target batch size: {target_batch_size}\n")
        f.write(f"Actual batch size: {actual_batch_size}\n")
        f.write(f"Accumulation steps: {accumulation_steps}\n")
        f.write(f"Analyzed batch: batch {args.start_batch}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write("Loss function: Contrastive Learning Loss\n")
        f.write("Gradient accumulation method: raw gradient values accumulated, absolute mean computed at the end\n")
        f.write("=" * 80 + "\n\n")
        
        # Parameter statistics
        f.write("Parameter Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total model parameters: {total_all_params:,}\n")
        f.write(f"Weight parameters: {total_weight_params:,}\n")
        f.write(f"Bias parameters: {total_all_params - total_weight_params:,}\n")
        f.write(f"Weight parameter ratio: {(total_weight_params / total_all_params * 100):.2f}%\n")
        f.write("\n")
        
        # Overall statistics
        f.write(f"Overall Statistics (Weight Parameters Only, {optimizer_name} Optimizer, Contrastive Learning Loss, Accumulated batch={target_batch_size}, Batch {args.start_batch}):\n")
        f.write("=" * 80 + "\n")
        f.write(f"Simple mean of weight/{optimizer_name} gradient ratios across all weight layers: {overall_stats['simple_mean']:.6f}\n")
        f.write(f"Weighted mean of weight/{optimizer_name} gradient ratios across all weight layers: {overall_stats['weighted_mean']:.6f}\n")
        f.write(f"  (Weighted formula: Σ(ratio × num_weight_params) / Σ(num_weight_params))\n")
        f.write(f"Number of valid weight layers: {overall_stats['total_valid_layers']}\n")
        f.write(f"Number of analyzed weight parameters: {overall_stats['total_params']:,}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Per-layer Details:\n")
        f.write("-" * 80 + "\n")
        
        # Sort layers
        sorted_layers = sorted(ratios.items())
        
        for name, stats in sorted_layers:
            f.write(f"Layer name: {name}\n")
            f.write(f"  Shape: {stats['shape']}\n")
            f.write(f"  Number of parameters: {stats['num_params']:,}\n")
            f.write(f"  Mean absolute weight: {stats['weight_abs_mean']:.6f}\n")
            f.write(f"  Mean absolute {optimizer_name} gradient: {stats['grad_abs_mean']:.6f}\n")
            f.write(f"  Weight/{optimizer_name} gradient ratio: {stats['weight_grad_ratio']:.6f}\n")
            
            # Compute this layer's contribution to the weighted mean
            if stats['weight_grad_ratio'] != float('inf') and not np.isnan(stats['weight_grad_ratio']):
                contribution = (stats['weight_grad_ratio'] * stats['num_params']) / overall_stats['total_params']
                f.write(f"  Contribution to weighted mean: {contribution:.6f}\n")
            
            f.write("-" * 40 + "\n")
        
        # Detailed summary statistics
        f.write("\nDetailed Summary Statistics:\n")
        f.write("-" * 80 + "\n")
        
        valid_ratios = overall_stats['valid_ratios']
        nonzero_ratios = [r for r in valid_ratios if r != 0]  # Keep only non-zero ratios
        
        f.write(f"Total weight layers: {len(ratios)}\n")
        f.write(f"Weight layers with valid ratios: {len(valid_ratios)}\n")
        f.write(f"Weight layers with valid non-zero ratios: {len(nonzero_ratios)}\n")
        
        if nonzero_ratios:
            min_ratio = np.min(nonzero_ratios)
            max_ratio = np.max(nonzero_ratios)
            max_min_ratio = max_ratio / min_ratio if min_ratio != 0 else float('inf')
            variance = np.var(nonzero_ratios)
            f.write(f"Non-zero ratio median: {np.median(nonzero_ratios):.6f}\n")
            f.write(f"Non-zero ratio std dev: {np.std(nonzero_ratios):.6f}\n")
            f.write(f"Non-zero ratio minimum: {min_ratio:.6f}\n")
            f.write(f"Non-zero ratio maximum: {max_ratio:.6f}\n")
            f.write(f"Max / Min ratio: {max_min_ratio:.6f}\n")
            f.write(f"Non-zero ratio variance: {variance:.6f}\n")
            f.write(f"Non-zero ratio 25th percentile: {np.percentile(nonzero_ratios, 25):.6f}\n")
            f.write(f"Non-zero ratio 75th percentile: {np.percentile(nonzero_ratios, 75):.6f}\n")
        else:
            f.write("No non-zero ratio layers found!\n")

        f.write(f"\nAverage contrastive loss over accumulated batch: {loss:.6f}\n")
        f.write(f"Equivalent sample count: {target_batch_size}\n")
        f.write(f"Analyzed batch: batch {args.start_batch}\n")
        f.write(f"Dataset: COCO (image-text pairs)\n")
        f.write(f"Samples used: {args.max_samples}\n")
    
    print(f"\nSigLIP {optimizer_name} analysis complete! Results saved to: {results_file}")
    
    # Print key results to console
    print("\n" + "=" * 60)
    print(f"SigLIP {optimizer_name} Key Results Summary (Weight Parameters Only, Contrastive Learning Loss, Batch {args.start_batch}, Accumulated batch={target_batch_size}):")
    print("=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Total model parameters: {total_all_params:,}")
    print(f"Weight parameters: {total_weight_params:,}")
    print(f"Weight parameter ratio: {(total_weight_params / total_all_params * 100):.2f}%")
    print(f"Target batch size: {target_batch_size}")
    print(f"Actual batch size: {actual_batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Analysis batch: batch {args.start_batch}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Loss function: Contrastive Learning Loss")
    print(f"Weight/{optimizer_name} gradient ratio simple mean: {overall_stats['simple_mean']:.6f}")
    print(f"Weight/{optimizer_name} gradient ratio weighted mean: {overall_stats['weighted_mean']:.6f}")
    print(f"Number of valid weight layers: {overall_stats['total_valid_layers']}")
    print(f"Average contrastive loss over accumulated batch: {loss:.6f}")
    print(f"Dataset: COCO ({args.max_samples} samples used)")
    print("=" * 60)
