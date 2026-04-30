import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import timm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import os
import argparse
import webdataset as wds
import glob

train_tars = glob.glob("/home/yangtao/imagenet-1k/imagenet-1k/imagenet1k-train-*.tar")


def get_model(model_name, pretrained):
    if model_name == "vitb":
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=1000)
        label = f"vitb_{'pretrain' if pretrained else 'random'}"
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        # if pretrained:
        #     # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # else:
        #     # model = models.resnet50(weights=None)
        # # model.fc = torch.nn.Linear(model.fc.in_features, 1000)
        label = f"resnet50_{'pretrain' if pretrained else 'random'}"
    else:
        raise ValueError("Only 'vitb' or 'resnet50' are supported")
    return model, label

class ImageNetCustomDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_csv)
        
        if 'filename' in self.labels_df.columns:
            self.image_files = self.labels_df['filename'].tolist()
            self.labels = self.labels_df['label'].tolist()
        elif 'image_name' in self.labels_df.columns:
            self.image_files = self.labels_df['image_name'].tolist()
            self.labels = self.labels_df['true_label'].tolist()
        else:
            self.image_files = self.labels_df.iloc[:, 0].tolist()
            self.labels = self.labels_df.iloc[:, 1].tolist()
        
        print(f"✅ Dataset initialized successfully, total {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

def print_stage(stage, message=""):
    """Print stage information"""
    print(f"\n{'='*50}")
    print(f"🔄 {stage}")
    if message:
        print(f"   {message}")
    print(f"{'='*50}")

def is_weight_parameter(name):
    """Check if a parameter is a weight parameter (excluding bias)"""
    return "weight" in name.lower() and "bias" not in name.lower()

def compute_weight_stats(model):
    """Compute mean absolute value of weights (weight parameters only)"""
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

def compute_sgd_gradient_stats_accumulated(model, dataloader, criterion, target_batch_size, actual_batch_size, start_batch=10):
    """Compute SGD gradient statistics using gradient accumulation (processes the start_batch-th batch)"""
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
            input_data, target = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            input_data, target = next(dataloader_iter)
    
    print(f"Processing batch {start_batch} (SGD gradient stats)...")
    
    for step in tqdm(range(accumulation_steps), desc=f"Accumulating gradients for batch {start_batch} (SGD)"):
        try:
            input_data, target = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            input_data, target = next(dataloader_iter)
        
        input_data = input_data.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        if step == 0:  # Print details only on the first step
            print(f"Batch {start_batch} Step {step}: loss = {loss.item():.6f}")
        
        # Scale loss
        scaled_loss = loss / accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        total_loss += loss.item()
    
    # Compute gradient statistics
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and is_weight_parameter(name):
            grad_abs_mean = torch.abs(param.grad).mean().item()
            gradient_stats[name] = {
                'grad_abs_mean': grad_abs_mean,
                'shape': list(param.shape),
                'num_params': param.numel()
            }
    
    avg_loss = total_loss / accumulation_steps
    return gradient_stats, avg_loss

def compute_adam_dynamics_stats_accumulated(model, dataloader, criterion, target_batch_size, actual_batch_size, 
                                            exp_avgs, exp_avg_sqs, state_steps, start_batch=10, beta1=0.9, beta2=0.999, eps=1e-8):
    """Compute Adam gradient statistics using gradient accumulation"""
    model.train()
    accumulation_steps = target_batch_size // actual_batch_size
    
    print(f"Computing Adam gradient stats for batch {start_batch} with {target_batch_size} samples...")
    print(f"Accumulation steps: {accumulation_steps}, batch size per step: {actual_batch_size}")
    print("Optimizer: Adam")
    
    # Zero gradients
    model.zero_grad()
    
    total_loss = 0.0
    dataloader_iter = iter(dataloader)
    
    # Skip the first (start_batch - 1) full accumulation cycles
    print(f"Skipping first {start_batch-1} large batches...")
    
    for skip_batch in tqdm(range((start_batch - 1)), desc="Skipping batches"):
        # Zero gradients
        model.zero_grad()
        
        for skip_step in range(accumulation_steps):
            try:
                input_data, target = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                input_data, target = next(dataloader_iter)
    
            input_data = input_data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)
            
            # Forward pass
            output = model(input_data)
            loss = criterion(output, target)
            
            # Scale loss
            scaled_loss = loss / accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
        # Update Adam state (for skipped batches)
        weight_param_index = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and is_weight_parameter(name):
                grad = param.grad
                exp_avg = exp_avgs[weight_param_index]
                exp_avg_sq = exp_avg_sqs[weight_param_index]
                state_steps[weight_param_index] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                
                weight_param_index += 1
    
    print(f"Processing batch {start_batch}...")
    
    # Zero gradients
    model.zero_grad()
    
    for step in tqdm(range(accumulation_steps), desc=f"Accumulating gradients for batch {start_batch}"):
        try:
            input_data, target = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            input_data, target = next(dataloader_iter)

        input_data = input_data.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Scale loss
        scaled_loss = loss / accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        total_loss += loss.item()
    
    # Compute Adam update and collect statistics
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
            
            # Compute Adam update step size
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

def compute_adams_dynamics_stats_accumulated(model, dataloader, criterion, target_batch_size, actual_batch_size, 
                                            exp_avgs, exp_avg_sqs, init_avg_abs_params, state_steps, start_batch=10, beta1=0.9, beta2=0.999, eps=1e-8):
    """Compute AdamS gradient statistics using gradient accumulation"""
    model.train()
    accumulation_steps = target_batch_size // actual_batch_size
    
    print(f"Computing AdamS gradient stats for batch {start_batch} with {target_batch_size} samples...")
    print(f"Accumulation steps: {accumulation_steps}, batch size per step: {actual_batch_size}")
    print("Optimizer: AdamS")
    
    # Zero gradients
    model.zero_grad()
    
    total_loss = 0.0
    dataloader_iter = iter(dataloader)
    
    # Skip the first (start_batch - 1) full accumulation cycles
    print(f"Skipping first {start_batch-1} large batches...")
    
    for skip_batch in tqdm(range((start_batch - 1)), desc="Skipping batches"):
        # Zero gradients
        model.zero_grad()
        
        for skip_step in range(accumulation_steps):
            try:
                input_data, target = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                input_data, target = next(dataloader_iter)
    
            input_data = input_data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)
            
            # Forward pass
            output = model(input_data)
            loss = criterion(output, target)
            
            # Scale loss
            scaled_loss = loss / accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
        # Update Adam state (for skipped batches)
        weight_param_index = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and is_weight_parameter(name):
                grad = param.grad
                exp_avg = exp_avgs[weight_param_index]
                exp_avg_sq = exp_avg_sqs[weight_param_index]
                state_steps[weight_param_index] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                
                weight_param_index += 1
    
    print(f"Processing batch {start_batch}...")
    
    # Zero gradients
    model.zero_grad()
    
    for step in tqdm(range(accumulation_steps), desc=f"Accumulating gradients for batch {start_batch}"):
        try:
            input_data, target = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            input_data, target = next(dataloader_iter)

        input_data = input_data.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Scale loss
        scaled_loss = loss / accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        total_loss += loss.item()
    
    # Compute Adam update and collect statistics
    gradient_stats = {}
    
    weight_param_index = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and is_weight_parameter(name):
            grad = param.grad
            exp_avg = exp_avgs[weight_param_index]
            exp_avg_sq = exp_avg_sqs[weight_param_index]
            init_avg_abs_param = init_avg_abs_params[weight_param_index]
            step = state_steps[weight_param_index]
    
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # Adam update
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            
            # Compute Adam update step size
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

def compute_weight_gradient_ratios(weight_stats, gradient_stats):
    """Compute the ratio between weights and gradients"""
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

def compute_overall_statistics(ratios):
    """Compute overall summary statistics"""
    valid_ratios = []
    valid_params = []
    total_weighted_sum = 0
    total_params = 0
    
    for name, stats in ratios.items():
        ratio = stats['weight_grad_ratio']
        num_params = stats['num_params']
        
        # Only consider finite ratios
        if ratio != float('inf') and not np.isnan(ratio):
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

# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use')
parser.add_argument('--model_name', type=str, choices=['vitb', 'resnet50'], required=True, help='Model name: vitb or resnet50')
parser.add_argument('--pretrained', type=lambda x: x.lower() == 'true', required=True, help='Whether to load pretrained weights: True/False')
parser.add_argument('--loop_num', type=int, default=100, help='Number of sample pairs for NAO analysis')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adams'], default='adam', help='Optimizer type')
args = parser.parse_args()

# Configuration
target_batch_size = 32  # Target large batch size
actual_batch_size = 32  # Actual batch size per step
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure divisibility
if target_batch_size % actual_batch_size != 0:
    print(f"Warning: target_batch_size ({target_batch_size}) is not divisible by actual_batch_size ({actual_batch_size})")
    target_batch_size = (target_batch_size // actual_batch_size) * actual_batch_size
    print(f"Adjusted target_batch_size to: {target_batch_size}")

accumulation_steps = target_batch_size // actual_batch_size

if __name__ == '__main__':
    print_stage("Initializing Model")
    # Load model
    model, model_label = get_model(args.model_name, args.pretrained)
    model.to(device)
    print(f"✅ Model loaded successfully: {model_label}")
    
    # Initialize Adam state (if Adam statistics are needed)
    exp_avgs = []
    exp_avg_sqs = []
    state_steps = []
    init_avg_abs_params = []
    init_avg_abs_param_buffer = 1
    
    if args.optimizer == 'adam' or args.optimizer == 'adams':
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
    
    # Set up loss function and data transforms
    criterion = nn.CrossEntropyLoss()
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Image dataset paths
    # image_dir = "/home/yangtao/duikang/TransferAttack-main/TransferAttack-main/data/images"
    # label_csv = "/home/yangtao/duikang/TransferAttack-main/TransferAttack-main/data/labels.csv"
    
    # print_stage("Loading Dataset")
    # test_dataset = ImageNetCustomDataset(image_dir, label_csv, transform=test_transform)
    # test_loader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False)

    def tuple_transform(img, label):
        img = test_transform(img)
        # label may be bytes, needs to be converted to int
        if isinstance(label, bytes):
            label = int(label.decode())
        else:
            label = int(label)
        return img, label
    
    print_stage("Loading Dataset")
    
    test_dataset = (
        wds.WebDataset(train_tars)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(
            lambda img: test_transform(img),
            lambda label: int(label.decode()) if isinstance(label, bytes) else int(label)
        )
    )
    
    test_loader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False, num_workers=2)



    
    # print(f"✅ Loaded {len(test_dataset)} images")
    
    print(f"Starting weight and gradient analysis ({args.optimizer.upper()} version, weight parameters only, excluding bias)...")
    print(f"Target batch size: {target_batch_size}")
    print(f"Actual batch size: {actual_batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Analysis batch: batch 10")
    print(f"Optimizer: {args.optimizer.upper()}")
    
    # 1. Compute weight statistics
    print("Computing mean absolute values of weights...")
    weight_stats = compute_weight_stats(model)
    print(f"Found {len(weight_stats)} weight parameter layers")
    
    # 2. Compute gradient statistics
    print(f"Computing {args.optimizer.upper()} gradient stats for batch 10...")
    start_batch=10
    if args.optimizer == 'sgd':
        gradient_stats, loss = compute_sgd_gradient_stats_accumulated(
            model, test_loader, criterion, target_batch_size, actual_batch_size, start_batch=start_batch
        )
    elif args.optimizer == 'adam':  # adam
        gradient_stats, loss = compute_adam_dynamics_stats_accumulated(
            model, test_loader, criterion, target_batch_size, actual_batch_size, 
            exp_avgs, exp_avg_sqs, state_steps, start_batch=start_batch
        )
    else:  # adams
        gradient_stats, loss = compute_adams_dynamics_stats_accumulated(
            model, test_loader, criterion, target_batch_size, actual_batch_size, 
            exp_avgs, exp_avg_sqs, init_avg_abs_params, state_steps, start_batch=start_batch
        )
    
    print(f"Computed {args.optimizer.upper()} gradient stats for {len(gradient_stats)} weight parameter layers")
    
    # 3. Compute ratios
    print(f"Computing weight / {args.optimizer.upper()} gradient ratios...")
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
    
    results_file = f"{results_dir}/fei0——{start_batch}_weight_gradient_analysis_{args.optimizer.upper()}_vision_{model_label}_batch{target_batch_size}_batch10.txt"
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Weight vs {args.optimizer.upper()} Gradient Analysis Report (Weight Parameters Only, Vision Task, Batch 10, Gradient Accumulation batch={target_batch_size})\n")
        f.write(f"Model: {model_label}\n")
        f.write(f"Analysis device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Target batch size: {target_batch_size}\n")
        f.write(f"Actual batch size: {actual_batch_size}\n")
        f.write(f"Accumulation steps: {accumulation_steps}\n")
        f.write(f"Analyzed batch: batch 10\n")
        f.write(f"Optimizer: {args.optimizer.upper()}\n")
        f.write("Loss function: CrossEntropyLoss (classification task)\n")
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
        f.write(f"Overall Statistics (Weight Parameters Only, {args.optimizer.upper()} Optimizer, Accumulated batch={target_batch_size}, Batch 10):\n")
        f.write("=" * 80 + "\n")
        f.write(f"Simple mean of weight/{args.optimizer.upper()} gradient ratios across all weight layers: {overall_stats['simple_mean']:.6f}\n")
        f.write(f"Weighted mean of weight/{args.optimizer.upper()} gradient ratios across all weight layers: {overall_stats['weighted_mean']:.6f}\n")
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
            f.write(f"  Mean absolute {args.optimizer.upper()} gradient: {stats['grad_abs_mean']:.6f}\n")
            f.write(f"  Weight/{args.optimizer.upper()} gradient ratio: {stats['weight_grad_ratio']:.6f}\n")
            
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
        if valid_ratios:
            # min_ratio = np.min(valid_ratios)
            min_ratio = np.min(nonzero_ratios)
            max_ratio = np.max(valid_ratios)
            max_min_ratio = max_ratio / min_ratio if min_ratio != 0 else float('inf')
            variance = np.var(valid_ratios)
            
            f.write(f"Total weight layers: {len(ratios)}\n")
            f.write(f"Weight layers with valid ratios: {len(valid_ratios)}\n")
            f.write(f"Ratio median: {np.median(valid_ratios):.6f}\n")
            f.write(f"Ratio std dev: {np.std(valid_ratios):.6f}\n")
            f.write(f"Ratio minimum: {np.min(valid_ratios):.6f}\n")
            f.write(f"Ratio maximum: {np.max(valid_ratios):.6f}\n")
            f.write(f"Max / Min ratio: {max_min_ratio:.6f}\n")
            f.write(f"Ratio variance: {variance:.6f}\n")
            f.write(f"Ratio 25th percentile: {np.percentile(valid_ratios, 25):.6f}\n")
            f.write(f"Ratio 75th percentile: {np.percentile(valid_ratios, 75):.6f}\n")
        
        f.write(f"\nAverage loss over accumulated batch: {loss:.6f}\n")
        f.write(f"Equivalent sample count: {target_batch_size}\n")
        f.write(f"Analyzed batch: batch 10\n")
    
    print(f"\n{args.optimizer.upper()} analysis complete! Results saved to: {results_file}")
    
    # Print key results to console
    print("\n" + "=" * 60)
    print(f"{args.optimizer.upper()} Key Results Summary (Weight Parameters Only, Vision Task, Batch 10, Accumulated batch={target_batch_size}):")
    print("=" * 60)
    print(f"Model: {model_label}")
    print(f"Total model parameters: {total_all_params:,}")
    print(f"Weight parameters: {total_weight_params:,}")
    print(f"Weight parameter ratio: {(total_weight_params / total_all_params * 100):.2f}%")
    print(f"Target batch size: {target_batch_size}")
    print(f"Actual batch size: {actual_batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Analysis batch: batch 10")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Loss function: CrossEntropyLoss (classification task)")
    print(f"Weight/{args.optimizer.upper()} gradient ratio simple mean: {overall_stats['simple_mean']:.6f}")
    print(f"Weight/{args.optimizer.upper()} gradient ratio weighted mean: {overall_stats['weighted_mean']:.6f}")
    print(f"Number of valid weight layers: {overall_stats['total_valid_layers']}")
    print(f"Average loss over accumulated batch: {loss:.6f}")
    print("=" * 60)
