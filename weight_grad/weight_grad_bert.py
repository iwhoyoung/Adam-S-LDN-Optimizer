import os
import math
import json

import torch
import numpy as np
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
)
from transformers.models.bert.configuration_bert import BertConfig
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

# Allow BertConfig to be deserialized from checkpoints safely
add_safe_globals([BertConfig])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

model_path = "checkpoints/lr5e-03_sgd/final_model_lr5e-03_sgd.pth"
file_paths = [
    "/home/yangtao/Salesforce___wikitext/wikitext-103-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train-00001-of-00002.arrow",
    "/home/yangtao/Salesforce___wikitext/wikitext-103-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train-00000-of-00002.arrow",
]

target_batch_size = 32   # Logical (effective) batch size
actual_batch_size  = 32  # Physical mini-batch size per forward pass
block_size         = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure target_batch_size is divisible by actual_batch_size
if target_batch_size % actual_batch_size != 0:
    target_batch_size = (target_batch_size // actual_batch_size) * actual_batch_size
    print(f"[Warning] target_batch_size adjusted to {target_batch_size}")

accumulation_steps = target_batch_size // actual_batch_size

# ---------------------------------------------------------------------------
# Tokenizer & Model
# ---------------------------------------------------------------------------

print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(
    '/home/yangtao/clip-main/distilbert-base-uncased-local'
)

print(f"Loading model from {model_path}...")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
config = checkpoint['config']
model = BertForMaskedLM(config)
model.to(device)
print("Model loaded.")

# ---------------------------------------------------------------------------
# Dataset loading & preprocessing
# ---------------------------------------------------------------------------

print(f"Loading {len(file_paths)} dataset file(s)...")
datasets_list = []
total_samples = 0
for i, fp in enumerate(file_paths):
    print(f"  [{i+1}/{len(file_paths)}] {os.path.basename(fp)}")
    part = Dataset.from_file(fp)
    datasets_list.append(part)
    total_samples += len(part)

raw_dataset = concatenate_datasets(datasets_list)
print(f"Combined dataset: {len(raw_dataset):,} samples")

if len(raw_dataset) != total_samples:
    print(f"[Warning] Expected {total_samples:,} samples, got {len(raw_dataset):,}")


def tokenize_function(examples):
    """Tokenize raw text with truncation to block_size."""
    texts = examples["text"]
    if isinstance(texts, list):
        texts = [str(t) if t is not None else "" for t in texts]
    else:
        texts = str(texts) if texts is not None else ""
    return tokenizer(texts, truncation=True, max_length=block_size)


def group_texts(examples):
    """
    Concatenate all token sequences and split into fixed-length blocks.
    Drops the remainder that does not fill a complete block.
    """
    input_ids       = sum(examples["input_ids"], [])
    attention_mask  = sum(examples["attention_mask"], [])
    total_length    = (len(input_ids) // block_size) * block_size
    return {
        "input_ids": [
            input_ids[i : i + block_size] for i in range(0, total_length, block_size)
        ],
        "attention_mask": [
            attention_mask[i : i + block_size] for i in range(0, total_length, block_size)
        ],
    }


print("Tokenizing dataset...")
tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_dataset.column_names,
)

print("Grouping into fixed-length blocks...")
lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
lm_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
print(f"Final dataset size (blocks): {len(lm_dataset):,}")

# MLM data collator — automatically generates masked labels
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataloader = DataLoader(
    lm_dataset,
    batch_size=actual_batch_size,
    collate_fn=data_collator,
    num_workers=4,
    shuffle=True,
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def is_weight_parameter(name: str) -> bool:
    """Return True for weight tensors, excluding bias terms."""
    return "weight" in name.lower() and "bias" not in name.lower()


def compute_weight_stats(model) -> dict:
    """
    Compute per-layer mean absolute weight value for all weight parameters.

    Returns:
        dict mapping parameter name → {'weight_abs_mean', 'shape', 'num_params'}
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None and is_weight_parameter(name):
            stats[name] = {
                'weight_abs_mean': torch.abs(param.data).mean().item(),
                'shape':           list(param.shape),
                'num_params':      param.numel(),
            }
    return stats


def _accumulate_gradients(model, dataloader_iter, dataloader, accumulation_steps, device):
    """
    Run one full gradient-accumulation cycle (accumulation_steps mini-batches).
    Gradients are *not* zeroed before or after — caller is responsible.

    Returns:
        total_loss (float): sum of unscaled per-step losses.
    """
    total_loss = 0.0
    for step in range(accumulation_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        (loss / accumulation_steps).backward()
        total_loss += loss.item()
    return total_loss


def _skip_batches(model, dataloader_iter, dataloader, n_batches, accumulation_steps, device):
    """
    Advance the dataloader by n_batches logical batches without recording stats.
    Gradients are zeroed at the start of each logical batch.
    """
    for _ in tqdm(range(n_batches), desc="Skipping batches"):
        model.zero_grad()
        _accumulate_gradients(model, dataloader_iter, dataloader, accumulation_steps, device)


# ---------------------------------------------------------------------------
# Gradient-statistics functions
# ---------------------------------------------------------------------------

def compute_sgd_gradient_stats_accumulated(
    model, dataloader, target_batch_size, actual_batch_size, start_batch=10
):
    """
    Compute per-layer mean absolute gradient (SGD update direction) for the
    `start_batch`-th logical batch using gradient accumulation.

    The raw accumulated gradient is used directly (no momentum / adaptive scaling),
    matching the SGD update rule.

    Returns:
        gradient_stats (dict): per-layer {'grad_abs_mean', 'shape', 'num_params'}
        avg_loss (float): mean MLM loss over the accumulated mini-batches
    """
    model.train()
    accumulation_steps = target_batch_size // actual_batch_size
    dataloader_iter = iter(dataloader)

    # Advance past the first (start_batch - 1) logical batches
    _skip_batches(
        model, dataloader_iter, dataloader,
        start_batch - 1, accumulation_steps, device
    )

    # Accumulate gradients for the target logical batch
    model.zero_grad()
    print(f"Accumulating gradients for logical batch {start_batch} (SGD)...")
    total_loss = 0.0
    for step in tqdm(range(accumulation_steps), desc=f"Batch {start_batch} accumulation"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        if step == 0:
            print(f"  Step 0 — keys: {list(batch.keys())}, "
                  f"labels[:10]: {batch['labels'][0][:10]}")

        outputs = model(**batch)
        loss = outputs.loss
        if step == 0:
            print(f"  Step 0 — MLM loss: {loss.item():.6f}")
        (loss / accumulation_steps).backward()
        total_loss += loss.item()

    # Compute mean absolute gradient per weight layer
    gradient_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and is_weight_parameter(name):
            gradient_stats[name] = {
                'grad_abs_mean': torch.abs(param.grad).mean().item(),
                'shape':         list(param.shape),
                'num_params':    param.numel(),
            }

    return gradient_stats, total_loss / accumulation_steps


def compute_adam_dynamics_stats_accumulated(
    model, dataloader, target_batch_size, actual_batch_size,
    exp_avgs, exp_avg_sqs, state_steps,
    start_batch=10, beta1=0.9, beta2=0.999, eps=1e-8,
):
    """
    Compute per-layer mean absolute Adam update magnitude for the
    `start_batch`-th logical batch.

    Runs the Adam EMA updates for the first (start_batch - 1) batches to warm
    up the moment estimates, then computes the effective Adam step size on the
    target batch.

    Returns:
        gradient_stats (dict): per-layer {'grad_abs_mean', 'shape', 'num_params'}
        avg_loss (float): mean MLM loss over the target accumulated batch
    """
    model.train()
    accumulation_steps = target_batch_size // actual_batch_size
    dataloader_iter = iter(dataloader)

    # Warm up Adam moments over the first (start_batch - 1) logical batches
    for _ in tqdm(range(start_batch - 1), desc="Warming up Adam moments"):
        model.zero_grad()
        _accumulate_gradients(model, dataloader_iter, dataloader, accumulation_steps, device)

        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and is_weight_parameter(name):
                grad = param.grad
                exp_avgs[idx].mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sqs[idx].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                state_steps[idx] += 1
                idx += 1

    # Accumulate gradients for the target logical batch
    model.zero_grad()
    print(f"Accumulating gradients for logical batch {start_batch} (Adam)...")
    total_loss = 0.0
    for step in tqdm(range(accumulation_steps), desc=f"Batch {start_batch} accumulation"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        if step == 0:
            print(f"  Step 0 — keys: {list(batch.keys())}, "
                  f"labels[:10]: {batch['labels'][0][:10]}")

        outputs = model(**batch)
        loss = outputs.loss
        if step == 0:
            print(f"  Step 0 — MLM loss: {loss.item():.6f}")
        (loss / accumulation_steps).backward()
        total_loss += loss.item()

    # Compute Adam update magnitude per weight layer
    gradient_stats = {}
    idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and is_weight_parameter(name):
            grad        = param.grad
            exp_avg     = exp_avgs[idx]
            exp_avg_sq  = exp_avg_sqs[idx]
            step_t      = state_steps[idx]

            bias_correction1 = 1 - beta1 ** step_t
            bias_correction2 = 1 - beta2 ** step_t

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            denom       = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            adam_update = (exp_avg / bias_correction1) / denom

            gradient_stats[name] = {
                'grad_abs_mean': torch.abs(adam_update).mean().item(),
                'shape':         list(param.shape),
                'num_params':    param.numel(),
            }
            idx += 1

    return gradient_stats, total_loss / accumulation_steps


def compute_adams_dynamics_stats_accumulated(
    model, dataloader, target_batch_size, actual_batch_size,
    exp_avgs, exp_avg_sqs, init_avg_abs_params, state_steps,
    start_batch=10, beta1=0.9, beta2=0.999, eps=1e-8,
):
    """
    Compute per-layer AdamS update magnitude for the `start_batch`-th logical
    batch. AdamS scales the Adam update by the initial mean absolute parameter
    value, providing a parameter-scale-aware effective step size.

    Returns:
        gradient_stats (dict): per-layer {'grad_abs_mean', 'shape', 'num_params'}
        avg_loss (float): mean MLM loss over the target accumulated batch
    """
    model.train()
    accumulation_steps = target_batch_size // actual_batch_size
    dataloader_iter = iter(dataloader)

    # Warm up AdamS moments over the first (start_batch - 1) logical batches
    for _ in tqdm(range(start_batch - 1), desc="Warming up AdamS moments"):
        model.zero_grad()
        _accumulate_gradients(model, dataloader_iter, dataloader, accumulation_steps, device)

        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and is_weight_parameter(name):
                grad = param.grad
                exp_avgs[idx].mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sqs[idx].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                state_steps[idx] += 1
                idx += 1

    # Accumulate gradients for the target logical batch
    model.zero_grad()
    print(f"Accumulating gradients for logical batch {start_batch} (AdamS)...")
    total_loss = 0.0
    for step in tqdm(range(accumulation_steps), desc=f"Batch {start_batch} accumulation"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        if step == 0:
            print(f"  Step 0 — keys: {list(batch.keys())}, "
                  f"labels[:10]: {batch['labels'][0][:10]}")

        outputs = model(**batch)
        loss = outputs.loss
        if step == 0:
            print(f"  Step 0 — MLM loss: {loss.item():.6f}")
        (loss / accumulation_steps).backward()
        total_loss += loss.item()

    # Compute AdamS update magnitude per weight layer
    gradient_stats = {}
    idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and is_weight_parameter(name):
            grad                = param.grad
            exp_avg             = exp_avgs[idx]
            exp_avg_sq          = exp_avg_sqs[idx]
            init_avg_abs_param  = init_avg_abs_params[idx]
            step_t              = state_steps[idx]

            bias_correction1 = 1 - beta1 ** step_t
            bias_correction2 = 1 - beta2 ** step_t

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            denom       = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            adam_update = (exp_avg / bias_correction1) / denom

            # AdamS: scale effective step by initial mean absolute parameter value
            grad_abs_mean = (torch.abs(adam_update).mean() * init_avg_abs_param).item()

            gradient_stats[name] = {
                'grad_abs_mean': grad_abs_mean,
                'shape':         list(param.shape),
                'num_params':    param.numel(),
            }
            idx += 1

    return gradient_stats, total_loss / accumulation_steps


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def compute_weight_gradient_ratios(weight_stats: dict, gradient_stats: dict) -> dict:
    """
    Compute the ratio of mean absolute weight to mean absolute update for each
    layer present in both dicts.
    """
    ratios = {}
    for name, w in weight_stats.items():
        if name not in gradient_stats:
            continue
        g = gradient_stats[name]['grad_abs_mean']
        ratios[name] = {
            'weight_abs_mean':   w['weight_abs_mean'],
            'grad_abs_mean':     g,
            'weight_grad_ratio': w['weight_abs_mean'] / g if g != 0 else float('inf'),
            'shape':             w['shape'],
            'num_params':        w['num_params'],
        }
    return ratios


def compute_overall_statistics(ratios: dict) -> dict:
    """
    Compute summary statistics over all per-layer weight/gradient ratios.

    Returns a dict with:
        simple_mean   — arithmetic mean of per-layer ratios
        weighted_mean — parameter-count-weighted mean
        total_valid_layers, total_params, valid_ratios, valid_params
    """
    valid_ratios, valid_params = [], []
    weighted_sum, total_params = 0.0, 0

    for stats in ratios.values():
        r = stats['weight_grad_ratio']
        n = stats['num_params']
        if r != float('inf') and not np.isnan(r):
            valid_ratios.append(r)
            valid_params.append(n)
            weighted_sum += r * n
            total_params += n

    return {
        'simple_mean':        np.mean(valid_ratios) if valid_ratios else 0.0,
        'weighted_mean':      weighted_sum / total_params if total_params > 0 else 0.0,
        'total_valid_layers': len(valid_ratios),
        'total_params':       total_params,
        'valid_ratios':       valid_ratios,
        'valid_params':       valid_params,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Initialize Adam/AdamS moment buffers for all weight parameters
    exp_avgs, exp_avg_sqs, state_steps, init_avg_abs_params = [], [], [], []
    prev_init_scale = 1.0

    for name, param in model.named_parameters():
        if param.requires_grad and is_weight_parameter(name):
            exp_avgs.append(torch.zeros_like(param.data))
            exp_avg_sqs.append(torch.zeros_like(param.data))

            init_scale = torch.mean(torch.abs(param.data))
            if init_scale < 1e-6:
                # Fall back to a fraction of the previous layer's scale
                init_scale = 0.1 * prev_init_scale * torch.ones_like(
                    init_scale, memory_format=torch.preserve_format
                )
            prev_init_scale = init_scale
            init_avg_abs_params.append(init_scale)
            state_steps.append(1)  # Adam step counter starts at 1

    # Verify that the data collator produces MLM labels
    print("Verifying data collator output...")
    test_batch = next(iter(dataloader))
    assert 'labels' in test_batch, "Data collator did not produce labels!"
    masked_count = (test_batch['labels'][0] != -100).sum().item()
    print(f"  Labels shape: {test_batch['labels'].shape}")
    print(f"  Masked tokens in first sample: {masked_count}")

    start_batch = 10

    print("\nComputing weight statistics...")
    weight_stats = compute_weight_stats(model)
    print(f"  Found {len(weight_stats)} weight parameter layers.")

    print(f"\nComputing AdamS gradient statistics for logical batch {start_batch}...")
    gradient_stats, loss = compute_adams_dynamics_stats_accumulated(
        model, dataloader, target_batch_size, actual_batch_size,
        exp_avgs, exp_avg_sqs, init_avg_abs_params, state_steps,
        start_batch=start_batch,
    )
    print(f"  Computed stats for {len(gradient_stats)} weight layers.")

    print("\nComputing weight / update ratios...")
    ratios = compute_weight_gradient_ratios(weight_stats, gradient_stats)

    print("Computing overall statistics...")
    overall_stats = compute_overall_statistics(ratios)

    total_all_params    = sum(p.numel() for p in model.parameters())
    total_weight_params = sum(
        p.numel() for n, p in model.named_parameters() if is_weight_parameter(n)
    )

    # --- Save results ---
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    model_name   = os.path.basename(model_path).replace('.pth', '')
    results_file = (
        f"{results_dir}/{start_batch}_weight_gradient_analysis_adams_weights"
        f"_accumulated_batch{target_batch_size}_batch10_{model_name}.txt"
    )

    with open(results_file, 'w') as f:
        sep = "=" * 80

        f.write(sep + "\n")
        f.write(
            f"Weight / Update Ratio Analysis Report\n"
            f"  Optimizer : AdamS  |  Loss : MLM  |  "
            f"Logical batch : {target_batch_size}  |  Analyzed batch : {start_batch}\n"
            f"  Model     : {model_path}\n"
            f"  Device    : {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n"
        )
        f.write(sep + "\n\n")

        f.write("Parameter summary\n" + "-" * 40 + "\n")
        f.write(f"  Total parameters  : {total_all_params:,}\n")
        f.write(f"  Weight parameters : {total_weight_params:,}\n")
        f.write(f"  Bias parameters   : {total_all_params - total_weight_params:,}\n")
        f.write(f"  Weight ratio      : {total_weight_params / total_all_params * 100:.2f}%\n\n")

        f.write("Overall statistics\n" + sep + "\n")
        f.write(f"  Simple mean   (weight / update): {overall_stats['simple_mean']:.6f}\n")
        f.write(f"  Weighted mean (weight / update): {overall_stats['weighted_mean']:.6f}\n")
        f.write(f"    (weighted by parameter count: Σ(ratio × n_params) / Σ(n_params))\n")
        f.write(f"  Valid layers  : {overall_stats['total_valid_layers']}\n")
        f.write(f"  Total params  : {overall_stats['total_params']:,}\n")
        f.write(sep + "\n\n")

        f.write("Per-layer details\n" + "-" * 80 + "\n")
        for name, stats in sorted(ratios.items()):
            f.write(f"Layer : {name}\n")
            f.write(f"  Shape              : {stats['shape']}\n")
            f.write(f"  Parameters         : {stats['num_params']:,}\n")
            f.write(f"  |weight| mean      : {stats['weight_abs_mean']:.6f}\n")
            f.write(f"  |update| mean      : {stats['grad_abs_mean']:.6f}\n")
            f.write(f"  Ratio              : {stats['weight_grad_ratio']:.6f}\n")
            r = stats['weight_grad_ratio']
            if r != float('inf') and not np.isnan(r):
                contrib = r * stats['num_params'] / overall_stats['total_params']
                f.write(f"  Weighted contrib.  : {contrib:.6f}\n")
            f.write("-" * 40 + "\n")

        valid_ratios = overall_stats['valid_ratios']
        if valid_ratios:
            min_r, max_r = np.min(valid_ratios), np.max(valid_ratios)
            f.write("\nDistribution summary\n" + "-" * 80 + "\n")
            f.write(f"  Layers (total / valid) : {len(ratios)} / {len(valid_ratios)}\n")
            f.write(f"  Min                    : {min_r:.6f}\n")
            f.write(f"  Max                    : {max_r:.6f}\n")
            f.write(f"  Max / Min              : {max_r / min_r if min_r != 0 else float('inf'):.6f}\n")
            f.write(f"  Median                 : {np.median(valid_ratios):.6f}\n")
            f.write(f"  Std dev                : {np.std(valid_ratios):.6f}\n")
            f.write(f"  Variance               : {np.var(valid_ratios):.6f}\n")
            f.write(f"  25th percentile        : {np.percentile(valid_ratios, 25):.6f}\n")
            f.write(f"  75th percentile        : {np.percentile(valid_ratios, 75):.6f}\n")

        f.write(f"\nMLM loss (avg over accumulated batch) : {loss:.6f}\n")
        f.write(f"Effective sample count                 : {target_batch_size}\n")

    print(f"\nResults saved to: {results_file}")

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("Key results summary")
    print("=" * 60)
    print(f"  Total parameters      : {total_all_params:,}")
    print(f"  Weight parameters     : {total_weight_params:,} "
          f"({total_weight_params / total_all_params * 100:.2f}%)")
    print(f"  Logical batch size    : {target_batch_size}")
    print(f"  Accumulation steps    : {accumulation_steps}")
    print(f"  Analyzed batch        : {start_batch}")
    print(f"  Simple mean  (W/U)    : {overall_stats['simple_mean']:.6f}")
    print(f"  Weighted mean (W/U)   : {overall_stats['weighted_mean']:.6f}")
    print(f"  Valid layers          : {overall_stats['total_valid_layers']}")
    print(f"  MLM loss              : {loss:.6f}")
    print("=" * 60)
