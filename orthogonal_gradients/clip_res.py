import os
import sys
import time
import json
import random
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append('/home/clipre50/clip-main/clip/')
from clip import clip


# ---------------------------------------------------------------------------
# Logging utility
# ---------------------------------------------------------------------------

def print_stage(stage_name: str, details: str = "") -> None:
    print(f"\n{'='*60}")
    print(f"🔄 {stage_name}")
    if details:
        print(f"   {details}")
    print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def get_logits(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-image and per-text logit matrices while preserving gradients
    only for the first sample in the batch (index 0).
    """
    img_detached = image_features.detach().clone()
    txt_detached = text_features.detach().clone()

    # Restore gradients for the first sample only
    img_detached[0] = image_features[0]
    txt_detached[0] = text_features[0]

    logits_per_image = img_detached @ txt_detached.T
    logits_per_text  = txt_detached @ img_detached.T
    return logits_per_image, logits_per_text


def cal_clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    """Symmetric cross-entropy contrastive loss (CLIP-style)."""
    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2
    return loss


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Symmetric cross-entropy loss on a pre-computed logit matrix,
    preserving gradients only for the first row/column.
    """
    targets = torch.arange(logits.size(0), device=logits.device)
    logits_mixed = logits.detach().clone()
    logits_mixed[0, :] = logits[0, :]
    logits_mixed[:, 0] = logits[:, 0]
    loss = (
        F.cross_entropy(logits_mixed, targets) +
        F.cross_entropy(logits_mixed.t(), targets)
    ) / 2
    return loss


def siglip_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Pairwise sigmoid loss (SigLIP-style),
    preserving gradients only for the first row/column.
    """
    n = logits.size(0)
    labels = 2 * torch.eye(n, device=logits.device) - 1  # +1 on diagonal, -1 off
    logits_mixed = logits.detach().clone()
    logits_mixed[0, :] = logits[0, :]
    logits_mixed[:, 0] = logits[:, 0]
    return -torch.sum(F.logsigmoid(labels * logits_mixed)) / n


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCODataset(Dataset):
    """COCO Captions dataset that returns (image, caption, image_id) tuples."""

    def __init__(self, images_path: str, annotations_path: str, max_samples: int = None):
        print_stage("Initializing COCO Dataset", f"Image dir: {images_path}")
        self.images_path = images_path

        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)

        # Build image_id -> captions mapping
        self.image_id_to_captions: dict[int, list[str]] = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            self.image_id_to_captions.setdefault(img_id, []).append(ann['caption'])

        # Keep only images that have at least one caption
        self.images = [
            img for img in coco_data['images']
            if img['id'] in self.image_id_to_captions
        ]

        if max_samples:
            self.images = self.images[:max_samples]

        print(f"✅ Dataset ready: {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_info = self.images[idx]
        img_path = os.path.join(self.images_path, image_info['file_name'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        caption = random.choice(self.image_id_to_captions[image_info['id']])
        return image, caption, image_info['id']


def custom_collate_fn(batch):
    images, captions, image_ids = zip(*batch)
    return list(images), list(captions), list(image_ids)


# ---------------------------------------------------------------------------
# NAO computation
# ---------------------------------------------------------------------------

def cal_nao_distribution(
    outer_loader: DataLoader,
    inner_loader: DataLoader,
    criterion,
    model,
    preprocess,
    path: str,
    loop_num: int = 100,
    encoder: str = "text",       # "text" -> 'transformer'; "image" -> 'visual'
) -> None:
    """
    Compute the Normalized Gradient Alignment Orthogonality (NAO) distribution
    between pairs of mini-batches.

    Args:
        encoder: which encoder's gradients to analyse.
                 'text'  -> parameters whose name contains 'transformer'
                 'image' -> parameters whose name contains 'visual'
    """
    # Select the parameter name keyword based on the chosen encoder
    param_keyword = "transformer" if encoder == "text" else "visual"

    print_stage("Computing NAO Distribution", f"Output: {path} | Encoder: {encoder} ({param_keyword})")

    naos      = [0] * 10
    group_num = 10
    total_num = 0
    avg       = 0.0
    device    = model.visual.conv1.weight.device

    model.eval()

    outer_pbar = tqdm(
        enumerate(outer_loader),
        total=min(loop_num, len(outer_loader)),
        desc="Outer loop",
        ncols=100,
    )

    for i, (images, captions, _) in outer_pbar:
        if i >= loop_num:
            break

        outer_pbar.set_postfix({
            'batch': f'{i+1}/{loop_num}',
            'total_pairs': total_num,
            'avg_nao': f'{avg / (total_num + 1e-8):.4f}',
        })

        if len(images) <= 1:
            print(f"⚠️ Skipping batch with size={len(images)}")
            continue

        # Preprocess and encode outer batch
        images_tensor = torch.stack([preprocess(img) for img in images]).to(device)
        texts_tensor  = clip.tokenize(captions).to(device)

        image_features = F.normalize(model.encode_image(images_tensor), p=2, dim=1)
        text_features  = F.normalize(model.encode_text(texts_tensor),   p=2, dim=1)

        loss = cal_clip_loss(image_features, text_features, 0)
        loss.backward()

        # Collect reference gradients from the selected encoder
        init_grads = []
        for name, param in model.named_parameters():
            if param_keyword in name and param.grad is not None:
                init_grads.append(param.grad.clone().reshape(-1))
            if param.grad is not None:
                param.grad.zero_()

        model.eval()

        inner_pbar = tqdm(
            enumerate(inner_loader),
            desc=f"Inner loop (outer {i+1})",
            leave=False,
            ncols=80,
        )

        for j, (inner_images, inner_captions, _) in inner_pbar:
            if j <= i:
                continue
            if j >= loop_num:
                break

            total_num += 1
            inner_pbar.set_postfix({'total_pairs': total_num})

            if len(inner_images) <= 1:
                total_num -= 1
                continue

            # Preprocess and encode inner batch
            inner_images_tensor = torch.stack([preprocess(img) for img in inner_images]).to(device)
            inner_texts_tensor  = clip.tokenize(inner_captions).to(device)

            inner_image_features = F.normalize(model.encode_image(inner_images_tensor), p=2, dim=1)
            inner_text_features  = F.normalize(model.encode_text(inner_texts_tensor),   p=2, dim=1)

            inner_loss = cal_clip_loss(inner_image_features, inner_text_features, 0)
            inner_loss.backward()

            # Compute weighted NAO between reference and inner gradients
            nao      = 0
            num      = 0
            grad_idx = 0

            for name, param in model.named_parameters():
                if param_keyword in name and param.grad is not None:
                    g        = param.grad.reshape(-1)
                    next_num = num + g.size(0)

                    if grad_idx < len(init_grads):
                        init_grad = init_grads[grad_idx]
                        # Cosine-like alignment: dot product of unit-normalised absolute gradients
                        nao_term = torch.sum(
                            torch.abs(init_grad / torch.norm(init_grad)) *
                            torch.abs(g        / torch.norm(g))
                        )
                        # Weighted running average (avoids overflow from large nao*num)
                        nao = nao * (num / next_num) + (g.size(0) / next_num) * nao_term
                        grad_idx += 1

                    num = next_num

                if param.grad is not None:
                    param.grad.zero_()

            if torch.isnan(nao).any() or torch.isinf(nao).any():
                total_num -= 1
                continue

            avg += nao.item()
            index    = 0
            nao_temp = nao.clone()
            while nao_temp > 1 / group_num and index < 9:
                index   += 1
                nao_temp -= 1 / group_num
            naos[index] += 1

        inner_pbar.close()

        if (i + 1) % 10 == 0:
            print(f"\n📈 Intermediate stats (outer {i+1}/{loop_num}):")
            print(f"   Total pairs : {total_num}")
            print(f"   Average NAO : {avg / total_num:.6f}")
            print(f"   Distribution: {[f'{n/total_num:.3f}' for n in naos[:5]]}...")

    outer_pbar.close()

    print_stage("Finalising Results")
    naos = [n / total_num for n in naos]
    naos.append(avg / total_num)

    print(f"📊 Final stats:")
    print(f"   Total pairs : {total_num}")
    print(f"   Average NAO : {avg / total_num:.6f}")
    print(f"   Distribution: {[f'{n:.4f}' for n in naos[:-1]]}")

    pd.DataFrame(data=naos).to_csv(path)
    print(f"✅ Results saved to: {path}")


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------

def load_optimizer_csvs(name_list: list, label_list: list, save_path: str) -> str:
    """Merge multiple per-model NAO CSV files into a single summary CSV."""
    print_stage("Merging CSV Files", f"Output: {save_path}")
    data_dict = {}

    pbar = tqdm(zip(name_list, label_list), total=len(name_list), desc="Loading CSVs", ncols=100)
    for file, label in pbar:
        pbar.set_postfix({'file': os.path.basename(file)})
        try:
            df = pd.read_csv(file, index_col=0)
            values = df.iloc[:, 0].values
            if len(values) != 11:
                print(f"⚠️ Skipping {file}: expected 11 rows, got {len(values)}")
                continue
            data_dict[label] = values
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")
    pbar.close()

    poly_bins  = [f"{i/10:.1f}" for i in range(11)]
    result_df  = pd.DataFrame.from_dict(data_dict, orient='index', columns=poly_bins)
    result_df.to_csv(save_path)
    print(f"✅ Merged CSV saved to: {save_path}")
    return save_path


def heat_map(csv_path: str, labels: list, save_path: str) -> None:
    """Plot a heatmap of the NAO distribution across models."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path, index_col=0)
    plt.figure(figsize=(8, 4))
    sns.heatmap(df.iloc[:, :-1], annot=True, cmap='Blues', yticklabels=labels)
    plt.title('Gradient Orthogonality Heatmap (CLIP Contrastive Loss)')
    plt.xlabel('NAO Bin')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Heatmap saved to: {save_path}")


def plot_average_nao(csv_path: str, labels: list, save_path: str) -> None:
    """Bar chart of the average NAO value per model (last column of the CSV)."""
    import matplotlib.pyplot as plt

    df       = pd.read_csv(csv_path, index_col=0)
    last_col = df.iloc[:, -1]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, last_col)
    plt.ylabel('Average NAO')
    plt.title('Average Gradient Orthogonality (CLIP Contrastive Loss)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Average NAO bar chart saved to: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NAO analysis for CLIP on COCO Captions")
    parser.add_argument('--gpu_id',        type=int,   default=0,    help='GPU index to use')
    parser.add_argument('--loop_num',      type=int,   default=100,  help='Number of batch pairs for NAO analysis')
    parser.add_argument('--max_samples',   type=int,   default=1000, help='Maximum number of dataset samples')
    parser.add_argument('--batch_size',    type=int,   default=8,    help='Batch size (must be > 1)')
    parser.add_argument(
        '--encoder',
        type=str,
        default='text',
        choices=['text', 'image'],
        help="Which encoder's gradients to analyse: 'text' (transformer) or 'image' (visual)",
    )
    args = parser.parse_args()

    # Derive output tag and file paths from the chosen encoder
    tag              = f"clip_res_{args.encoder}_contrastive"
    log_path         = f"./grad_{tag}_coco.csv"
    merge_csv_path   = f"./compare_{tag}_coco.csv"
    heatmap_path     = f"./heatmap_{tag}.png"
    avg_nao_path     = f"./avg_nao_{tag}.png"

    print_stage("Starting NAO Analysis", f"Encoder: {args.encoder} | Model: CLIP RN50")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load("RN50", device=device)
    model.eval()

    images_path      = "/home/train2017/train2017"
    annotations_path = "/home/annotations/annotations/captions_train2017.json"

    print_stage("Loading COCO Dataset")
    dataset     = COCODataset(images_path, annotations_path, max_samples=args.max_samples)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    print(f"✅ Loaded {len(dataset)} images, batch_size={args.batch_size}")

    print_stage(f"Running NAO: {tag}")
    cal_nao_distribution(
        data_loader, data_loader,
        criterion=contrastive_loss,
        model=model,
        preprocess=preprocess,
        path=log_path,
        loop_num=args.loop_num,
        encoder=args.encoder,
    )

    load_optimizer_csvs([log_path], [tag], merge_csv_path)
    heat_map(merge_csv_path, [tag], heatmap_path)
    plot_average_nao(merge_csv_path, [tag], avg_nao_path)

    print_stage("All Done 🎉")
