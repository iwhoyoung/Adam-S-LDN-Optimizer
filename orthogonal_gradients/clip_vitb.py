import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import json
from torch import nn
import torch.nn.functional as F
import argparse
from transformers import AutoProcessor, AutoModel
import random
import numpy as np


def print_stage(stage_name, details=""):
    print(f"\n{'='*60}")
    print(f"🔄 {stage_name}")
    if details:
        print(f"   {details}")
    print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


def get_logits(image_features, text_features, logit_scale):
    img_detached = image_features.detach().clone()
    txt_detached = text_features.detach().clone()

    # Preserve gradients for the first sample only
    img_detached[0] = image_features[0]
    txt_detached[0] = text_features[0]

    logits_per_image = img_detached @ txt_detached.T
    logits_per_text  = txt_detached @ img_detached.T
    return logits_per_image, logits_per_text


def cal_clip_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features  = F.normalize(text_features,  p=2, dim=1)
    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    total_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text,  labels)
    ) / 2
    return total_loss


def contrastive_loss(logits):
    targets      = torch.arange(logits.size(0)).to(logits.device)
    logits_mixed = logits.detach().clone()
    logits_mixed[0, :] = logits[0, :]
    logits_mixed[:, 0] = logits[:, 0]
    loss_images = F.cross_entropy(logits_mixed,     targets)
    loss_texts  = F.cross_entropy(logits_mixed.t(), targets)
    return (loss_images + loss_texts) / 2


def siglip_loss(logits):
    n      = logits.size(0)
    labels = 2 * torch.eye(n, device=logits.device) - 1  # +1 on diagonal, -1 off-diagonal
    logits_mixed = logits.detach().clone()
    logits_mixed[0, :] = logits[0, :]
    logits_mixed[:, 0] = logits[:, 0]
    return -torch.sum(F.logsigmoid(labels * logits_mixed)) / n


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCODataset(Dataset):
    def __init__(self, images_path, annotations_path, processor=None, max_samples=None):
        print_stage("Initializing COCO Dataset", f"Image dir: {images_path}")
        self.images_path = images_path
        self.processor   = processor

        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)

        self.image_id_to_captions = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            self.image_id_to_captions.setdefault(img_id, []).append(ann['caption'])

        self.images = [
            img for img in coco_data['images']
            if img['id'] in self.image_id_to_captions
        ]

        if max_samples:
            self.images = self.images[:max_samples]

        print(f"✅ Dataset ready: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_path   = os.path.join(self.images_path, image_info['file_name'])
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
# Gradient Orthogonality computation
# ---------------------------------------------------------------------------

def cal_nao_distribution(
    outer_loader,
    inner_loader,
    criterion,
    model,
    processor,
    path,
    loop_num=100,
    encoder="image",    # "image" -> 'vision_model';  "text" -> 'text_model'
):
    """
    Compute the Gradient Orthogonality distribution between pairs of mini-batches.

    Args:
        encoder: which encoder's gradients to analyse.
                 'image' -> parameters whose name contains 'vision_model'
                 'text'  -> parameters whose name contains 'text_model'
    """
    param_keyword = "vision_model" if encoder == "image" else "text_model"

    print_stage("Computing Gradient Orthogonality Distribution",
                f"Output: {path} | Encoder: {encoder} ({param_keyword})")

    go_bins   = [0] * 10   # histogram bins for gradient orthogonality scores
    group_num = 10
    model.eval()
    total_num = 0
    avg       = 0.0

    print(f"✅ Using CLIP contrastive loss | Encoder: {encoder}")

    outer_pbar = tqdm(
        enumerate(outer_loader),
        total=min(loop_num, len(outer_loader)),
        desc="🔄 Outer loop",
        ncols=100,
    )

    for i, batch_data in outer_pbar:
        if i >= loop_num:
            break

        outer_pbar.set_postfix({
            'batch':       f'{i+1}/{loop_num}',
            'total_pairs': total_num,
            'avg_go':      f'{avg / (total_num + 1e-8):.4f}',
        })

        images, captions, image_ids = batch_data
        if len(images) <= 1:
            print(f"⚠️ Skipping batch with size={len(images)}")
            continue

        try:
            inputs = processor(
                text=captions, images=images,
                return_tensors="pt", padding=True, truncation=True,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
        except Exception as e:
            print(f"⚠️ Failed to process batch: {e}")
            continue

        outputs        = model(**inputs)
        image_features = outputs.image_embeds
        text_features  = outputs.text_embeds

        loss = cal_clip_loss(image_features, text_features, 0)
        loss.backward()

        # Collect reference gradients from the target encoder
        ref_grads = []
        for name, param in model.named_parameters():
            if param_keyword in name and param.grad is not None:
                ref_grads.append(param.grad.clone().reshape(-1))
            if param.grad is not None:
                param.grad.zero_()

        inner_pbar = tqdm(
            enumerate(inner_loader),
            desc=f"Inner loop (outer {i+1})",
            leave=False,
            ncols=80,
        )
        inner_count = 0

        for j, inner_batch_data in inner_pbar:
            if j <= i:
                continue
            if j >= loop_num:
                break

            inner_count += 1
            total_num   += 1
            inner_pbar.set_postfix({'pair': f'{inner_count}', 'total': total_num})

            inner_images, inner_captions, inner_image_ids = inner_batch_data
            if len(inner_images) <= 1:
                total_num -= 1
                continue

            try:
                inner_inputs = processor(
                    text=inner_captions, images=inner_images,
                    return_tensors="pt", padding=True, truncation=True,
                )
                inner_inputs = {k: v.cuda() for k, v in inner_inputs.items()}
            except Exception as e:
                print(f"⚠️ Failed to process inner batch: {e}")
                total_num -= 1
                continue

            inner_outputs        = model(**inner_inputs)
            inner_image_features = inner_outputs.image_embeds
            inner_text_features  = inner_outputs.text_embeds

            inner_loss = cal_clip_loss(inner_image_features, inner_text_features, 0)
            inner_loss.backward()

            # Compute weighted gradient orthogonality score across target encoder layers
            go_score = 0
            num      = 0
            grad_idx = 0

            for name, param in model.named_parameters():
                if param_keyword in name and param.grad is not None:
                    g        = param.grad.reshape(-1)
                    next_num = num + g.size(0)

                    if grad_idx < len(ref_grads):
                        ref_grad = ref_grads[grad_idx]
                        # Dot product of unit-normalised absolute gradients (gradient orthogonality term)
                        go_term  = torch.sum(
                            torch.abs(ref_grad / torch.norm(ref_grad)) *
                            torch.abs(g        / torch.norm(g))
                        )
                        # Layer-size-weighted running average
                        go_score = go_score * (num / next_num) + (g.size(0) / next_num) * go_term
                        grad_idx += 1

                    num = next_num

                if param.grad is not None:
                    param.grad.zero_()

            if torch.isnan(go_score).any() or torch.isinf(go_score).any():
                total_num -= 1
                continue

            avg     += go_score.item()
            index    = 0
            go_temp  = go_score.clone()
            while go_temp > 1 / group_num and index < 9:
                index   += 1
                go_temp -= 1 / group_num
            go_bins[index] += 1

        inner_pbar.close()

        if (i + 1) % 10 == 0:
            print(f"\n📈 Intermediate stats (outer {i+1}/{loop_num}):")
            print(f"   Total pairs        : {total_num}")
            print(f"   Avg gradient orthogonality: {avg / total_num:.6f}")
            current_dist = [n / total_num for n in go_bins]
            print(f"   Distribution (first 5 bins): {[f'{d:.3f}' for d in current_dist[:5]]}...")

    outer_pbar.close()

    print_stage("Finalising Results")
    go_bins = [n / total_num for n in go_bins]
    go_bins.append(avg / total_num)  # last entry: average gradient orthogonality

    print(f"📊 Final stats:")
    print(f"   Total pairs        : {total_num}")
    print(f"   Avg gradient orthogonality: {avg / total_num:.6f}")
    print(f"   Distribution: {[f'{n:.4f}' for n in go_bins[:-1]]}")

    pd.DataFrame(data=go_bins).to_csv(path)
    print(f"✅ Results saved to: {path}")


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------

def load_optimizer_csvs(name_list, label_list, save_path):
    print_stage("Merging CSV Files", f"Output: {save_path}")
    data_dict = {}
    pbar = tqdm(zip(name_list, label_list), total=len(name_list), desc="📁 Loading CSVs", ncols=100)

    for file, label in pbar:
        pbar.set_postfix({'file': os.path.basename(file)})
        try:
            df     = pd.read_csv(file, index_col=0)
            values = df.iloc[:, 0].values
            if len(values) != 11:
                print(f"⚠️ Skipping {file}: expected 11 rows, got {len(values)}")
                continue
            data_dict[label] = values
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")

    pbar.close()
    poly_bins = [f"{i/10:.1f}" for i in range(11)]
    result_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=poly_bins)
    result_df.to_csv(save_path)
    print(f"✅ Merged CSV saved to: {save_path}")
    return save_path


def heat_map(csv_path, labels, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv(csv_path, index_col=0)
    plt.figure(figsize=(8, 4))
    sns.heatmap(df.iloc[:, :-1], annot=True, cmap='Blues', yticklabels=labels)
    plt.title('Gradient Orthogonality Heatmap - CLIP (Image-Text Contrastive Loss)')
    plt.xlabel('Gradient Orthogonality Bin')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Heatmap saved to: {save_path}")


def plot_last_column_from_heatmap(csv_path, labels, save_path):
    import matplotlib.pyplot as plt
    df       = pd.read_csv(csv_path, index_col=0)
    last_col = df.iloc[:, -1]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, last_col)
    plt.ylabel('Average Gradient Orthogonality')
    plt.title('Average Gradient Orthogonality - CLIP (Image-Text Contrastive Loss)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Bar chart saved to: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Gradient Orthogonality analysis for CLIP on COCO Captions"
    )
    parser.add_argument('--gpu_id',          type=int, default=0,
                        help='GPU index to use')
    parser.add_argument('--pretrained_path', type=str, default="/home/clip-vit-base-patch32",
                        help='Path to pretrained model weights')
    parser.add_argument('--loop_num',        type=int, default=100,
                        help='Number of batch pairs for gradient orthogonality analysis')
    parser.add_argument('--max_samples',     type=int, default=1000,
                        help='Maximum number of dataset samples to use')
    parser.add_argument('--batch_size',      type=int, default=8,
                        help='Batch size (must be > 1 for contrastive loss)')
    parser.add_argument(
        '--encoder',
        type=str,
        default='image',
        choices=['image', 'text'],
        help="Encoder to analyse: 'image' (vision_model) or 'text' (text_model)",
    )
    args = parser.parse_args()

    # Output tag and file paths derived from the chosen encoder
    tag              = f"clip_vit_{args.encoder}_contrastive"
    log_path         = f"./grad_{tag}_coco.csv"
    merge_csv_path   = f"./compare_{tag}_coco.csv"
    heatmap_path     = f"./heatmap_{tag}.png"
    last_column_path = f"./last_column_{tag}.png"

    print_stage("Starting", f"Model: CLIP ViT | Encoder: {args.encoder}")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained(args.pretrained_path)
    model     = CLIPModel.from_pretrained(args.pretrained_path)
    model     = model.to(device)
    model.eval()

    criterion = contrastive_loss

    images_path      = "/home/train2017/train2017"
    annotations_path = "/home/annotations/annotations/captions_train2017.json"

    print_stage("Loading COCO Dataset")
    dataset     = COCODataset(images_path, annotations_path, processor=processor, max_samples=args.max_samples)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print(f"✅ Loaded {len(dataset)} images, batch_size={args.batch_size}")

    print_stage(f"Running: {tag}")
    cal_nao_distribution(
        data_loader, data_loader,
        criterion=criterion,
        model=model,
        processor=processor,
        path=log_path,
        loop_num=args.loop_num,
        encoder=args.encoder,
    )

    load_optimizer_csvs([log_path], [tag], merge_csv_path)
    heat_map(merge_csv_path, [tag], heatmap_path)
    plot_last_column_from_heatmap(merge_csv_path, [tag], last_column_path)

    print_stage("Done 🎉", "All tasks completed.")
