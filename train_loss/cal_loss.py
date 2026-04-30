import os
import re
import json
import math
import time
import random
import collections
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset as HFDataset, concatenate_datasets
from finalbert import BERTModel, build_token_groups, Vocab


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def count_corpus(tokens):
    """Count token frequencies. Flattens nested lists if necessary."""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
    Build token sequence and segment IDs for BERT input.
    Format: [CLS] tokens_a [SEP] (tokens_b [SEP])
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def try_gpu(i=0):
    """Return gpu(i) if available, otherwise cpu."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WikiTextBERTDataset(Dataset):
    """
    PyTorch Dataset for BERT pre-training on WikiText.

    Loads Arrow-format HuggingFace dataset files, builds MLM + NSP examples,
    and pads them to a fixed sequence length.

    Args:
        file_paths (list[str]): Paths to Arrow dataset files.
        max_len (int): Maximum token sequence length.
        min_freq (int): Minimum token frequency for vocabulary inclusion.
        vocab (Vocab, optional): Pre-built vocabulary. If None, builds from data.
    """

    def __init__(self, file_paths, max_len=128, min_freq=5, vocab=None):
        self.max_len = max_len

        print("Loading datasets from Arrow files...")
        datasets_list = [HFDataset.from_file(fp) for fp in file_paths]
        self.dataset = concatenate_datasets(datasets_list)

        print("Preprocessing text data...")
        paragraphs = self._extract_paragraphs()
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]

        if vocab is None:
            print("Building vocabulary...")
            self.vocab = Vocab(
                sentences,
                min_freq=min_freq,
                reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>']
            )
        else:
            print("Using provided vocabulary.")
            self.vocab = vocab

        print("Generating training examples...")
        examples = []
        for paragraph in paragraphs:
            examples.extend(
                self._get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len)
            )

        examples = [
            (self._get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
            for tokens, segments, is_next in examples
        ]

        (
            self.all_token_ids,
            self.all_segments,
            self.valid_lens,
            self.all_pred_positions,
            self.all_mlm_weights,
            self.all_mlm_labels,
            self.nsp_labels,
        ) = self._pad_bert_inputs(examples, max_len, self.vocab)

        print(f"Dataset ready: {len(self.all_token_ids)} examples.")

    def _extract_paragraphs(self):
        """
        Extract and tokenize paragraphs from the raw dataset.
        Returns at most 50,000 shuffled paragraphs, each with >= 2 sentences.
        """
        paragraphs = []

        for item in self.dataset:
            text = item['text']
            if not text or len(text.strip()) < 10:
                continue

            text = text.strip().lower()
            text = re.sub(r'\n+', ' . ', text)
            text = re.sub(r'\s+', ' ', text)

            sentences = [s.strip() for s in text.split(' . ') if s.strip()]

            tokenized_sentences = [
                s.split() for s in sentences if len(s.split()) >= 3
            ]

            if len(tokenized_sentences) >= 2:
                paragraphs.append(tokenized_sentences)

        random.shuffle(paragraphs)
        return paragraphs[:50000]

    def _get_next_sentence(self, sentence, next_sentence, paragraphs):
        """
        Sample a sentence pair for the NSP task.
        With 50% probability, use the actual next sentence (is_next=True);
        otherwise replace with a random sentence (is_next=False).
        """
        if random.random() < 0.5:
            is_next = True
        else:
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    def _get_nsp_data_from_paragraph(self, paragraph, paragraphs, vocab, max_len):
        """Generate NSP training pairs from a single paragraph."""
        nsp_data = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self._get_next_sentence(
                paragraph[i], paragraph[i + 1], paragraphs
            )
            # Skip pairs that exceed max_len after adding special tokens
            if len(tokens_a) + len(tokens_b) + 3 > max_len:
                continue
            tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data.append((tokens, segments, is_next))
        return nsp_data

    def _replace_mlm_tokens(self, tokens, candidate_pred_positions, num_mlm_preds, vocab):
        """
        Apply the MLM masking strategy:
          - 80% of the time: replace with <mask>
          - 10% of the time: keep the original token
          - 10% of the time: replace with a random vocabulary token
        """
        mlm_input_tokens = list(tokens)
        pred_positions_and_labels = []

        random.shuffle(candidate_pred_positions)
        for position in candidate_pred_positions:
            if len(pred_positions_and_labels) >= num_mlm_preds:
                break

            if random.random() < 0.8:
                masked_token = '<mask>'
            elif random.random() < 0.5:
                masked_token = tokens[position]
            else:
                masked_token = random.choice(vocab.idx_to_token)

            mlm_input_tokens[position] = masked_token
            pred_positions_and_labels.append((position, tokens[position]))

        return mlm_input_tokens, pred_positions_and_labels

    def _get_mlm_data_from_tokens(self, tokens, vocab):
        """
        Build MLM inputs from a token sequence.
        Masks ~15% of non-special tokens and returns token IDs,
        prediction positions, and ground-truth label IDs.
        """
        candidate_pred_positions = [
            i for i, token in enumerate(tokens) if token not in ('<cls>', '<sep>')
        ]
        num_mlm_preds = max(1, round(len(tokens) * 0.15))

        mlm_input_tokens, pred_positions_and_labels = self._replace_mlm_tokens(
            tokens, candidate_pred_positions, num_mlm_preds, vocab
        )

        pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
        pred_positions = [v[0] for v in pred_positions_and_labels]
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

        return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

    def _pad_bert_inputs(self, examples, max_len, vocab):
        """Pad all BERT inputs to fixed length and return as tensor lists."""
        max_num_mlm_preds = round(max_len * 0.15)
        all_token_ids, all_segments, valid_lens = [], [], []
        all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
        nsp_labels = []

        for token_ids, pred_positions, mlm_pred_label_ids, segments, is_next in examples:
            all_token_ids.append(torch.tensor(
                token_ids + [vocab['<pad>']] * (max_len - len(token_ids)),
                dtype=torch.long
            ))
            all_segments.append(torch.tensor(
                segments + [0] * (max_len - len(segments)),
                dtype=torch.long
            ))
            valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
            all_pred_positions.append(torch.tensor(
                pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)),
                dtype=torch.long
            ))
            all_mlm_weights.append(torch.tensor(
                [1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32
            ))
            all_mlm_labels.append(torch.tensor(
                mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
                dtype=torch.long
            ))
            nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

        return (
            all_token_ids, all_segments, valid_lens,
            all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels,
        )

    def __getitem__(self, idx):
        return (
            self.all_token_ids[idx],
            self.all_segments[idx],
            self.valid_lens[idx],
            self.all_pred_positions[idx],
            self.all_mlm_weights[idx],
            self.all_mlm_labels[idx],
            self.nsp_labels[idx],
        )

    def __len__(self):
        return len(self.all_token_ids)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def main():
    # --- Configuration ---
    ckpt_root = Path("/home/yangtao/train_bert/adams_bert_exp_bs32_len128_hid256_heads4_layers4_drop1_lr1e-2_steps50000")
    file_paths = [
        "/home/yangtao/Salesforce___wikitext/wikitext-103-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train-00001-of-00002.arrow",
        "/home/yangtao/Salesforce___wikitext/wikitext-103-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train-00000-of-00002.arrow",
    ]
    batch_size = 32
    num_groups = 10
    max_eval_samples = 1000
    output_png = "adams_lr1e-2_group_losses1.png"
    output_csv = "adams_lr1e-2_group_losses1.csv"

    # --- Load experiment config ---
    config_path = ckpt_root / "experiment_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # --- Discover and sort checkpoint files ---
    ckpt_files = sorted(
        [
            p for p in ckpt_root.iterdir()
            if p.is_file() and re.match(r"checkpoint_step_\d+\.pth", p.name)
        ],
        key=lambda p: int(p.name.split("_")[2].split(".")[0]),
    )
    if not ckpt_files:
        raise RuntimeError("No checkpoint files found!")

    # Load vocabulary from the first checkpoint
    ckpt_sample = torch.load(ckpt_files[0], map_location='cpu', weights_only=False)
    vocab = ckpt_sample['vocab']

    # --- Build evaluation dataset ---
    print("Loading evaluation dataset...")
    eval_dataset = WikiTextBERTDataset(file_paths, max_len=config['max_len'], vocab=vocab)
    if max_eval_samples is not None:
        indices = list(range(min(max_eval_samples, len(eval_dataset))))
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices)

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Build token frequency groups for fine-grained loss analysis ---
    print("Building token groups...")
    token_to_group, group_token_ids, group_stats, token_freq_dict = build_token_groups(
        vocab, num_groups=num_groups
    )

    # --- Evaluate each checkpoint ---
    all_metrics = defaultdict(list)
    steps = []

    for ckpt_file in ckpt_files:
        step = int(ckpt_file.name.split("_")[2].split(".")[0])
        steps.append(step)
        print(f"Evaluating checkpoint: {ckpt_file.name}  (step {step})")

        net = BERTModel(
            vocab_size=len(vocab),
            num_hiddens=config['num_hiddens'],
            norm_shape=[config['num_hiddens']],
            ffn_num_input=config['num_hiddens'],
            ffn_num_hiddens=config['num_hiddens'] * 4,
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_len=config['max_len'],
            key_size=config['num_hiddens'],
            query_size=config['num_hiddens'],
            value_size=config['num_hiddens'],
            hid_in_features=config['num_hiddens'],
            mlm_in_features=config['num_hiddens'],
            nsp_in_features=config['num_hiddens'],
        ).cuda().eval()

        checkpoint = torch.load(ckpt_file, map_location='cuda', weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])

        total_loss = 0.0
        total_count = 0
        nsp_loss_sum = 0.0
        nsp_count = 0
        group_losses = [0.0] * num_groups
        group_counts = [0] * num_groups

        with torch.no_grad():
            for batch in eval_loader:
                (tokens_X, segments_X, valid_lens_x,
                 pred_positions_X, mlm_weights_X, mlm_Y, nsp_y) = batch

                tokens_X       = tokens_X.cuda()
                segments_X     = segments_X.cuda()
                valid_lens_x   = valid_lens_x.cuda()
                pred_positions_X = pred_positions_X.cuda()
                mlm_weights_X  = mlm_weights_X.cuda()
                mlm_Y          = mlm_Y.cuda()
                nsp_y          = nsp_y.cuda()

                _, mlm_Y_hat, nsp_Y_hat = net(
                    tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_X
                )

                # NSP loss
                nsp_l = F.cross_entropy(nsp_Y_hat, nsp_y)
                nsp_loss_sum += nsp_l.item()
                nsp_count += 1

                # Per-token MLM loss, accumulated per frequency group
                bs, num_pred = mlm_Y_hat.shape[:2]
                for b in range(bs):
                    for j in range(num_pred):
                        if mlm_weights_X[b, j] > 0:
                            label = mlm_Y[b, j].item()
                            logits = mlm_Y_hat[b, j]
                            loss = F.cross_entropy(
                                logits.unsqueeze(0),
                                torch.tensor([label], device=logits.device),
                                reduction='none',
                            ).item()
                            total_loss += loss
                            total_count += 1
                            gid = token_to_group.get(label, -1)
                            if 0 <= gid < num_groups:
                                group_losses[gid] += loss
                                group_counts[gid] += 1

        # Record aggregated metrics
        avg_mlm = total_loss / total_count if total_count > 0 else float('inf')
        avg_nsp = nsp_loss_sum / nsp_count if nsp_count > 0 else float('inf')
        all_metrics["total_loss"].append(avg_mlm)
        all_metrics["nsp_loss"].append(avg_nsp)

        for gid in range(num_groups):
            avg_g = group_losses[gid] / group_counts[gid] if group_counts[gid] > 0 else float('inf')
            all_metrics[f"group_{gid}_loss"].append(avg_g)

        print(f"  Step {step}: MLM loss = {avg_mlm:.4f} | NSP loss = {avg_nsp:.4f}")
        print(f"  Group counts: {group_counts}")

        del net
        torch.cuda.empty_cache()

    # --- Save metrics to CSV ---
    df = pd.DataFrame(all_metrics)
    df.insert(0, 'step', steps)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")

    # --- Plot per-group loss curves ---
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab10')
    for idx, (metric_name, values) in enumerate(sorted(all_metrics.items())):
        if metric_name == 'total_loss':
            continue
        plt.plot(
            steps, values,
            label=metric_name.replace("_", " "),
            color=cmap(idx % 10),
            alpha=0.7,
        )
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Per-group cross-entropy loss vs. training step")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Saved group loss plot to {output_png}")

    # --- Plot total MLM + NSP loss curves ---
    plt.figure(figsize=(10, 6))
    plt.plot(steps, all_metrics["total_loss"], linewidth=2, label="MLM Loss")
    plt.plot(steps, all_metrics["nsp_loss"],   linewidth=2, label="NSP Loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Total cross-entropy loss vs. training step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    total_loss_png = output_png.replace('.png', '_total_loss.png')
    plt.savefig(total_loss_png, dpi=150)
    print(f"Saved total loss plot to {total_loss_png}")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    main()
