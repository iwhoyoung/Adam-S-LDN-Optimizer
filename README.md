# Adam-S-LDN-Optimizer

A lightweight variant of the Adam optimization algorithm, requiring half the additional memory and achieving faster computational efficiency.

---

## Features

- Fine-tune OpenCLIP / SigLIP models on custom CSV datasets
- Plug-and-play optimizer selection via `--opt` flag
- Per-epoch wall-clock time logging (console, TensorBoard, W&B)
- Full support for distributed training (DDP / Horovod)
- Compatible with AMP, gradient checkpointing, and `torch.compile`

---

## Requirements

- Python >= 3.9
- CUDA 12.x
- PyTorch 2.8.0

**Core dependencies**

| Package | Version |
|---|---|
| torch | 2.8.0 |
| torchvision | 0.23.0 |
| numpy | 2.2.6 |
| timm | 1.0.26 |
| transformers | 4.56.0 |
| tokenizers | 0.22.0 |
| huggingface-hub | 0.34.4 |
| datasets | 4.0.0 |
| tensorboard | 2.20.0 |
| pillow | 11.3.0 |
| webdataset | 1.0.2 |
| ftfy | 6.3.1 |

> Full dependency list: see [`requirements.txt`](requirements.txt)

---

## Installation

```bash
git clone https://github.com/your-username/Adam-S-LDN-Optimizer.git
cd Adam-S-LDN-Optimizer
pip install -r requirements.txt
pip install -e ".[training]"
```

## Quick Start
cd src && python -m open_clip_train.main \
  --dataset-type csv \
  --train-data "/path/to/train.csv" \
  --csv-img-key filepath \
  --csv-caption-key caption \
  --csv-separator "," \
  --model ViT-B-16-SigLIP-i18n-256 \
  --pretrained /path/to/open_clip_pytorch_model.bin \
  --warmup 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --wd 0.1 \
  --epochs 32 \
  --workers 8 \
  --opt adamw \
  --report-to tensorboard \
  --logs ./logs \
  --name my_experiment



## Optimizers
Select an optimizer with --opt. All optimizers automatically apply
zero weight decay to gain/bias-like parameters (norms, biases, logit_scale).

--opt value	Class	Notes
adamw	torch.optim.AdamW	Default
sgd	torch.optim.SGD	--beta1/2/eps are ignored
adam_s	Adam_s	Custom
adam_s_ldn	Adam_s_ldn	Custom
adam_ldn	Adam_ldn	Custom
sgd_s_ldn	Sgd_s_ldn	Custom
timm/<name>	timm optimizer	e.g. timm/lamb
Relevant flags: --lr, --wd, --beta1, --beta2, --eps

## Key Arguments
Argument	Default	Description
--model	—	OpenCLIP model name
--pretrained	—	Path or HuggingFace tag for pretrained weights
--train-data	—	Path to training CSV
--val-data	—	Path to validation CSV (optional)
--batch-size	64	Per-GPU batch size
--epochs	32	Total training epochs
--lr	5e-4	Peak learning rate
--wd	0.2	Weight decay
--warmup	10000	LR warmup steps
--lr-scheduler	cosine	cosine | const | const-cooldown
--opt	adamw	Optimizer (see table above)
--precision	amp	fp32 | fp16 | amp
--workers	4	DataLoader worker processes
--grad-checkpointing	off	Enable gradient checkpointing
--lock-image	off	Freeze image encoder
--lock-text	off	Freeze text encoder
--resume	—	Resume from checkpoint path, or latest
--report-to	—	tensorboard | wandb | all
--logs	./logs	Root directory for logs and checkpoints
--name	auto	Experiment name (auto-generated if not set)
