# LLM_TEST

A minimal GPT-style Large Language Model (LLM) implemented from scratch using PyTorch.

## Features

- Transformer decoder architecture with multi-head self-attention
- Causal (auto-regressive) language modelling
- Character-level tokenizer (easy to swap for a BPE tokenizer)
- Training script with validation, checkpointing and cosine-LR schedule
- Text generation with temperature and top-k sampling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python train.py --data data.txt --epochs 10 --batch_size 32
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Path to training text file |
| `--output_dir` | `checkpoints` | Directory to save checkpoints |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--seq_len` | `128` | Context window length |
| `--d_model` | `256` | Model hidden dimension |
| `--num_heads` | `8` | Number of attention heads |
| `--num_layers` | `6` | Number of transformer layers |
| `--lr` | `3e-4` | Learning rate |

### Generate

```bash
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 40
```

## Project Structure

```
├── model.py        # LLM model architecture
├── train.py        # Training loop & character tokenizer
├── generate.py     # Inference / text generation
├── test_model.py   # Unit tests
└── requirements.txt
```

## Run Tests

```bash
pip install pytest
pytest test_model.py -v
```
