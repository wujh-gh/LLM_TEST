"""
Training script for the LLM model.

Usage:
    python train.py --data <text_file> [options]

Example:
    python train.py --data data.txt --epochs 5 --batch_size 32
"""

import argparse
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import LLM


# ---------------------------------------------------------------------------
# Tokenizer (character-level, kept minimal for clarity)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        unknown = [ch for ch in text if ch not in self.vocab]
        if unknown:
            import warnings
            warnings.warn(
                f"Skipping {len(unknown)} unknown character(s): "
                + repr("".join(sorted(set(unknown))))
            )
        return [self.vocab[ch] for ch in text if ch in self.vocab]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv_vocab.get(i, "") for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.vocab = data["vocab"]
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        tok.vocab_size = len(tok.vocab)
        return tok


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Sliding-window dataset over a token sequence."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load / prepare data
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    token_ids = tokenizer.encode(text)
    split = int(len(token_ids) * 0.9)
    train_ids, val_ids = token_ids[:split], token_ids[split:]

    train_dataset = TextDataset(train_ids, args.seq_len)
    val_dataset = TextDataset(val_ids, args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Build model
    model = LLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # --- Validation ---
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        scheduler.step()

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_config": {
                        "vocab_size": tokenizer.vocab_size,
                        "d_model": args.d_model,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "max_seq_len": args.seq_len,
                        "dropout": args.dropout,
                    },
                },
                checkpoint_path,
            )
            print(f"  → Saved best model (val loss {val_loss:.4f})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the LLM model.")
    parser.add_argument("--data", required=True, help="Path to the training text file.")
    parser.add_argument("--output_dir", default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
