"""
Text generation script for the LLM model.

Usage:
    python generate.py --checkpoint <path> --prompt <text> [options]

Example:
    python generate.py --checkpoint checkpoints/best_model.pt \
        --prompt "Once upon a time" --max_new_tokens 200 --temperature 0.8
"""

import argparse
import os
import torch

from model import LLM
from train import CharTokenizer


def generate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = checkpoint["model_config"]

    model = LLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.checkpoint), "tokenizer.json")
    tokenizer = CharTokenizer.load(tokenizer_path)

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt contains characters not seen during training.")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with the LLM model.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.pt).")
    parser.add_argument("--prompt", required=True, help="Text prompt to continue.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (optional).")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
