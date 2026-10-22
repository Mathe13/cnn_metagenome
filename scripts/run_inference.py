#!/usr/bin/env python3
"""
CLI entrypoint for clean forward inference with 1D-CNN viral models.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

try:
    from viral_cnn.data.encoding import NUCLEOTIDES, NUM_FEATURES
    from viral_cnn.models.builder import build_model, MODEL_REGISTRY
    from viral_cnn.models.registry import (
        list_model_variants,
        get_model_spec,
        load_model_checkpoint,
    )
    from viral_cnn.inference.engine import predict_sequence
except ImportError as e:
    sys.stderr.write(f"ImportError: Failed to import viral_cnn modules: {e}\n")
    sys.exit(2)


def validate_dna_sequence(seq: str) -> None:
    if not seq or not seq.strip():
        raise ValueError("Input sequence cannot be empty.")
    # Valid DNA characters: canonical A, C, G, T, ambiguous N, R, Y, S, W, K, M, B, D, H, V, gaps -, padding _
    valid_chars = set("ACGTNRYWSKMBDHV-_")
    invalid_chars = set(seq.upper().strip()) - valid_chars
    if invalid_chars:
        raise ValueError(
            f"Sequence contains invalid non-DNA characters: {sorted(list(invalid_chars))}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run clean forward inference on viral DNA sequence using 1D-CNN."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", type=str, help="Raw DNA nucleotide sequence string")
    group.add_argument("--fasta", type=str, help="Path to input FASTA file")

    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help=f"Registered model variant name. Supported: {list_model_variants()}",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help=f"Model architecture type. Supported: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of output taxonomic classes (default: 13683 for Species, 2702 for Genus).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: 'cpu').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model weights checkpoint (.pt file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic initialization when checkpoint is absent (default: 42).",
    )

    args = parser.parse_args()

    # Resolve variant or defaults
    if args.model_variant is not None:
        try:
            spec = get_model_spec(args.model_variant)
        except ValueError as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.exit(2)
        model_type = args.model_type if args.model_type is not None else spec.architecture
        num_classes = args.num_classes if args.num_classes is not None else spec.num_classes
        default_ckpt = spec.checkpoint_rel_path
    else:
        model_type = args.model_type if args.model_type is not None else "resnet50_1d"
        num_classes = args.num_classes if args.num_classes is not None else 13683
        default_ckpt = None

    # Model type validation
    if model_type.lower().strip() not in MODEL_REGISTRY:
        sys.stderr.write(
            f"Error: Unknown model-type '{model_type}'. Supported: {list(MODEL_REGISTRY.keys())}\n"
        )
        sys.exit(2)

    # Sequence extraction and validation
    if args.sequence is not None:
        seq = args.sequence.strip()
    elif args.fasta is not None:
        fasta_path = Path(args.fasta)
        if not fasta_path.exists():
            sys.stderr.write(f"Error: FASTA file not found: {fasta_path}\n")
            sys.exit(2)
        lines = fasta_path.read_text(encoding="utf-8").splitlines()
        seq = "".join(l.strip() for l in lines if l and not l.startswith(">"))
    else:
        sys.stderr.write("Error: Either --sequence or --fasta must be provided.\n")
        sys.exit(2)

    try:
        validate_dna_sequence(seq)
    except ValueError as e:
        sys.stderr.write(f"Error: Invalid sequence input: {e}\n")
        sys.exit(2)

    # Device selection
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("Warning: CUDA requested but not available. Falling back to CPU.\n")
        device_str = "cpu"

    # Deterministic model initialization
    torch.manual_seed(args.seed)
    model = build_model(
        model_type=model_type,
        num_classes=num_classes,
        max_seq_len=512,
        dropout_rate=0.0,
    )

    checkpoint_to_load = args.checkpoint
    if checkpoint_to_load is None and default_ckpt is not None:
        # Check if default relative checkpoint exists locally or under DATA_ROOT
        data_root = os.environ.get("DATA_ROOT", ".")
        candidate = Path(data_root) / default_ckpt
        if candidate.exists():
            checkpoint_to_load = str(candidate)

    if checkpoint_to_load is not None:
        ckpt_path = Path(checkpoint_to_load)
        if not ckpt_path.exists():
            sys.stderr.write(f"Error: Checkpoint file not found: {ckpt_path}\n")
            sys.exit(2)
        try:
            load_model_checkpoint(
                model=model,
                checkpoint=ckpt_path,
                expected_classes=num_classes,
                map_location=device_str,
            )
        except Exception as e:
            sys.stderr.write(f"Error: Failed to load checkpoint: {e}\n")
            sys.exit(2)

    result = predict_sequence(
        model=model,
        sequence=seq,
        device=device_str,
        max_len=512,
        top_k=5,
    )

    output = {
        "status": "success",
        "model_type": model_type,
        "num_classes": num_classes,
        "channels": result.channels,
        "predicted_class_index": result.predicted_class_index,
        "predicted_csv_id": result.predicted_class_index + 1,
        "confidence": round(result.confidence, 6),
        "top5_class_indices": result.top5_class_indices,
        "top5_classes": result.top5_class_indices,
        "top5_csv_ids": [idx + 1 for idx in result.top5_class_indices],
        "top5_confidences": [round(c, 6) for c in result.top5_confidences],
        "input_length": result.input_length,
        "device": device_str,
    }

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
