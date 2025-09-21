#!/usr/bin/env python3
"""Simple inference entry point for the MyXTTS model.

This script mirrors the training setup in ``train_main.py`` and provides a
straightforward way to run text-to-speech inference against the same model
configuration. It loads the latest checkpoint by default (or a user-provided
checkpoint), synthesizes audio for the supplied text, and optionally saves the
mel spectrogram.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf  # noqa: F401  # Ensures TF is initialised before model import

from train_main import build_config
from myxtts.config.config import XTTSConfig
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import setup_logging, find_latest_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference for the MyXTTS model using the training defaults."
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config_path",
        help="Optional YAML config. When omitted, uses build_config from train_main.py",
    )
    parser.add_argument(
        "--checkpoint",
        help="Exact checkpoint prefix to load (e.g. checkpoints/checkpoint_5000).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpointsmain",
        help="Directory to search when --checkpoint is not provided (default: ./checkpointsmain).",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument(
        "--text-file",
        help="Path to a text file. Entire file contents are synthesized.",
    )
    parser.add_argument(
        "--reference-audio",
        help="Optional reference audio (wav/flac) for voice conditioning.",
    )
    parser.add_argument(
        "--language",
        help="Language code (default: value from config).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation (default: 1.0).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum mel length to generate (default: 1000).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="inference_outputs/output.wav",
        help="Output path for synthesized audio (default: inference_outputs/output.wav).",
    )
    parser.add_argument(
        "--save-mel",
        help="Optional path to save the generated mel spectrogram as .npy.",
    )
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    """Load text either from CLI or from a file."""
    if args.text is not None:
        return args.text.strip()

    with open(args.text_file, "r", encoding="utf-8") as f:
        contents = f.read().strip()

    if not contents:
        raise ValueError("Provided text file is empty.")

    return contents


def load_config(config_path: Optional[str], checkpoint_dir: str) -> XTTSConfig:
    """Load configuration from YAML or reuse the training build_config."""
    if config_path:
        config = XTTSConfig.from_yaml(config_path)
    else:
        config = build_config(checkpoint_dir=checkpoint_dir)

    # Ensure checkpoint directory is up to date regardless of source
    config.training.checkpoint_dir = checkpoint_dir
    return config


def resolve_checkpoint(args: argparse.Namespace, logger) -> str:
    """Resolve which checkpoint to load."""
    if args.checkpoint:
        checkpoint_prefix = args.checkpoint
        if not os.path.exists(f"{checkpoint_prefix}_model.weights.h5"):
            raise FileNotFoundError(
                f"Model weights not found at {checkpoint_prefix}_model.weights.h5"
            )
        return checkpoint_prefix

    checkpoint_prefix = find_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint_prefix:
        raise FileNotFoundError(
            f"No checkpoints found in {args.checkpoint_dir}. Provide --checkpoint explicitly."
        )

    logger.info(f"Using latest checkpoint: {checkpoint_prefix}")
    return checkpoint_prefix


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file if it doesn't exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    logger = setup_logging()

    # Log accelerator availability (useful for debugging inference speed)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPUs available: %s", ", ".join(gpu.name for gpu in gpus))
    else:
        logger.info("No GPU detected; inference will run on CPU.")

    text = load_text(args)
    config = load_config(args.config_path, args.checkpoint_dir)
    checkpoint_prefix = resolve_checkpoint(args, logger)

    # Instantiate inference engine
    inference_engine = XTTSInference(
        config=config,
        checkpoint_path=checkpoint_prefix,
    )

    # Run synthesis
    result = inference_engine.synthesize(
        text=text,
        reference_audio=args.reference_audio,
        language=args.language,
        max_length=args.max_length,
        temperature=args.temperature,
    )

    # Persist outputs
    ensure_parent_dir(args.output)
    inference_engine.save_audio(result["audio"], args.output, result.get("sample_rate"))

    if args.save_mel:
        ensure_parent_dir(args.save_mel)
        np.save(args.save_mel, result["mel_spectrogram"])
        logger.info("Saved mel spectrogram to %s", args.save_mel)

    logger.info(
        "Synthesis complete. Duration: %.2fs | Sample rate: %d",
        len(result["audio"]) / result["sample_rate"],
        result["sample_rate"],
    )


if __name__ == "__main__":
    main()
