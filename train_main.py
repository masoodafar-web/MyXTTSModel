#!/usr/bin/env python3
"""
Standalone training script mirroring MyXTTSTrain.ipynb (training-only).

Builds a complete XTTSConfig (model/data/training) and runs training
with checkpointing. Focuses only on training; no inference/extras.

Usage (basic):
  python train_only.py \
      --train-data ../dataset/dataset_train \
      --val-data   ../dataset/dataset_eval \
      --checkpoint-dir ./checkpoints

Optional overrides:
  --epochs 200 --batch-size 32 --lr 5e-5 --resume
"""

import os
import sys
import argparse
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
from myxtts.models.xtts import XTTS
from myxtts.training.trainer import XTTSTrainer
from myxtts.utils.commons import setup_logging, find_latest_checkpoint
from memory_optimizer import get_gpu_memory_info, get_recommended_settings


def build_config(
    batch_size: int = 32,
    grad_accum: int = 16,
    num_workers: int = 8,
    epochs: int = 200,
    lr: float = 5e-5,
    checkpoint_dir: str = "./checkpoints",
    text_dim: int = 256,
    decoder_dim: int = 512,
    max_attention_len: int = 256,
    enable_grad_checkpointing: bool = True,
    max_memory_fraction: float = 0.9,
) -> XTTSConfig:
    # Model configuration (memory-optimized defaults as in notebook)
    m = ModelConfig(
        text_encoder_dim=text_dim,
        text_encoder_layers=4,
        text_encoder_heads=4,
        text_vocab_size=256_256,

        audio_encoder_dim=text_dim,
        audio_encoder_layers=4,
        audio_encoder_heads=4,

        decoder_dim=decoder_dim,
        decoder_layers=6,
        decoder_heads=8,

        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024,

        # Language/tokenizer
        languages=[
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko",
        ],
        max_text_length=500,
        tokenizer_type="nllb",
        tokenizer_model="facebook/nllb-200-distilled-600M",

        # Memory optimizations
        enable_gradient_checkpointing=enable_grad_checkpointing,
        max_attention_sequence_length=max_attention_len,
        use_memory_efficient_attention=True,
    )

    # Training configuration (comprehensive parameters)
    t = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,

        optimizer="adamw",
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-6,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=grad_accum,
        max_memory_fraction=max_memory_fraction,

        warmup_steps=2000,
        scheduler="noam",
        scheduler_params={},

        mel_loss_weight=45.0,
        kl_loss_weight=1.0,
        duration_loss_weight=1.0,

        save_step=5000,
        checkpoint_dir=checkpoint_dir,
        val_step=1000,

        log_step=100,
        use_wandb=False,
        wandb_project="myxtts",

        multi_gpu=False,
        visible_gpus=None,
    )

    # Data configuration (comprehensive parameters)
    d = DataConfig(
        # Dataset split/subset controls
        train_subset_fraction=1.0,
        eval_subset_fraction=1.0,
        train_split=0.9,
        val_split=0.1,
        subset_seed=42,
        max_text_tokens=max_attention_len,

        # Audio/text processing
        sample_rate=22050,
        normalize_audio=True,
        trim_silence=True,
        text_cleaners=["english_cleaners"],
        language="en",
        add_blank=True,

        # Batching/workers and pipeline performance
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_buffer_size=12,
        shuffle_buffer_multiplier=30,
        enable_memory_mapping=True,
        cache_verification=True,
        prefetch_to_gpu=True,
        max_mel_frames=max_attention_len,
        enable_xla=True,
        enable_tensorrt=False,
        mixed_precision=True,
        pin_memory=True,
        persistent_workers=True,

        # Preprocessing/caching
        preprocessing_mode="precompute",
        use_tf_native_loading=True,
        enhanced_gpu_prefetch=True,
        optimize_cpu_gpu_overlap=True,

        # Dataset identity/paths are filled outside via CLI args
        dataset_path="",
        dataset_name="custom_dataset",
        metadata_train_file="metadata_train.csv",
        metadata_eval_file="metadata_eval.csv",
        wavs_train_dir="wavs",
        wavs_eval_dir="wavs",
    )

    return XTTSConfig(model=m, data=d, training=t)


def main():
    parser = argparse.ArgumentParser(description="MyXTTS training-only script (config-inlined)")
    # Make dataset paths optional with sensible defaults (matching the notebook)
    parser.add_argument(
        "--train-data",
        default="../dataset/dataset_train",
        help="Path to train subset root (default: ../dataset/dataset_train)"
    )
    parser.add_argument(
        "--val-data",
        default="../dataset/dataset_eval",
        help="Path to val subset root (default: ../dataset/dataset_eval)"
    )
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps (auto-tuned if omitted)"
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="(Deprecated) Resuming is automatic; kept for backwards compatibility"
    )
    parser.add_argument(
        "--reset-training",
        action="store_true",
        help="Ignore existing checkpoints and start training from scratch"
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Build full config
    gpu_available = bool(tf.config.list_physical_devices('GPU'))

    batch_flag = any(arg.startswith("--batch-size") for arg in sys.argv[1:])
    grad_flag = any(arg.startswith("--grad-accum") for arg in sys.argv[1:])

    recommended = None
    if gpu_available:
        try:
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                recommended = get_recommended_settings(gpu_info['total_memory'])
        except Exception as e:
            logger.debug(f"Could not determine GPU-based recommendations: {e}")

    # Auto-select batch size when user did not override it
    if recommended and not batch_flag:
        args.batch_size = recommended['batch_size']
        logger.info(
            f"Auto-selected batch_size={args.batch_size} based on GPU category: {recommended['description']}"
        )

    # Auto-tune gradient accumulation for better GPU saturation when not provided
    if args.grad_accum is None:
        if recommended and not grad_flag:
            args.grad_accum = recommended['gradient_accumulation_steps']
            logger.info(
                f"Auto-selected gradient accumulation = {args.grad_accum} based on GPU category: {recommended['description']}"
            )
        else:
            args.grad_accum = 1 if gpu_available else 16
            if gpu_available:
                logger.info("Auto-selected gradient accumulation = 1 for GPU saturation")
            else:
                logger.info("Auto-selected gradient accumulation = 16 for CPU fallback")

    if args.grad_accum > 1 and args.batch_size // args.grad_accum < 8:
        logger.warning(
            "Gradient accumulation is creating micro-batches smaller than 8 samples. "
            "This heavily limits GPU utilization. Increase --batch-size or lower --grad-accum."
        )

    text_dim = recommended['text_encoder_dim'] if recommended else 256
    decoder_dim = recommended['decoder_dim'] if recommended else 512
    max_attention_len = recommended['max_attention_sequence_length'] if recommended else 256
    enable_grad_ckpt = recommended['enable_gradient_checkpointing'] if recommended else True
    max_memory_fraction = recommended['max_memory_fraction'] if recommended else 0.9

    config = build_config(
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        text_dim=text_dim,
        decoder_dim=decoder_dim,
        max_attention_len=max_attention_len,
        enable_grad_checkpointing=enable_grad_ckpt,
        max_memory_fraction=max_memory_fraction,
    )

    # Instantiate model and trainer (optionally resume)
    resume_ckpt: Optional[str] = None
    if args.resume:
        logger.warning("--resume flag is deprecated; automatic checkpoint resumption is enabled by default.")

    if args.reset_training:
        logger.info("Reset requested — ignoring any existing checkpoints.")
    else:
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest:
            resume_ckpt = latest
            logger.info(f"Resuming from checkpoint: {resume_ckpt}")
        else:
            logger.info("No existing checkpoint found, starting fresh")

    model = XTTS(config.model)
    trainer = XTTSTrainer(config=config, model=model, resume_checkpoint=resume_ckpt)

    # Prepare datasets (will precompute caches when configured)
    train_ds, val_ds = trainer.prepare_datasets(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )

    # Train
    trainer.train(train_dataset=train_ds, val_dataset=val_ds, epochs=config.training.epochs)

    # Save a final checkpoint artifact for convenience
    try:
        final_base = os.path.join(config.training.checkpoint_dir, "final_model")
        trainer.save_checkpoint(final_base)
        logger.info(f"Final model checkpoint saved: {final_base}")
    except Exception as e:
        logger.warning(f"Could not save final checkpoint: {e}")


if __name__ == "__main__":
    main()
