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
import argparse
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
from myxtts.models.xtts import XTTS
from myxtts.training.trainer import XTTSTrainer
from myxtts.utils.commons import setup_logging, find_latest_checkpoint


def build_config(
    batch_size: int = 32,
    grad_accum: int = 16,
    num_workers: int = 8,
    epochs: int = 200,
    lr: float = 5e-5,
    checkpoint_dir: str = "./checkpoints",
) -> XTTSConfig:
    # Model configuration (memory-optimized defaults as in notebook)
    m = ModelConfig(
        text_encoder_dim=256,
        text_encoder_layers=4,
        text_encoder_heads=4,
        text_vocab_size=256_256,

        audio_encoder_dim=256,
        audio_encoder_layers=4,
        audio_encoder_heads=4,

        decoder_dim=512,
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
        enable_gradient_checkpointing=True,
        max_attention_sequence_length=256,
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
        max_mel_frames=1024,
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
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint if available")
    args = parser.parse_args()

    logger = setup_logging()

    # Build full config
    config = build_config(
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Instantiate model and trainer (optionally resume)
    resume_ckpt: Optional[str] = None
    if args.resume:
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
