#!/usr/bin/env python3
"""
High-Performance MyXTTS Training Script - Optimized for Fast Convergence

This script implements comprehensive performance optimizations to address slow loss convergence:
- Aggressive learning rate scheduling with proper warmup
- Optimized batch size and gradient accumulation for faster convergence
- Enhanced data pipeline with GPU prefetching and TensorRT acceleration
- Improved model architecture with increased capacity for better learning
- Advanced training techniques including mixed precision and gradient checkpointing

Key Performance Improvements:
- 2x higher default learning rate (1e-4 vs 5e-5)
- Larger effective batch sizes (384 vs 512)
- Cosine LR scheduling with warmup for better convergence
- TensorRT and XLA acceleration enabled
- Enhanced data prefetching and GPU utilization
- More frequent validation monitoring

Usage (optimized defaults):
  python train_main.py \
      --train-data ../dataset/dataset_train \
      --val-data   ../dataset/dataset_eval \
      --checkpoint-dir ./checkpoints

Advanced optimizations:
  python train_main.py \
      --aggressive-lr \
      --batch-size 64 \
      --grad-accum 6 \
      --lr 2e-4 \
      --max-gpu-memory 0.95

Performance flags:
  --aggressive-lr     : Enable more aggressive training optimizations
  --max-gpu-memory    : GPU memory fraction (default: 0.9)
  --enable-amp        : Automatic mixed precision (default: True)
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
    batch_size: int = 48,
    grad_accum: int = 8,
    num_workers: int = 16,
    epochs: int = 200,
    lr: float = 1e-4,
    checkpoint_dir: str = "./checkpoints",
) -> XTTSConfig:
    # Model configuration (memory-optimized defaults as in notebook)
    m = ModelConfig(
        text_encoder_dim=512,  # Increased for better representation
        text_encoder_layers=6,  # More layers for better learning
        text_encoder_heads=8,   # More heads for better attention
        text_vocab_size=256_256,

        audio_encoder_dim=512,  # Increased for better representation
        audio_encoder_layers=6,  # More layers for better learning
        audio_encoder_heads=8,   # More heads for better attention

        decoder_dim=768,        # Increased for better capacity
        decoder_layers=8,       # More layers for better modeling
        decoder_heads=12,       # More heads for better attention

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

        # Memory optimizations for training efficiency
        enable_gradient_checkpointing=True,
        max_attention_sequence_length=512,  # Increased for better context
        use_memory_efficient_attention=True,
    )

    # Training configuration (comprehensive parameters)
    t = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,

        optimizer="adamw",
        beta1=0.9,
        beta2=0.98,  # Optimized for faster convergence
        eps=1e-9,    # More stable training
        weight_decay=1e-4,  # Increased for better regularization
        gradient_clip_norm=0.5,  # Reduced for stability
        gradient_accumulation_steps=grad_accum,

        warmup_steps=max(1000, epochs // 10),  # Dynamic warmup based on epochs
        scheduler="cosine_with_warmup",  # Better convergence than noam
        scheduler_params={
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.01,
            "cycle_length": epochs
        },

        mel_loss_weight=45.0,
        kl_loss_weight=1.0,
        duration_loss_weight=1.0,

        save_step=max(2000, epochs * 10),  # Less frequent saves for faster training
        checkpoint_dir=checkpoint_dir,
        val_step=max(500, epochs * 2),     # More frequent validation for monitoring

        log_step=50,   # More frequent logging for monitoring
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
        prefetch_buffer_size=max(16, batch_size // 2),  # Dynamic buffer sizing
        shuffle_buffer_multiplier=50,  # Increased for better randomization
        enable_memory_mapping=True,
        cache_verification=True,
        prefetch_to_gpu=True,
        max_mel_frames=1024,
        enable_xla=True,
        enable_tensorrt=True,  # Enable TensorRT for GPU acceleration
        mixed_precision=True,
        pin_memory=True,
        persistent_workers=True,

        # Preprocessing/caching - CRITICAL FOR PERFORMANCE
        preprocessing_mode="precompute",  # Force precompute for maximum speed
        use_tf_native_loading=True,
        enhanced_gpu_prefetch=True,
        optimize_cpu_gpu_overlap=True,

        # Advanced performance optimizations
        enable_memory_pin=True,
        async_data_loading=True,
        prefetch_factor=4,  # Aggressive prefetching
        multiprocess_loading=True,

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
    parser.add_argument("--train-data", required=True, help="Path to train subset root (e.g., ../dataset/dataset_train)")
    parser.add_argument("--val-data", required=True, help="Path to val subset root (e.g., ../dataset/dataset_eval)")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=16, help="Data loader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint if available")
    
    # Performance optimization flags
    parser.add_argument("--aggressive-lr", action="store_true", help="Use more aggressive learning rate scheduling")
    parser.add_argument("--max-gpu-memory", type=float, default=0.9, help="Maximum GPU memory fraction to use")
    parser.add_argument("--enable-amp", action="store_true", default=True, help="Enable automatic mixed precision")
    parser.add_argument("--fast-convergence", action="store_true", help="Enable all fast convergence optimizations")
    
    args = parser.parse_args()

    logger = setup_logging()
    
    # Enable fast convergence mode automatically for aggressive optimizations
    if args.fast_convergence:
        args.aggressive_lr = True
        args.lr = max(args.lr, 2e-4)  # Ensure minimum aggressive LR
        args.batch_size = max(args.batch_size, 64)  # Ensure larger batch size
        logger.info("🔥 FAST CONVERGENCE MODE ENABLED - Maximum performance optimizations active")
    
    # Set aggressive performance environment variables
    if args.aggressive_lr:
        logger.info("🚀 Enabling aggressive performance optimizations")
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
        os.environ["TF_ENABLE_TENSOR_FLOAT_32"] = "1"
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_GPU_THREAD_COUNT"] = "2"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Use static memory allocation for performance
        
    # Configure GPU memory growth for optimal performance
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit if specified
                if args.max_gpu_memory < 1.0:
                    memory_limit = int(24000 * args.max_gpu_memory)  # Assume 24GB GPU
                    tf.config.experimental.set_memory_limit(gpu, memory_limit)
                    logger.info(f"🎯 GPU memory limit set to {memory_limit}MB ({args.max_gpu_memory*100:.1f}%)")
    except Exception as e:
        logger.warning(f"GPU configuration failed: {e}")

    # Build full config with performance optimizations
    config = build_config(
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr * (2.0 if args.aggressive_lr else 1.0),  # Increase LR if aggressive mode
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

    logger.info("🔥 Initializing model and trainer with performance optimizations...")
    model = XTTS(config.model)
    trainer = XTTSTrainer(config=config, model=model, resume_checkpoint=resume_ckpt)

    # Prepare datasets (will precompute caches when configured)
    logger.info("📊 Preparing optimized datasets with aggressive caching...")
    train_ds, val_ds = trainer.prepare_datasets(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )

    # Performance logging
    effective_batch_size = args.batch_size * args.grad_accum
    logger.info(f"🚀 Training Configuration:")
    logger.info(f"   • Batch Size: {args.batch_size} (effective: {effective_batch_size})")
    logger.info(f"   • Learning Rate: {config.training.learning_rate:.2e}")
    logger.info(f"   • Workers: {args.num_workers}")
    logger.info(f"   • Gradient Accumulation: {args.grad_accum}")
    logger.info(f"   • Scheduler: {config.training.scheduler}")
    logger.info(f"   • Mixed Precision: {config.data.mixed_precision}")
    logger.info(f"   • TensorRT: {config.data.enable_tensorrt}")

    # Train with performance monitoring
    import time
    start_time = time.time()
    logger.info("🏁 Starting optimized training process...")
    
    trainer.train(train_dataset=train_ds, val_dataset=val_ds, epochs=config.training.epochs)
    
    end_time = time.time()
    training_hours = (end_time - start_time) / 3600
    logger.info(f"✅ Training completed in {training_hours:.2f} hours")

    # Save a final checkpoint artifact for convenience
    try:
        final_base = os.path.join(config.training.checkpoint_dir, "final_model")
        trainer.save_checkpoint(final_base)
        logger.info(f"Final model checkpoint saved: {final_base}")
    except Exception as e:
        logger.warning(f"Could not save final checkpoint: {e}")


if __name__ == "__main__":
    main()

