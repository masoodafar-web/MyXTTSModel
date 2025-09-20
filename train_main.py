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

# Import other modules with error handling for missing dependencies
try:
    from myxtts.models.xtts import XTTS
    from myxtts.training.trainer import XTTSTrainer
    from myxtts.utils.commons import setup_logging, find_latest_checkpoint
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies are missing: {e}")
    print("Full training requires dependencies: pip install -r requirements.txt")
    DEPENDENCIES_AVAILABLE = False
    
    # Create dummy functions for missing dependencies
    def setup_logging():
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def find_latest_checkpoint(path):
        return None


def build_config(
    batch_size: int = 32,
    grad_accum: int = 16,
    num_workers: int = 8,
    epochs: int = 200,
    lr: float = 5e-5,
    checkpoint_dir: str = "./checkpoints",
    # GPU optimization parameters
    use_tf_native_loading: bool = True,
    enhanced_gpu_prefetch: bool = True,
    optimize_cpu_gpu_overlap: bool = True,
    prefetch_buffer_size: int = 12,
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
        prefetch_buffer_size=prefetch_buffer_size,  # GPU-optimized buffer size
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

        # Preprocessing/caching with GPU optimizations
        preprocessing_mode="precompute",
        use_tf_native_loading=use_tf_native_loading,
        enhanced_gpu_prefetch=enhanced_gpu_prefetch,
        optimize_cpu_gpu_overlap=optimize_cpu_gpu_overlap,

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
    
    # GPU optimization options (NEW - for fixing CPU bottlenecks)
    parser.add_argument("--disable-tf-native-loading", action="store_true", 
                       help="Disable TensorFlow-native file loading optimization (not recommended)")
    parser.add_argument("--disable-gpu-prefetch", action="store_true",
                       help="Disable enhanced GPU prefetching (not recommended)")
    parser.add_argument("--disable-cpu-gpu-overlap", action="store_true",
                       help="Disable CPU-GPU overlap optimizations (not recommended)")
    parser.add_argument("--prefetch-buffer-size", type=int, default=12,
                       help="Prefetch buffer size for GPU optimization (default: 12)")
    
    args = parser.parse_args()

    if not DEPENDENCIES_AVAILABLE:
        logger = setup_logging()
        logger.error("Training requires full dependencies. Please install requirements: pip install -r requirements.txt")
        return

    logger = setup_logging()
    
    # CRITICAL: GPU Setup Validation (addresses the Persian user's issue)
    logger.info("=" * 60)
    logger.info("🔍 CHECKING GPU SETUP (resolving CPU usage issue)...")
    logger.info("=" * 60)
    
    from myxtts.utils.commons import check_gpu_setup
    gpu_success, device, recommendations = check_gpu_setup()
    
    if not gpu_success:
        logger.error("❌ GPU SETUP ISSUES DETECTED:")
        for i, rec in enumerate(recommendations, 1):
            logger.error(f"   {i}. {rec}")
        
        logger.error("")
        logger.error("🚨 THIS IS WHY CPU IS BEING USED INSTEAD OF GPU!")
        logger.error("   The system cannot access GPU for computation.")
        logger.error("")
        logger.error("🔧 IMMEDIATE SOLUTIONS:")
        logger.error("   • For local setup: Install NVIDIA drivers + CUDA + TensorFlow-GPU")
        logger.error("   • For cloud/server: Enable GPU instance or add GPU acceleration")
        logger.error("   • For development: Use CPU mode temporarily (much slower)")
        logger.error("")
        
        # Ask user if they want to continue with CPU
        try:
            import sys
            if sys.stdin.isatty():  # Interactive terminal
                response = input("Continue with CPU training (much slower)? [y/N]: ").lower()
                if response not in ['y', 'yes']:
                    logger.info("Training cancelled. Fix GPU setup and try again.")
                    return
            else:
                logger.warning("🔄 Non-interactive mode: proceeding with CPU (very slow!)")
        except (KeyboardInterrupt, EOFError):
            logger.info("\nTraining cancelled.")
            return
    else:
        logger.info("✅ GPU setup validation successful!")
        logger.info(f"   Using device: {device}")
    
    logger.info("=" * 60)

    # Build full config with GPU optimizations
    config = build_config(
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        # Apply GPU optimization settings
        use_tf_native_loading=not args.disable_tf_native_loading,
        enhanced_gpu_prefetch=not args.disable_gpu_prefetch,
        optimize_cpu_gpu_overlap=not args.disable_cpu_gpu_overlap,
        prefetch_buffer_size=args.prefetch_buffer_size,
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

    logger.info("Starting XTTS training...")
    logger.info(f"Train data path: {args.train_data}")
    logger.info(f"Val data path: {args.val_data}")
    logger.info(f"Batch size: {config.data.batch_size}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"Num workers: {config.data.num_workers}")
    logger.info(f"Compute device: {device}")
    
    # Log GPU optimization settings
    logger.info("GPU Optimization Settings:")
    logger.info(f"   TF Native Loading: {config.data.use_tf_native_loading}")
    logger.info(f"   Enhanced GPU Prefetch: {config.data.enhanced_gpu_prefetch}")
    logger.info(f"   CPU-GPU Overlap: {config.data.optimize_cpu_gpu_overlap}")
    logger.info(f"   Prefetch Buffer Size: {config.data.prefetch_buffer_size}")

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
