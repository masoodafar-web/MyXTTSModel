#!/usr/bin/env python3
"""
Stable MyXTTS Training Script - NaN Loss Fix

This script includes comprehensive fixes for NaN loss issues.
اسکریپت تمرین پایدار با رفع مشکل NaN شدن loss
"""

import os
import sys
import argparse
import tensorflow as tf
from typing import Optional

# Configure TensorFlow for stability
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.run_functions_eagerly(False)

# Disable mixed precision by default for stability
policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)

from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
from myxtts.models.xtts import XTTS
from myxtts.training.trainer import XTTSTrainer
from myxtts.utils.commons import setup_logging, find_latest_checkpoint


def create_stable_config(model_size: str = "tiny") -> XTTSConfig:
    """Create stable configuration that prevents NaN losses."""
    
    # Model configuration with conservative settings
    m = ModelConfig(
        text_encoder_dim=256,
        text_encoder_layers=4,
        text_encoder_heads=4,
        text_vocab_size=32000,
        
        audio_encoder_dim=256,
        audio_encoder_layers=4,
        audio_encoder_heads=4,
        
        decoder_dim=768,
        decoder_layers=8,
        decoder_heads=12,
        
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        
        speaker_embedding_dim=256,
        use_voice_conditioning=True,
        voice_conditioning_layers=2,
        
        enable_gradient_checkpointing=True,
        max_attention_sequence_length=256,
        use_memory_efficient_attention=True,
        
        languages=["en", "es", "fr", "de", "it", "pt"],
        max_text_length=320,
    )
    
    # Training configuration with stability fixes
    t = TrainingConfig(
        epochs=500,
        learning_rate=1e-5,  # MUCH LOWER learning rate
        
        optimizer="adamw",  
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-7,  # Lower weight decay
        gradient_clip_norm=0.3,  # MUCH LOWER gradient clipping
        gradient_accumulation_steps=4,
        
        warmup_steps=3000,  # Longer warmup
        scheduler="noam",   # Simple scheduler
        
        # BALANCED LOSS WEIGHTS (key fix)
        mel_loss_weight=1.0,      # Reduced from 2.5
        kl_loss_weight=0.5,       # Reduced from 1.8
        duration_loss_weight=0.1,
        pitch_loss_weight=0.05,
        energy_loss_weight=0.05,
        
        # Voice cloning weights (much lower)
        voice_similarity_loss_weight=0.5,
        speaker_classification_loss_weight=0.3,
        voice_reconstruction_loss_weight=0.4,
        
        # STABILITY FEATURES
        use_adaptive_loss_weights=False,  # Disable adaptive weights
        loss_smoothing_factor=0.05,       # Stronger smoothing
        max_loss_spike_threshold=1.2,     # Lower threshold
        gradient_norm_threshold=1.0,      # Lower gradient monitoring
        
        # DISABLE PROBLEMATIC FEATURES
        use_label_smoothing=False,
        use_huber_loss=False,
        
        # More frequent monitoring
        save_step=1000,
        val_step=500,
        log_step=25,
        
        early_stopping_patience=20,
        early_stopping_min_delta=0.001,
        
        checkpoint_dir="./checkpoints_stable",
        use_wandb=False,
    )
    
    # Data configuration
    d = DataConfig(
        batch_size=16,  # Smaller batch size
        num_workers=4,
        sample_rate=22050,
        normalize_audio=True,
        trim_silence=True,
        add_blank=True,
        mixed_precision=False,  # Disable mixed precision
        
        dataset_path="",
        dataset_name="custom_dataset",
        metadata_train_file="metadata_train.csv",
        metadata_eval_file="metadata_eval.csv",
    )
    
    return XTTSConfig(model=m, data=d, training=t)


def main():
    parser = argparse.ArgumentParser(description="Stable MyXTTS Training (NaN Loss Fix)")
    parser.add_argument("--train-data", default="../dataset/dataset_train")
    parser.add_argument("--val-data", default="../dataset/dataset_eval")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_stable")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--model-size", default="tiny", choices=["tiny", "small", "normal"])
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🔧 Starting STABLE training with NaN loss fixes...")
    
    # Create stable configuration
    config = create_stable_config(args.model_size)
    config.training.epochs = args.epochs
    config.training.checkpoint_dir = args.checkpoint_dir
    
    # Log key stability settings
    logger.info("="*50)
    logger.info("🛡️  STABILITY FIXES APPLIED:")
    logger.info(f"   Learning rate: {config.training.learning_rate} (reduced)")
    logger.info(f"   Mel loss weight: {config.training.mel_loss_weight} (balanced)")
    logger.info(f"   Gradient clip: {config.training.gradient_clip_norm} (conservative)")
    logger.info(f"   Mixed precision: {config.data.mixed_precision} (disabled)")
    logger.info(f"   Batch size: {config.data.batch_size} (smaller)")
    logger.info(f"   Warmup steps: {config.training.warmup_steps} (longer)")
    logger.info("="*50)
    
    # Check for existing checkpoints
    resume_ckpt = None
    latest = find_latest_checkpoint(args.checkpoint_dir)
    if latest:
        resume_ckpt = latest
        logger.info(f"Resuming from: {resume_ckpt}")
    
    # Initialize model and trainer
    model = XTTS(config.model)
    trainer = XTTSTrainer(config=config, model=model, resume_checkpoint=resume_ckpt)
    
    # Prepare datasets
    train_ds, val_ds = trainer.prepare_datasets(args.train_data, args.val_data)
    
    # Start training with monitoring
    logger.info("🚀 Starting stable training...")
    try:
        trainer.train(train_dataset=train_ds, val_dataset=val_ds, epochs=config.training.epochs)
        logger.info("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    # Save final model
    final_path = os.path.join(config.training.checkpoint_dir, "final_model_stable")
    trainer.save_checkpoint(final_path)
    logger.info(f"💾 Final model saved: {final_path}")


if __name__ == "__main__":
    main()
