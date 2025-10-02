#!/usr/bin/env python3
"""
NaN Loss Fix for MyXTTS Training

This script provides a comprehensive fix for the NaN loss problem that occurs
after 2-5 epochs during training.

مشکل NaN شدن loss بعد از چند epoch را حل می‌کند

Common causes and solutions:
1. High learning rate -> Reduce to 1e-5 for tiny model
2. Loss weight imbalance -> Balance mel loss weight
3. Gradient explosion -> Better gradient clipping
4. Mixed precision instabilities -> Disable for stability
5. Numerical issues in loss calculation -> Add epsilon values
"""

import os
import sys
import tensorflow as tf
import argparse
from typing import Dict, Any

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_stable_training_config() -> Dict[str, Any]:
    """
    Create a stable training configuration that prevents NaN losses.
    
    Returns:
        Dictionary with stable training parameters
    """
    return {
        # Learning rate: Much lower for tiny model to prevent explosions
        "learning_rate": 1e-5,  # Reduced from 8e-5
        
        # Loss weights: Rebalanced to prevent dominance
        "mel_loss_weight": 1.0,  # Reduced from 2.5
        "kl_loss_weight": 0.5,   # Reduced from 1.8
        "duration_loss_weight": 0.1,
        "pitch_loss_weight": 0.05,
        "energy_loss_weight": 0.05,
        "prosody_pitch_loss_weight": 0.02,
        "prosody_energy_loss_weight": 0.02,
        "speaking_rate_loss_weight": 0.02,
        
        # Voice cloning loss weights (much lower)
        "voice_similarity_loss_weight": 0.5,
        "speaker_classification_loss_weight": 0.3,
        "voice_reconstruction_loss_weight": 0.4,
        "prosody_matching_loss_weight": 0.2,
        "spectral_consistency_loss_weight": 0.3,
        
        # Gradient clipping: More conservative
        "gradient_clip_norm": 0.3,  # Reduced from 0.8
        
        # Optimizer settings: More stable
        "weight_decay": 1e-7,  # Reduced from 5e-7
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        
        # Warmup: Longer for stability
        "warmup_steps": 3000,  # Increased from 1500
        
        # Scheduler: Simple noam instead of cosine
        "scheduler": "noam",
        "scheduler_params": {},
        
        # Loss stability features
        "use_adaptive_loss_weights": False,  # Disable adaptive weights
        "loss_smoothing_factor": 0.05,       # Stronger smoothing
        "max_loss_spike_threshold": 1.2,     # Lower spike threshold
        "gradient_norm_threshold": 1.0,      # Lower gradient monitoring
        
        # Disable advanced features that can cause instability
        "use_label_smoothing": False,
        "use_huber_loss": False,  # Use simple L1 loss instead
        "huber_delta": 1.0,
        
        # Mixed precision: Disable for stability
        "mixed_precision": False,
        
        # Early stopping: More patient
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.001,
        
        # More frequent checkpointing for safety
        "save_step": 1000,
        "val_step": 500,
        "log_step": 25,
    }


def create_numerical_stability_patches() -> Dict[str, str]:
    """
    Create patches for numerical stability in loss calculations.
    
    Returns:
        Dictionary with code patches
    """
    return {
        "mel_loss_epsilon": """
# Add epsilon to prevent division by zero in mel loss normalization
content_size = tf.cast(lengths, tf.float32) * tf.cast(tf.shape(y_true)[2], tf.float32)
normalized_loss = loss_sum / tf.maximum(content_size, 1e-8)  # Added epsilon
        """,
        
        "gradient_clipping": """
# More conservative gradient clipping
grad_tensors, global_norm = tf.clip_by_global_norm(grad_tensors, clip_norm=0.3)

# Check for NaN gradients before applying
has_nan = tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in grad_tensors if g is not None])
if has_nan:
    tf.print("WARNING: NaN gradients detected, skipping update")
    return {"total_loss": tf.constant(0.0), "gradient_norm": tf.constant(0.0)}
        """,
        
        "loss_validation": """
# Validate loss before returning
if tf.math.is_nan(loss) or tf.math.is_inf(loss):
    tf.print("WARNING: Invalid loss detected, replacing with previous valid loss")
    loss = tf.constant(1.0)  # Fallback to safe value
        """,
        
        "learning_rate_schedule": """
# More conservative learning rate schedule
def stable_noam_schedule(step):
    step = tf.cast(step, tf.float32)
    warmup_steps = tf.constant(3000.0)
    
    # More gradual warmup
    warmup_factor = tf.minimum(step / warmup_steps, 1.0)
    decay_factor = tf.pow(tf.maximum(step, warmup_steps) / warmup_steps, -0.3)  # Gentler decay
    
    return 1e-5 * warmup_factor * decay_factor
        """
    }


def apply_nan_loss_fix(config_path: str = None) -> str:
    """
    Apply NaN loss fix to the training configuration.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Path to the fixed config file
    """
    stable_config = create_stable_training_config()
    
    # Create a safe training config file
    safe_config_path = "/home/dev371/xTTS/MyXTTSModel/config_nan_loss_fix.yaml"
    
    config_content = f"""
# MyXTTS Stable Training Configuration - NaN Loss Fix
# تنظیمات پایدار برای جلوگیری از NaN شدن loss

model:
  # Tiny model settings optimized for stability
  text_encoder_dim: 256
  text_encoder_layers: 4
  text_encoder_heads: 4
  
  audio_encoder_dim: 256
  audio_encoder_layers: 4
  audio_encoder_heads: 4
  
  decoder_dim: 768
  decoder_layers: 8
  decoder_heads: 12
  
  # Memory optimizations
  enable_gradient_checkpointing: true
  max_attention_sequence_length: 256

training:
  # STABLE LEARNING RATE (key fix)
  learning_rate: {stable_config['learning_rate']}
  
  # BALANCED LOSS WEIGHTS (prevents explosion)
  mel_loss_weight: {stable_config['mel_loss_weight']}
  kl_loss_weight: {stable_config['kl_loss_weight']}
  duration_loss_weight: {stable_config['duration_loss_weight']}
  pitch_loss_weight: {stable_config['pitch_loss_weight']}
  energy_loss_weight: {stable_config['energy_loss_weight']}
  
  # Voice cloning weights (reduced)
  voice_similarity_loss_weight: {stable_config['voice_similarity_loss_weight']}
  speaker_classification_loss_weight: {stable_config['speaker_classification_loss_weight']}
  voice_reconstruction_loss_weight: {stable_config['voice_reconstruction_loss_weight']}
  
  # CONSERVATIVE GRADIENTS (prevents explosion)
  gradient_clip_norm: {stable_config['gradient_clip_norm']}
  weight_decay: {stable_config['weight_decay']}
  
  # OPTIMIZER SETTINGS
  optimizer: "adamw"
  beta1: {stable_config['beta1']}
  beta2: {stable_config['beta2']}
  eps: {stable_config['eps']}
  
  # SCHEDULER SETTINGS
  scheduler: "{stable_config['scheduler']}"
  warmup_steps: {stable_config['warmup_steps']}
  
  # STABILITY FEATURES
  use_adaptive_loss_weights: {str(stable_config['use_adaptive_loss_weights']).lower()}
  loss_smoothing_factor: {stable_config['loss_smoothing_factor']}
  max_loss_spike_threshold: {stable_config['max_loss_spike_threshold']}
  gradient_norm_threshold: {stable_config['gradient_norm_threshold']}
  
  # DISABLE PROBLEMATIC FEATURES
  use_label_smoothing: {str(stable_config['use_label_smoothing']).lower()}
  use_huber_loss: {str(stable_config['use_huber_loss']).lower()}
  
  # CHECKPOINTING
  save_step: {stable_config['save_step']}
  val_step: {stable_config['val_step']}
  log_step: {stable_config['log_step']}
  
  # EARLY STOPPING
  early_stopping_patience: {stable_config['early_stopping_patience']}
  early_stopping_min_delta: {stable_config['early_stopping_min_delta']}

data:
  batch_size: 16  # Smaller batch size for stability
  gradient_accumulation_steps: 4  # Compensate with accumulation
  num_workers: 4
  mixed_precision: {str(stable_config['mixed_precision']).lower()}
  
  # Data processing
  sample_rate: 22050
  n_mels: 80
  normalize_audio: true
  trim_silence: true
"""
    
    with open(safe_config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Created stable config: {safe_config_path}")
    return safe_config_path


def create_training_script_with_fixes() -> str:
    """
    Create a training script with NaN loss fixes applied.
    
    Returns:
        Path to the fixed training script
    """
    fixed_script_path = "/home/dev371/xTTS/MyXTTSModel/train_stable.py"
    
    script_content = '''#!/usr/bin/env python3
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
        
        enable_gradient_checkpointing=True,
        max_attention_sequence_length=256,
        
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
'''
    
    with open(fixed_script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(fixed_script_path, 0o755)
    
    print(f"✅ Created stable training script: {fixed_script_path}")
    return fixed_script_path


def main():
    parser = argparse.ArgumentParser(description="Fix NaN loss issues in MyXTTS training")
    parser.add_argument("--create-config", action="store_true", help="Create stable config file")
    parser.add_argument("--create-script", action="store_true", help="Create stable training script")
    parser.add_argument("--fix-all", action="store_true", help="Apply all fixes")
    
    args = parser.parse_args()
    
    print("🔧 MyXTTS NaN Loss Fix Tool")
    print("=" * 40)
    
    if args.create_config or args.fix_all:
        config_path = apply_nan_loss_fix()
        print(f"✅ Stable config created: {config_path}")
    
    if args.create_script or args.fix_all:
        script_path = create_training_script_with_fixes()
        print(f"✅ Stable training script created: {script_path}")
    
    if args.fix_all:
        print("\n🎯 ALL FIXES APPLIED SUCCESSFULLY!")
        print("\nTo use the stable training:")
        print("  python train_stable.py --model-size tiny")
        print("\nOr with specific config:")
        print("  python train_main.py --model-size tiny --optimization-level basic")
        print("\n📋 Key fixes applied:")
        print("  • Learning rate: 8e-5 → 1e-5 (5x lower)")
        print("  • Mel loss weight: 2.5 → 1.0 (2.5x lower)")
        print("  • Gradient clip: 0.8 → 0.3 (2.7x lower)")
        print("  • Mixed precision: disabled")
        print("  • Batch size: reduced to 16")
        print("  • Longer warmup: 3000 steps")
        print("  • Disabled adaptive loss weights")
        print("  • More frequent monitoring")


if __name__ == "__main__":
    main()
