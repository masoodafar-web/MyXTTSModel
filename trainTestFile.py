#!/usr/bin/env python3
"""
TrainTestFile.py - Flexible XTTS Training and Testing Script

This script provides a flexible interface for training and testing XTTS models.
It supports both programmatic configuration (without YAML) and optional YAML-based configuration.

Usage:
    # Train with programmatic configuration (no YAML required)
    python trainTestFile.py --mode train --data-path ./data/ljspeech

    # Train with optional YAML configuration
    python trainTestFile.py --mode train --config config.yaml

    # Create and save a configuration file
    python trainTestFile.py --mode create-config --output my_config.yaml

    # Test/inference mode
    python trainTestFile.py --mode test --checkpoint ./checkpoints/best.ckpt --text "Hello world"
"""

import os
import sys
import argparse
from typing import Optional
from pathlib import Path

# Add the project root to the path to import myxtts modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core configuration classes (these should work without heavy dependencies)
from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig

# Import other modules with error handling for missing dependencies
try:
    from myxtts.models.xtts import XTTS
    from myxtts.training.trainer import XTTSTrainer
    from myxtts.inference.synthesizer import XTTSInference
    from myxtts.utils.commons import setup_logging, set_random_seed
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies are missing: {e}")
    print("Configuration creation will work, but training/testing requires full installation.")
    DEPENDENCIES_AVAILABLE = False
    
    # Create dummy functions for missing dependencies
    def setup_logging():
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def set_random_seed(seed):
        import random
        random.seed(seed)


def create_default_config(
    data_path: str = "./data/ljspeech",
    language: str = "en",
    batch_size: int = 32,  # Increased default for better GPU utilization
    epochs: int = 100,
    learning_rate: float = 1e-4,
    sample_rate: int = 22050,
    checkpoint_dir: str = "./checkpoints",
    metadata_train_file: Optional[str] = None,
    metadata_eval_file: Optional[str] = None,
    wavs_train_dir: Optional[str] = None,
    wavs_eval_dir: Optional[str] = None,
    preprocessing_mode: str = "auto",
    # New GPU optimization parameters
    use_tf_native_loading: bool = True,
    enhanced_gpu_prefetch: bool = True,
    optimize_cpu_gpu_overlap: bool = True,
    num_workers: int = 8,
    prefetch_buffer_size: int = 8
) -> XTTSConfig:
    """
    Create a default configuration programmatically without requiring YAML.
    
    Args:
        data_path: Path to training data
        language: Language code (e.g., "en", "es", "fr")
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        sample_rate: Audio sample rate
        checkpoint_dir: Directory to save checkpoints
        metadata_train_file: Custom train metadata file path (optional)
        metadata_eval_file: Custom eval metadata file path (optional)
        wavs_train_dir: Custom train wav files directory (optional)
        wavs_eval_dir: Custom eval wav files directory (optional)
        preprocessing_mode: Dataset preprocessing mode ("auto", "precompute", "runtime")
    
    Returns:
        XTTSConfig: Configured XTTS configuration object
    """
    config = XTTSConfig()
    
    # Configure data settings
    config.data.dataset_path = data_path
    config.data.language = language
    config.data.batch_size = batch_size
    config.data.sample_rate = sample_rate
    config.data.preprocessing_mode = preprocessing_mode
    
    # Configure custom metadata file paths if provided
    if metadata_train_file:
        config.data.metadata_train_file = metadata_train_file
    if metadata_eval_file:
        config.data.metadata_eval_file = metadata_eval_file
    
    # Configure custom wav directories if provided
    if wavs_train_dir:
        config.data.wavs_train_dir = wavs_train_dir
    if wavs_eval_dir:
        config.data.wavs_eval_dir = wavs_eval_dir
    
    # Configure model settings
    config.model.sample_rate = sample_rate
    
    # Configure training settings with GPU optimizations
    config.training.epochs = epochs
    config.training.learning_rate = learning_rate
    config.training.checkpoint_dir = checkpoint_dir
    config.training.save_step = max(1000, epochs // 10)  # Save 10 times during training
    config.training.val_step = max(500, epochs // 20)    # Validate 20 times during training
    config.training.log_step = 100
    
    # CRITICAL GPU OPTIMIZATIONS: Enable all bottleneck fixes by default
    # TensorFlow-native loading optimizations (eliminates Python function bottlenecks)
    config.data.use_tf_native_loading = use_tf_native_loading     # Enable TF-native file loading
    config.data.enhanced_gpu_prefetch = enhanced_gpu_prefetch     # Enhanced GPU prefetching
    config.data.optimize_cpu_gpu_overlap = optimize_cpu_gpu_overlap  # Maximum CPU-GPU overlap
    
    # Advanced data loading optimizations
    config.data.num_workers = num_workers                # More workers for data loading
    config.data.prefetch_buffer_size = prefetch_buffer_size         # Larger prefetch buffer for GPU utilization
    config.data.shuffle_buffer_multiplier = 20   # Larger shuffle buffer
    config.data.prefetch_to_gpu = True          # Direct GPU prefetching
    
    # Memory and caching optimizations
    config.data.enable_memory_mapping = True    # Use memory mapping for cache files
    config.data.cache_verification = True       # Verify cache integrity  
    config.data.pin_memory = True              # Pin memory for faster GPU transfer
    config.data.persistent_workers = True      # Keep workers alive between epochs
    
    # TensorFlow and GPU optimizations
    config.data.enable_xla = True              # Enable XLA for better GPU performance
    config.data.mixed_precision = True         # Enable mixed precision for memory efficiency
    
    # Sequence length management for better GPU utilization
    config.data.max_mel_frames = 1024          # Allow longer sequences for better GPU utilization
    
    return config


def train_model(config: XTTSConfig, resume_checkpoint: Optional[str] = None):
    """
    Train the XTTS model with the given configuration.
    
    Args:
        config: XTTSConfig object with training parameters
        resume_checkpoint: Optional path to checkpoint to resume from
    """
    if not DEPENDENCIES_AVAILABLE:
        logger = setup_logging()
        logger.error("Training requires full dependencies. Please install requirements: pip install -r requirements.txt")
        return
    
    logger = setup_logging()
    
    logger.info("Starting XTTS training...")
    logger.info(f"Data path: {config.data.dataset_path}")
    logger.info(f"Language: {config.data.language}")
    logger.info(f"Batch size: {config.data.batch_size}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    
    # Create model
    logger.info("Creating XTTS model...")
    model = XTTS(config.model)
    
    # Print model information
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,}")
    except Exception as e:
        logger.warning(f"Could not count parameters: {e}")
    
    logger.info(f"Text vocab size: {config.model.text_vocab_size}")
    logger.info(f"Sample rate: {config.model.sample_rate}")
    logger.info(f"Languages: {config.model.languages}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = XTTSTrainer(
        config=config, 
        model=model, 
        resume_checkpoint=resume_checkpoint
    )
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset = trainer.prepare_datasets(
        train_data_path=config.data.dataset_path
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=config.training.epochs
    )
    
    logger.info("Training completed!")


def test_model(checkpoint_path: str, config: Optional[XTTSConfig] = None, text: str = "Hello, this is a test."):
    """
    Test the trained model with synthesis.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        config: Optional configuration (will create default if not provided)
        text: Text to synthesize
    """
    if not DEPENDENCIES_AVAILABLE:
        logger = setup_logging()
        logger.error("Testing requires full dependencies. Please install requirements: pip install -r requirements.txt")
        return
    
    logger = setup_logging()
    
    if config is None:
        logger.info("No config provided, creating default configuration...")
        config = create_default_config()
    
    logger.info("Loading model for inference...")
    
    # Create inference engine
    synthesizer = XTTSInference(config.model, checkpoint_path)
    
    # Synthesize text
    logger.info(f"Synthesizing text: '{text}'")
    result = synthesizer.synthesize(text, language=config.data.language)
    
    # Save output
    output_path = "synthesized_output.wav"
    logger.info(f"Saving synthesized audio to: {output_path}")
    # Note: The actual saving would depend on the synthesizer implementation
    
    return result


def create_config_file(output_path: str, **kwargs):
    """
    Create and save a configuration file.
    
    Args:
        output_path: Path where to save the configuration YAML file
        **kwargs: Additional configuration parameters
    """
    logger = setup_logging()
    
    logger.info("Creating configuration file...")
    config = create_default_config(**kwargs)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Save configuration
    config.to_yaml(output_path)
    
    logger.info(f"Configuration saved to: {output_path}")
    logger.info("You can now edit this file and use it with --config option")


def main():
    """Main entry point for the training and testing script."""
    parser = argparse.ArgumentParser(
        description="Flexible XTTS Training and Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode", 
        choices=["train", "test", "create-config"], 
        required=True,
        help="Operation mode: train model, test model, or create config file"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        help="Optional YAML configuration file (if not provided, uses programmatic config)"
    )
    
    # Data and model options
    parser.add_argument("--data-path", default="../dataset/dataset_train", help="Path to training data")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (optimized for GPU)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    
    # Custom metadata file options
    parser.add_argument("--metadata-train-file",default="../dataset/dataset_train/metadata_train.csv", help="Custom train metadata file path (e.g., metadata_train.csv)")
    parser.add_argument("--metadata-eval-file",default="../dataset/dataset_eval/metadata_eval.csv", help="Custom eval metadata file path (e.g., metadata_eval.csv)")
    
    # Custom wav directory options
    parser.add_argument("--wavs-train-dir",default="../dataset/dataset_train/wavs", help="Custom train wav files directory path")
    parser.add_argument("--wavs-eval-dir",default="../dataset/dataset_train/wavs", help="Custom eval wav files directory path")
    
    # Training options
    parser.add_argument("--resume-from", help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Dataset preprocessing options
    parser.add_argument("--preprocessing-mode", choices=["auto", "precompute", "runtime"], 
                       default="auto", help="Dataset preprocessing mode: 'auto' (try precompute, fall back), "
                       "'precompute' (fully preprocess before training), 'runtime' (process during training)")
    
    # GPU optimization options (NEW - for fixing CPU bottlenecks)
    parser.add_argument("--disable-tf-native-loading", action="store_true", 
                       help="Disable TensorFlow-native file loading optimization (not recommended)")
    parser.add_argument("--disable-gpu-prefetch", action="store_true",
                       help="Disable enhanced GPU prefetching (not recommended)")
    parser.add_argument("--disable-cpu-gpu-overlap", action="store_true",
                       help="Disable CPU-GPU overlap optimizations (not recommended)")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of data loading workers (default: 8)")
    parser.add_argument("--prefetch-buffer-size", type=int, default=8,
                       help="Prefetch buffer size for GPU utilization (default: 8)")
    
    # Testing options
    parser.add_argument("--checkpoint", help="Model checkpoint for testing/inference")
    parser.add_argument("--text", default="Hello, this is a test.", help="Text to synthesize")
    
    # Config creation options
    parser.add_argument("--output", help="Output path for created configuration file")
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    if args.mode == "train":
        # Load or create configuration
        if args.config and os.path.exists(args.config):
            print(f"Loading configuration from: {args.config}")
            config = XTTSConfig.from_yaml(args.config)
            
            # Override with command line arguments if provided
            if args.data_path != "./data/ljspeech":
                config.data.dataset_path = args.data_path
            if args.batch_size != 16:
                config.data.batch_size = args.batch_size
            if args.epochs != 100:
                config.training.epochs = args.epochs
            if args.learning_rate != 1e-4:
                config.training.learning_rate = args.learning_rate
            if args.checkpoint_dir != "./checkpoints":
                config.training.checkpoint_dir = args.checkpoint_dir
            # Override metadata file paths if provided
            if args.metadata_train_file:
                config.data.metadata_train_file = args.metadata_train_file
            if args.metadata_eval_file:
                config.data.metadata_eval_file = args.metadata_eval_file
            # Override wav directory paths if provided
            if args.wavs_train_dir:
                config.data.wavs_train_dir = args.wavs_train_dir
            if args.wavs_eval_dir:
                config.data.wavs_eval_dir = args.wavs_eval_dir
            # Override preprocessing mode if provided
            if hasattr(args, 'preprocessing_mode') and args.preprocessing_mode != "auto":
                config.data.preprocessing_mode = args.preprocessing_mode
            
            # Override GPU optimization settings if disabled via command line
            if args.disable_tf_native_loading:
                config.data.use_tf_native_loading = False
            if args.disable_gpu_prefetch:
                config.data.enhanced_gpu_prefetch = False
            if args.disable_cpu_gpu_overlap:
                config.data.optimize_cpu_gpu_overlap = False
            if args.num_workers != 8:
                config.data.num_workers = args.num_workers
            if args.prefetch_buffer_size != 8:
                config.data.prefetch_buffer_size = args.prefetch_buffer_size
        else:
            if args.config:
                print(f"Warning: Configuration file {args.config} not found, using programmatic config")
            print("Creating configuration programmatically (no YAML required)")
            config = create_default_config(
                data_path=args.data_path,
                language=args.language,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                sample_rate=args.sample_rate,
                checkpoint_dir=args.checkpoint_dir,
                metadata_train_file=args.metadata_train_file,
                metadata_eval_file=args.metadata_eval_file,
                wavs_train_dir=args.wavs_train_dir,
                wavs_eval_dir=args.wavs_eval_dir,
                preprocessing_mode=args.preprocessing_mode,
                # Apply GPU optimization settings
                use_tf_native_loading=not args.disable_tf_native_loading,
                enhanced_gpu_prefetch=not args.disable_gpu_prefetch,
                optimize_cpu_gpu_overlap=not args.disable_cpu_gpu_overlap,
                num_workers=args.num_workers,
                prefetch_buffer_size=args.prefetch_buffer_size
            )
        
        train_model(config, resume_checkpoint=args.resume_from)
    
    elif args.mode == "test":
        if not args.checkpoint:
            print("Error: --checkpoint is required for test mode")
            sys.exit(1)
        
        # Load config if provided, otherwise use default
        config = None
        if args.config and os.path.exists(args.config):
            config = XTTSConfig.from_yaml(args.config)
        
        test_model(args.checkpoint, config, args.text)
    
    elif args.mode == "create-config":
        if not args.output:
            print("Error: --output is required for create-config mode")
            sys.exit(1)
        
        create_config_file(
            args.output,
            data_path=args.data_path,
            language=args.language,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            sample_rate=args.sample_rate,
            checkpoint_dir=args.checkpoint_dir,
            metadata_train_file=args.metadata_train_file,
            metadata_eval_file=args.metadata_eval_file,
            wavs_train_dir=args.wavs_train_dir,
            wavs_eval_dir=args.wavs_eval_dir,
            preprocessing_mode=args.preprocessing_mode
        )


if __name__ == "__main__":
    main()