"""
Command Line Interface for MyXTTS.

This module provides CLI commands for training, inference, and utilities
for the MyXTTS text-to-speech system.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from .config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
from .models.xtts import XTTS
from .training.trainer import XTTSTrainer
from .inference.synthesizer import XTTSInference
from .data.ljspeech import LJSpeechDataset
from .utils.commons import setup_logging, set_random_seed


def create_config_command(args):
    """Create a new configuration file."""
    logger = setup_logging()
    
    # Create default configuration
    config = XTTSConfig()
    
    # Update with command line arguments
    if hasattr(args, 'sample_rate') and args.sample_rate:
        config.model.sample_rate = args.sample_rate
        config.data.sample_rate = args.sample_rate
    
    if hasattr(args, 'language') and args.language:
        config.data.language = args.language
    
    if hasattr(args, 'dataset_path') and args.dataset_path:
        config.data.dataset_path = args.dataset_path
    
    # Save configuration
    config.to_yaml(args.output)
    
    logger.info(f"Created configuration file: {args.output}")
    logger.info("Edit the configuration file to customize model settings")


def train_command(args):
    """Train XTTS model."""
    logger = setup_logging()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = XTTSConfig.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config.data.dataset_path = args.data_path
    
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    
    if args.epochs:
        config.training.epochs = args.epochs
    
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Create model
    model = XTTS(config.model)
    
    # Create trainer
    trainer = XTTSTrainer(
        config=config,
        model=model,
        resume_checkpoint=args.resume_from
    )
    
    # Prepare datasets
    train_dataset, val_dataset = trainer.prepare_datasets(
        train_data_path=config.data.dataset_path,
        val_data_path=args.val_data_path
    )
    
    logger.info("Starting training...")
    
    # Start training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=config.training.epochs
    )
    
    logger.info("Training completed!")


def inference_command(args):
    """Run inference with trained model."""
    logger = setup_logging()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = XTTSConfig.from_yaml(args.config)
    
    # Create inference engine
    inference = XTTSInference(
        config=config,
        checkpoint_path=args.checkpoint
    )
    
    # Load reference audio if provided
    reference_audio = None
    if args.reference_audio:
        if not os.path.exists(args.reference_audio):
            logger.error(f"Reference audio not found: {args.reference_audio}")
            sys.exit(1)
        reference_audio = args.reference_audio
    
    # Synthesize speech
    if args.text_file:
        # Read text from file
        if not os.path.exists(args.text_file):
            logger.error(f"Text file not found: {args.text_file}")
            sys.exit(1)
        
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = args.text
    
    if not text:
        logger.error("No text provided for synthesis")
        sys.exit(1)
    
    logger.info(f"Synthesizing: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Run synthesis
    result = inference.synthesize(
        text=text,
        reference_audio=reference_audio,
        language=args.language,
        temperature=args.temperature,
        max_length=args.max_length
    )
    
    # Save output
    output_path = args.output or "output.wav"
    inference.save_audio(
        result["audio"],
        output_path,
        result["sample_rate"]
    )
    
    logger.info(f"Audio saved to: {output_path}")
    logger.info(f"Duration: {len(result['audio']) / result['sample_rate']:.2f}s")


def clone_voice_command(args):
    """Clone voice from reference audio."""
    logger = setup_logging()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = XTTSConfig.from_yaml(args.config)
    
    # Check if voice conditioning is enabled
    if not config.model.use_voice_conditioning:
        logger.error("Voice conditioning is not enabled in the model configuration")
        sys.exit(1)
    
    # Create inference engine
    inference = XTTSInference(
        config=config,
        checkpoint_path=args.checkpoint
    )
    
    # Validate reference audio
    if not os.path.exists(args.reference_audio):
        logger.error(f"Reference audio not found: {args.reference_audio}")
        sys.exit(1)
    
    # Read text
    if args.text_file:
        if not os.path.exists(args.text_file):
            logger.error(f"Text file not found: {args.text_file}")
            sys.exit(1)
        
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = args.text
    
    if not text:
        logger.error("No text provided for voice cloning")
        sys.exit(1)
    
    logger.info(f"Cloning voice for: {text[:100]}{'...' if len(text) > 100 else ''}")
    logger.info(f"Reference audio: {args.reference_audio}")
    
    # Run voice cloning
    result = inference.clone_voice(
        text=text,
        reference_audio=args.reference_audio,
        language=args.language,
        temperature=args.temperature,
        max_length=args.max_length
    )
    
    # Save output
    output_path = args.output or "cloned_voice.wav"
    inference.save_audio(
        result["audio"],
        output_path,
        result["sample_rate"]
    )
    
    logger.info(f"Cloned voice saved to: {output_path}")
    logger.info(f"Duration: {len(result['audio']) / result['sample_rate']:.2f}s")


def dataset_info_command(args):
    """Show dataset information."""
    logger = setup_logging()
    
    # Load configuration
    config = XTTSConfig()
    config.data.dataset_path = args.data_path
    
    # Load dataset
    dataset = LJSpeechDataset(
        data_path=args.data_path,
        config=config.data,
        subset=args.subset,
        download=False
    )
    
    # Get statistics
    stats = dataset.get_statistics()
    
    logger.info("Dataset Statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Total duration: {stats['audio_duration_hours']:.2f} hours")
    logger.info(f"  Sample rate: {stats['sample_rate']} Hz")
    logger.info(f"  Vocabulary size: {stats['vocab_size']}")
    
    logger.info("Text length statistics:")
    logger.info(f"  Min: {stats['text_length']['min']} chars")
    logger.info(f"  Max: {stats['text_length']['max']} chars") 
    logger.info(f"  Mean: {stats['text_length']['mean']:.1f} chars")
    
    logger.info("Audio length statistics:")
    logger.info(f"  Min: {stats['audio_length']['min']} samples")
    logger.info(f"  Max: {stats['audio_length']['max']} samples")
    logger.info(f"  Mean: {stats['audio_length']['mean']:.0f} samples")


def benchmark_command(args):
    """Benchmark model inference speed."""
    logger = setup_logging()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = XTTSConfig.from_yaml(args.config)
    
    # Create inference engine
    inference = XTTSInference(
        config=config,
        checkpoint_path=args.checkpoint
    )
    
    # Prepare test texts
    test_texts = [
        "Hello, this is a test of the MyXTTS text-to-speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",
        "To be or not to be, that is the question.",
        "The weather today is sunny with a chance of rain in the afternoon."
    ]
    
    if args.text_file:
        # Load texts from file
        with open(args.text_file, 'r', encoding='utf-8') as f:
            test_texts = [line.strip() for line in f if line.strip()]
    
    # Load reference audio if provided
    reference_audio = args.reference_audio if args.reference_audio else None
    
    # Run benchmark
    results = inference.benchmark(
        texts=test_texts,
        reference_audio=reference_audio,
        num_runs=args.num_runs
    )
    
    logger.info("Benchmark Results:")
    logger.info(f"  Real-time factor: {results['real_time_factor']:.2f}")
    logger.info(f"  Average synthesis time: {results['average_synthesis_time']:.2f}s")
    logger.info(f"  Average audio length: {results['average_audio_length']:.2f}s")
    logger.info(f"  Throughput: {results['throughput_samples_per_second']:.2f} samples/s")
    logger.info(f"  Total runs: {results['total_runs']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="myxtts",
        description="MyXTTS: TensorFlow-based XTTS Text-to-Speech System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create configuration file")
    config_parser.add_argument("--output", "-o", required=True, help="Output configuration file")
    config_parser.add_argument("--sample-rate", type=int, help="Sample rate")
    config_parser.add_argument("--language", help="Default language")
    config_parser.add_argument("--dataset-path", help="Dataset path")
    config_parser.set_defaults(func=create_config_command)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train XTTS model")
    train_parser.add_argument("--config", "-c", required=True, help="Configuration file")
    train_parser.add_argument("--data-path", help="Training data path")
    train_parser.add_argument("--val-data-path", help="Validation data path")
    train_parser.add_argument("--checkpoint-dir", help="Checkpoint directory")
    train_parser.add_argument("--resume-from", help="Resume training from checkpoint")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.set_defaults(func=train_command)
    
    # Inference command
    inference_parser = subparsers.add_parser("synthesize", help="Synthesize speech")
    inference_parser.add_argument("--config", "-c", required=True, help="Configuration file")
    inference_parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    inference_parser.add_argument("--text", help="Text to synthesize")
    inference_parser.add_argument("--text-file", help="File containing text to synthesize")
    inference_parser.add_argument("--reference-audio", help="Reference audio for voice conditioning")
    inference_parser.add_argument("--output", "-o", help="Output audio file")
    inference_parser.add_argument("--language", help="Language code")
    inference_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    inference_parser.add_argument("--max-length", type=int, default=1000, help="Maximum generation length")
    inference_parser.set_defaults(func=inference_command)
    
    # Voice cloning command
    clone_parser = subparsers.add_parser("clone-voice", help="Clone voice from reference audio")
    clone_parser.add_argument("--config", "-c", required=True, help="Configuration file")
    clone_parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    clone_parser.add_argument("--text", help="Text to synthesize")
    clone_parser.add_argument("--text-file", help="File containing text to synthesize")
    clone_parser.add_argument("--reference-audio", required=True, help="Reference audio for voice cloning")
    clone_parser.add_argument("--output", "-o", help="Output audio file")
    clone_parser.add_argument("--language", help="Language code")
    clone_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    clone_parser.add_argument("--max-length", type=int, default=1000, help="Maximum generation length")
    clone_parser.set_defaults(func=clone_voice_command)
    
    # Dataset info command
    dataset_parser = subparsers.add_parser("dataset-info", help="Show dataset information")
    dataset_parser.add_argument("--data-path", required=True, help="Dataset path")
    dataset_parser.add_argument("--subset", default="train", help="Dataset subset")
    dataset_parser.set_defaults(func=dataset_info_command)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark inference speed")
    benchmark_parser.add_argument("--config", "-c", required=True, help="Configuration file")
    benchmark_parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    benchmark_parser.add_argument("--text-file", help="File with test texts")
    benchmark_parser.add_argument("--reference-audio", help="Reference audio")
    benchmark_parser.add_argument("--num-runs", type=int, default=5, help="Number of benchmark runs")
    benchmark_parser.set_defaults(func=benchmark_command)
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()