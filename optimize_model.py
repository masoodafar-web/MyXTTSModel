#!/usr/bin/env python3
"""Model optimization script for creating lightweight XTTS models.

This script provides end-to-end model optimization including:
- Model compression (pruning + quantization)
- Knowledge distillation for creating smaller student models
- Performance benchmarking and evaluation

Usage Examples:
    # Create lightweight model with compression
    python optimize_model.py --model checkpoints/model.h5 --output optimized_model --compress
    
    # Create distilled student model
    python optimize_model.py --model checkpoints/model.h5 --output student_model --distill
    
    # Full optimization pipeline
    python optimize_model.py --model checkpoints/model.h5 --output optimized --compress --distill --benchmark
    
    # Create configuration for lightweight training
    python optimize_model.py --create-config --output lightweight_config.yaml
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add myxtts to path if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tensorflow as tf
    from myxtts.optimization import (
        ModelCompressor, 
        QuantizationAwareTrainer,
        ModelDistiller,
        OptimizedInference,
        CompressionConfig,
        DistillationConfig,
        InferenceConfig
    )
    from myxtts.optimization.compression import create_lightweight_config
    from myxtts.config.config import XTTSConfig
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    TF_AVAILABLE = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XTTS model optimization for real-time inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        "--model", "-m",
        help="Path to trained XTTS model to optimize"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory/path for optimized model"
    )
    
    # Optimization methods
    parser.add_argument(
        "--compress", "-c",
        action="store_true",
        help="Apply model compression (pruning + quantization)"
    )
    parser.add_argument(
        "--distill", "-d",
        action="store_true",
        help="Create distilled student model"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create lightweight model configuration for training"
    )
    
    # Configuration options
    parser.add_argument(
        "--compression-config",
        help="JSON file with compression configuration"
    )
    parser.add_argument(
        "--distillation-config",
        help="JSON file with distillation configuration"
    )
    parser.add_argument(
        "--target-speedup",
        type=float,
        default=2.0,
        help="Target inference speedup (default: 2.0x)"
    )
    parser.add_argument(
        "--max-quality-loss",
        type=float,
        default=0.1,
        help="Maximum acceptable quality loss (0.1 = 10%%)"
    )
    
    # Data for training/evaluation
    parser.add_argument(
        "--train-data",
        help="Training dataset for quantization-aware training"
    )
    parser.add_argument(
        "--val-data",
        help="Validation dataset"
    )
    parser.add_argument(
        "--test-data",
        help="Test dataset for evaluation"
    )
    
    # Evaluation and benchmarking
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark on optimized model"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run quality evaluation using TTS metrics"
    )
    parser.add_argument(
        "--test-texts",
        help="Text file with test sentences for benchmarking"
    )
    
    # Output formats
    parser.add_argument(
        "--save-tflite",
        action="store_true",
        help="Convert to TensorFlow Lite format"
    )
    parser.add_argument(
        "--save-onnx",
        action="store_true",
        help="Convert to ONNX format (if available)"
    )
    
    # Misc options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for optimization (if available)"
    )
    
    return parser.parse_args()


def load_config_from_file(config_path: str, config_class) -> object:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object from dictionary
    return config_class(**config_dict)


def create_compression_config(args: argparse.Namespace) -> CompressionConfig:
    """Create compression configuration from arguments."""
    if args.compression_config:
        return load_config_from_file(args.compression_config, CompressionConfig)
    
    return CompressionConfig(
        target_speedup=args.target_speedup,
        max_quality_loss=args.max_quality_loss,
        enable_pruning=True,
        enable_quantization=True,
        final_sparsity=0.5,  # 50% weight removal
        reduce_decoder_layers=True,
        target_decoder_layers=8,  # Reduce from 16
        reduce_decoder_dim=True,
        target_decoder_dim=768   # Reduce from 1536
    )


def create_distillation_config(args: argparse.Namespace) -> DistillationConfig:
    """Create distillation configuration from arguments."""
    if args.distillation_config:
        return load_config_from_file(args.distillation_config, DistillationConfig)
    
    return DistillationConfig(
        temperature=4.0,
        distillation_loss_weight=0.7,
        student_loss_weight=0.3,
        epochs=50,
        student_decoder_dim=384,     # Much smaller than teacher (1536)
        student_decoder_layers=4,    # Much smaller than teacher (16)
        student_decoder_heads=6,     # Much smaller than teacher (24)
        student_text_encoder_layers=2,
        student_audio_encoder_layers=2
    )


def load_model(model_path: str) -> tf.keras.Model:
    """Load XTTS model from path."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # Try loading as SavedModel first
        model = tf.saved_model.load(model_path)
        logger.info("Loaded as TensorFlow SavedModel")
        return model
    except Exception:
        try:
            # Try loading as Keras model
            model = tf.keras.models.load_model(model_path)
            logger.info("Loaded as Keras model")
            return model
        except Exception as e:
            raise ValueError(f"Could not load model from {model_path}: {e}")


def compress_model(model: tf.keras.Model, 
                  config: CompressionConfig,
                  output_path: str,
                  args: argparse.Namespace) -> tf.keras.Model:
    """Apply model compression."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model compression...")
    
    # Create compressor
    compressor = ModelCompressor(config)
    
    # Apply compression
    compressed_model = compressor.compress_model(model)
    
    # Quantization-aware training if data provided
    if args.train_data and args.val_data:
        logger.info("Running quantization-aware training...")
        
        # Load datasets (placeholder - would need actual implementation)
        # train_ds = load_dataset(args.train_data)
        # val_ds = load_dataset(args.val_data)
        
        qat_trainer = QuantizationAwareTrainer(config)
        # compressed_model = qat_trainer.train(compressed_model, train_ds, val_ds)
        logger.warning("QAT training skipped - dataset loading not implemented")
    
    # Finalize compression
    final_model = compressor.finalize_compression(compressed_model)
    
    # Save compressed model
    compressed_path = f"{output_path}_compressed"
    final_model.save(compressed_path)
    logger.info(f"Compressed model saved to: {compressed_path}")
    
    # Generate compression report
    stats = compressor.get_compression_stats(model, final_model)
    
    # Save compression report
    report_path = f"{compressed_path}_compression_report.json"
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Compression statistics:")
    logger.info(f"  Parameters: {stats['original_parameters']:,} → {stats['compressed_parameters']:,}")
    logger.info(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
    logger.info(f"  Size reduction: {stats['size_reduction_percent']:.1f}%")
    logger.info(f"  Estimated speedup: {stats['estimated_speedup']:.1f}x")
    
    return final_model


def distill_model(teacher_model: tf.keras.Model,
                 config: DistillationConfig,
                 output_path: str,
                 args: argparse.Namespace) -> tf.keras.Model:
    """Create distilled student model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model distillation...")
    
    # Create distiller
    distiller = ModelDistiller(config)
    
    # Create student model
    student_model = distiller.create_student_model(teacher_model)
    
    # Distillation training if data provided
    if args.train_data and args.val_data:
        logger.info("Running knowledge distillation training...")
        
        # Load datasets (placeholder)
        # train_ds = load_dataset(args.train_data)
        # val_ds = load_dataset(args.val_data)
        
        # student_model = distiller.distill(teacher_model, student_model, train_ds, val_ds)
        logger.warning("Distillation training skipped - dataset loading not implemented")
    
    # Save student model
    student_path = f"{output_path}_student"
    student_model.save(student_path)
    logger.info(f"Student model saved to: {student_path}")
    
    # Evaluate compression if test data provided
    if args.test_data:
        logger.info("Evaluating distillation results...")
        # test_ds = load_dataset(args.test_data)
        # eval_results = distiller.evaluate_compression(teacher_model, student_model, test_ds)
        logger.warning("Distillation evaluation skipped - dataset loading not implemented")
    
    return student_model


def benchmark_model(model_path: str, 
                   output_path: str,
                   args: argparse.Namespace) -> Dict:
    """Benchmark optimized model performance."""
    logger = logging.getLogger(__name__)
    logger.info("Starting performance benchmark...")
    
    # Create inference config
    inference_config = InferenceConfig(
        use_tflite=model_path.endswith('.tflite'),
        use_gpu_acceleration=args.gpu,
        quality_mode="balanced",
        target_rtf=0.1  # Target 10% real-time factor
    )
    
    # Create optimized inference engine
    inference_engine = OptimizedInference(model_path, inference_config)
    
    # Load test texts
    test_texts = []
    if args.test_texts and os.path.exists(args.test_texts):
        with open(args.test_texts, 'r', encoding='utf-8') as f:
            test_texts = [line.strip() for line in f if line.strip()]
    else:
        # Default test texts
        test_texts = [
            "Hello, this is a test of the text-to-speech system.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can be optimized for better performance.",
            "Real-time text-to-speech synthesis requires efficient algorithms.",
            "This sentence is used to evaluate the quality and speed of synthesis."
        ]
    
    # Run benchmark
    benchmark_results = inference_engine.benchmark(test_texts, repetitions=5)
    
    # Save benchmark results
    benchmark_path = f"{output_path}_benchmark_results.json"
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Save performance log
    performance_log_path = f"{output_path}_performance_log.json"
    inference_engine.save_performance_log(performance_log_path)
    
    logger.info("Benchmark results:")
    logger.info(f"  Average RTF: {benchmark_results['average_rtf']:.3f}")
    logger.info(f"  P95 RTF: {benchmark_results['p95_rtf']:.3f}")
    logger.info(f"  Real-time capable: {benchmark_results['real_time_percentage']:.1f}%")
    logger.info(f"  Target met: {benchmark_results['target_met']}")
    
    return benchmark_results


def evaluate_model_quality(model_path: str,
                          output_path: str,
                          args: argparse.Namespace) -> Dict:
    """Evaluate model quality using TTS metrics."""
    logger = logging.getLogger(__name__)
    logger.info("Starting quality evaluation...")
    
    try:
        from myxtts.evaluation import TTSEvaluator
        
        # Create evaluator
        evaluator = TTSEvaluator(
            enable_mosnet=True,
            enable_asr_wer=True,
            enable_cmvn=True,
            enable_spectral=True
        )
        
        # Generate test audio samples (placeholder)
        # In practice, you would synthesize audio using the optimized model
        # and evaluate against reference texts
        
        logger.warning("Quality evaluation placeholder - would need synthesized audio samples")
        
        return {"status": "Quality evaluation not implemented yet"}
        
    except ImportError:
        logger.warning("TTS evaluation module not available")
        return {"error": "Evaluation module not available"}


def create_lightweight_training_config(output_path: str):
    """Create configuration for training lightweight models."""
    logger = logging.getLogger(__name__)
    logger.info("Creating lightweight training configuration...")
    
    # Create lightweight config
    lightweight_config = create_lightweight_config({})
    
    # Save as YAML (would need PyYAML)
    config_path = f"{output_path}_lightweight_config.json"
    with open(config_path, 'w') as f:
        json.dump(lightweight_config, f, indent=2)
    
    logger.info(f"Lightweight configuration saved to: {config_path}")
    logger.info("Key optimizations:")
    logger.info(f"  Decoder dim: 1536 → {lightweight_config['decoder_dim']}")
    logger.info(f"  Decoder layers: 16 → {lightweight_config['decoder_layers']}")
    logger.info(f"  Decoder heads: 24 → {lightweight_config['decoder_heads']}")
    logger.info(f"  Text encoder layers: 8 → {lightweight_config['text_encoder_layers']}")
    
    return lightweight_config


def main():
    """Main optimization function."""
    if not TF_AVAILABLE:
        print("Error: TensorFlow and optimization modules not available")
        print("Please install required dependencies")
        sys.exit(1)
    
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create lightweight configuration
        if args.create_config:
            create_lightweight_training_config(args.output)
            return
        
        # Load model
        if not args.model:
            logger.error("Model path required for optimization")
            sys.exit(1)
        
        model = load_model(args.model)
        logger.info(f"Original model parameters: {model.count_params():,}")
        
        optimized_models = []
        
        # Apply compression
        if args.compress:
            compression_config = create_compression_config(args)
            compressed_model = compress_model(model, compression_config, args.output, args)
            optimized_models.append(("compressed", compressed_model))
            
            # Convert to TensorFlow Lite if requested
            if args.save_tflite:
                qat_trainer = QuantizationAwareTrainer(compression_config)
                tflite_model = qat_trainer.convert_to_tflite(compressed_model)
                tflite_path = f"{args.output}_compressed.tflite"
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                logger.info(f"TensorFlow Lite model saved to: {tflite_path}")
        
        # Apply distillation
        if args.distill:
            distillation_config = create_distillation_config(args)
            student_model = distill_model(model, distillation_config, args.output, args)
            optimized_models.append(("student", student_model))
        
        # Benchmark optimized models
        if args.benchmark:
            for model_type, opt_model in optimized_models:
                model_path = f"{args.output}_{model_type}"
                benchmark_model(model_path, f"{args.output}_{model_type}", args)
            
            # Also benchmark TensorFlow Lite if available
            tflite_path = f"{args.output}_compressed.tflite"
            if os.path.exists(tflite_path):
                benchmark_model(tflite_path, f"{args.output}_tflite", args)
        
        # Evaluate quality
        if args.evaluate:
            for model_type, opt_model in optimized_models:
                model_path = f"{args.output}_{model_type}"
                evaluate_model_quality(model_path, f"{args.output}_{model_type}", args)
        
        logger.info("Model optimization completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Original model: {args.model}")
        print(f"Original parameters: {model.count_params():,}")
        
        for model_type, opt_model in optimized_models:
            print(f"{model_type.capitalize()} model parameters: {opt_model.count_params():,}")
            compression_ratio = model.count_params() / opt_model.count_params()
            print(f"{model_type.capitalize()} compression ratio: {compression_ratio:.1f}x")
        
        print(f"\nOptimized models saved to: {args.output}_*")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()