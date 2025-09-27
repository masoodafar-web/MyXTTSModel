#!/usr/bin/env python3
"""
Comprehensive Model and Training Validation Tool

This script performs a complete validation of the MyXTTS model and training pipeline
to identify any issues that could cause slow loss convergence or poor performance.

Usage:
    python comprehensive_validation.py --data-path ./data/ljspeech --quick-test
    python comprehensive_validation.py --data-path ./data/ljspeech --full-validation
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MyXTTS components
from myxtts.config.config import XTTSConfig, DataConfig, TrainingConfig, ModelConfig
from myxtts.models.xtts import XTTS
from myxtts.data.ljspeech import LJSpeechDataset
from myxtts.training.losses import XTTSLoss
from myxtts.training.trainer import XTTSTrainer
from myxtts.utils.commons import setup_logging, check_gpu_setup


class ComprehensiveValidator:
    """Comprehensive validation tool for MyXTTS model and training pipeline."""
    
    def __init__(self, data_path: str, config_path: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            data_path: Path to dataset
            config_path: Optional path to config file
        """
        self.data_path = data_path
        self.logger = setup_logging()
        self.results = {}
        self.issues = []
        self.recommendations = []
        
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            self.config = XTTSConfig.from_yaml(config_path)
            self.logger.info(f"Loaded config from: {config_path}")
        else:
            self.config = self._create_test_config()
            self.logger.info("Created default test configuration")
    
    def _create_test_config(self) -> XTTSConfig:
        """Create optimized test configuration."""
        config = XTTSConfig()
        
        # Data configuration - optimized for testing
        config.data.dataset_path = self.data_path
        config.data.batch_size = 8  # Small batch for testing
        config.data.num_workers = 4
        config.data.prefetch_buffer_size = 4
        config.data.preprocessing_mode = "runtime"  # For initial testing
        config.data.sample_rate = 22050
        config.data.n_mels = 80
        config.data.max_mel_frames = 512  # Smaller for testing
        
        # Training configuration - optimized for convergence
        config.training.learning_rate = 1e-4
        config.training.mel_loss_weight = 35.0
        config.training.kl_loss_weight = 1.0
        config.training.gradient_clip_norm = 1.0
        config.training.warmup_steps = 1000
        
        # Enable stability improvements
        config.training.use_adaptive_loss_weights = True
        config.training.loss_smoothing_factor = 0.1
        config.training.use_label_smoothing = True
        config.training.use_huber_loss = True
        
        return config
    
    def validate_environment(self) -> bool:
        """Validate environment setup."""
        self.logger.info("🔍 Validating Environment Setup...")
        
        try:
            # Check TensorFlow
            self.logger.info(f"TensorFlow version: {tf.__version__}")
            
            # Check GPU setup
            gpu_success, device, gpu_recommendations = check_gpu_setup()
            self.results['gpu_available'] = gpu_success
            self.results['compute_device'] = device
            
            if not gpu_success:
                self.issues.append("GPU not available or not properly configured")
                self.recommendations.extend(gpu_recommendations)
                self.logger.warning("⚠️ GPU issues detected - training will be slow")
            else:
                self.logger.info(f"✅ GPU available: {device}")
            
            # Check memory
            if gpu_success:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                    self.logger.info(f"GPU memory: {gpu_info['peak'] / 1024**3:.1f}GB peak")
            
            return True
            
        except Exception as e:
            self.issues.append(f"Environment validation failed: {str(e)}")
            self.logger.error(f"❌ Environment validation error: {e}")
            return False
    
    def validate_dataset(self) -> bool:
        """Validate dataset loading and preprocessing."""
        self.logger.info("🔍 Validating Dataset...")
        
        try:
            # Check if data path exists
            if not os.path.exists(self.data_path):
                self.issues.append(f"Dataset path does not exist: {self.data_path}")
                return False
            
            # Create dataset instance
            dataset = LJSpeechDataset(
                data_path=self.data_path,
                config=self.config.data,
                subset="train",
                download=False,
                preprocess=False
            )
            
            # Check dataset size
            try:
                data_size = len(dataset.filelist) if hasattr(dataset, 'filelist') else 0
                self.results['dataset_size'] = data_size
                self.logger.info(f"Dataset size: {data_size} samples")
                
                if data_size == 0:
                    self.issues.append("Dataset appears to be empty")
                    return False
                elif data_size < 100:
                    self.issues.append(f"Dataset very small ({data_size} samples) - may affect training")
                    
            except Exception as e:
                self.logger.warning(f"Could not determine dataset size: {e}")
            
            # Test data loading
            self.logger.info("Testing data loading...")
            tf_dataset = dataset.get_tf_dataset()
            
            # Try to load a few samples
            sample_count = 0
            total_time = 0
            
            for i, batch in enumerate(tf_dataset.take(5)):
                start_time = time.time()
                
                # Validate batch structure
                if isinstance(batch, dict):
                    self.logger.info(f"Batch {i}: {list(batch.keys())}")
                    for key, tensor in batch.items():
                        if isinstance(tensor, tf.Tensor):
                            self.logger.info(f"  {key}: {tensor.shape} {tensor.dtype}")
                else:
                    self.logger.info(f"Batch {i}: {type(batch)}")
                
                total_time += time.time() - start_time
                sample_count += 1
            
            avg_time = total_time / sample_count if sample_count > 0 else 0
            self.results['data_loading_time'] = avg_time
            self.logger.info(f"Average batch loading time: {avg_time:.3f}s")
            
            if avg_time > 1.0:
                self.issues.append(f"Data loading is slow ({avg_time:.3f}s/batch)")
                self.recommendations.append("Consider using preprocessing_mode='precompute' for faster loading")
            
            return True
            
        except Exception as e:
            self.issues.append(f"Dataset validation failed: {str(e)}")
            self.logger.error(f"❌ Dataset validation error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def validate_model(self) -> bool:
        """Validate model architecture and initialization."""
        self.logger.info("🔍 Validating Model Architecture...")
        
        try:
            # Create model
            model = XTTS(self.config.model)
            self.logger.info("✅ Model created successfully")
            
            # Check model parameters
            try:
                total_params = sum(p.numel() for p in model.parameters() if hasattr(p, 'numel'))
                trainable_params = sum(p.numel() for p in model.parameters() if hasattr(p, 'numel') and getattr(p, 'requires_grad', True))
                
                self.results['total_parameters'] = total_params
                self.results['trainable_parameters'] = trainable_params
                
                self.logger.info(f"Total parameters: {total_params:,}")
                self.logger.info(f"Trainable parameters: {trainable_params:,}")
                
                if total_params == 0:
                    self.issues.append("Model has no parameters")
                    return False
                    
            except Exception as e:
                self.logger.warning(f"Could not count parameters: {e}")
            
            # Test forward pass with dummy data
            self.logger.info("Testing model forward pass...")
            
            try:
                # Create dummy input
                batch_size = 2
                seq_len = 128
                mel_len = 256
                
                dummy_input = {
                    'text_input': tf.random.uniform((batch_size, seq_len), 0, 100, dtype=tf.int32),
                    'mel_input': tf.random.normal((batch_size, mel_len, self.config.data.n_mels)),
                    'speaker_id': tf.zeros((batch_size,), dtype=tf.int32)
                }
                
                # Forward pass
                with tf.GradientTape() as tape:
                    output = model(dummy_input, training=True)
                
                # Validate output
                if isinstance(output, dict):
                    self.logger.info("Model output keys:")
                    for key, tensor in output.items():
                        if isinstance(tensor, tf.Tensor):
                            self.logger.info(f"  {key}: {tensor.shape} {tensor.dtype}")
                            
                            # Check for NaN/Inf
                            if tf.reduce_any(tf.math.is_nan(tensor)):
                                self.issues.append(f"Model output contains NaN values in {key}")
                            if tf.reduce_any(tf.math.is_inf(tensor)):
                                self.issues.append(f"Model output contains Inf values in {key}")
                    
                    self.logger.info("✅ Forward pass successful")
                else:
                    self.issues.append(f"Unexpected model output type: {type(output)}")
                
                # Test gradient computation
                if isinstance(output, dict) and 'mel_output' in output:
                    loss = tf.reduce_mean(output['mel_output'])
                    grads = tape.gradient(loss, model.trainable_variables)
                    
                    grad_norms = []
                    for grad in grads:
                        if grad is not None:
                            grad_norm = tf.norm(grad)
                            grad_norms.append(grad_norm)
                    
                    if grad_norms:
                        avg_grad_norm = tf.reduce_mean(grad_norms)
                        self.results['avg_gradient_norm'] = float(avg_grad_norm)
                        self.logger.info(f"Average gradient norm: {avg_grad_norm:.6f}")
                        
                        if avg_grad_norm == 0:
                            self.issues.append("Gradients are zero - model may not be learning")
                        elif avg_grad_norm > 10:
                            self.issues.append(f"Large gradients detected ({avg_grad_norm:.3f}) - may need gradient clipping")
                    else:
                        self.issues.append("No gradients computed - model may not be trainable")
                
            except Exception as e:
                self.issues.append(f"Model forward pass failed: {str(e)}")
                self.logger.error(f"❌ Forward pass error: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.issues.append(f"Model validation failed: {str(e)}")
            self.logger.error(f"❌ Model validation error: {e}")
            return False
    
    def validate_losses(self) -> bool:
        """Validate loss functions and weights."""
        self.logger.info("🔍 Validating Loss Functions...")
        
        try:
            # Create loss function
            loss_fn = XTTSLoss(
                mel_loss_weight=self.config.training.mel_loss_weight,
                kl_loss_weight=self.config.training.kl_loss_weight,
                use_adaptive_weights=self.config.training.use_adaptive_loss_weights,
                loss_smoothing_factor=self.config.training.loss_smoothing_factor,
                use_label_smoothing=self.config.training.use_label_smoothing,
                use_huber_loss=self.config.training.use_huber_loss
            )
            
            self.logger.info("✅ Loss function created successfully")
            
            # Test loss computation with dummy data
            batch_size = 2
            mel_len = 256
            
            y_true = {
                'mel_target': tf.random.normal((batch_size, mel_len, self.config.data.n_mels)),
                'stop_target': tf.random.uniform((batch_size, mel_len), 0, 2, dtype=tf.int32),
                'mel_lengths': tf.constant([mel_len-10, mel_len-20], dtype=tf.int32)
            }
            
            y_pred = {
                'mel_output': tf.random.normal((batch_size, mel_len, self.config.data.n_mels)),
                'stop_output': tf.random.uniform((batch_size, mel_len), 0, 1),
                'attention_weights': tf.random.uniform((batch_size, mel_len, 128), 0, 1)
            }
            
            # Compute loss
            total_loss = loss_fn(y_true, y_pred)
            
            # Validate loss
            if tf.math.is_nan(total_loss):
                self.issues.append("Loss function returns NaN")
                return False
            elif tf.math.is_inf(total_loss):
                self.issues.append("Loss function returns Inf")
                return False
            
            self.results['test_loss_value'] = float(total_loss)
            self.logger.info(f"Test loss value: {total_loss:.6f}")
            
            # Check individual loss components
            if hasattr(loss_fn, 'get_losses'):
                individual_losses = loss_fn.get_losses()
                self.logger.info("Individual loss components:")
                for name, value in individual_losses.items():
                    if isinstance(value, tf.Tensor):
                        loss_val = float(value)
                        self.logger.info(f"  {name}: {loss_val:.6f}")
                        
                        if tf.math.is_nan(value):
                            self.issues.append(f"Loss component {name} is NaN")
                        elif tf.math.is_inf(value):
                            self.issues.append(f"Loss component {name} is Inf")
            
            # Check loss weights
            mel_weight = self.config.training.mel_loss_weight
            kl_weight = self.config.training.kl_loss_weight
            
            self.logger.info(f"Loss weights - Mel: {mel_weight}, KL: {kl_weight}")
            
            if mel_weight <= 0 or kl_weight <= 0:
                self.issues.append("Loss weights should be positive")
            
            if mel_weight > 100:
                self.recommendations.append(f"Mel loss weight ({mel_weight}) is quite high - consider reducing if loss is unstable")
            
            return True
            
        except Exception as e:
            self.issues.append(f"Loss validation failed: {str(e)}")
            self.logger.error(f"❌ Loss validation error: {e}")
            return False
    
    def validate_training_step(self) -> bool:
        """Validate a complete training step."""
        self.logger.info("🔍 Validating Training Step...")
        
        try:
            # Create components
            model = XTTS(self.config.model)
            loss_fn = XTTSLoss(
                mel_loss_weight=self.config.training.mel_loss_weight,
                kl_loss_weight=self.config.training.kl_loss_weight
            )
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config.training.learning_rate,
                beta_1=self.config.training.beta1,
                beta_2=self.config.training.beta2,
                epsilon=self.config.training.eps,
                weight_decay=self.config.training.weight_decay
            )
            
            # Create dummy training batch
            batch_size = 2
            seq_len = 128
            mel_len = 256
            
            batch = {
                'text_input': tf.random.uniform((batch_size, seq_len), 0, 100, dtype=tf.int32),
                'mel_input': tf.random.normal((batch_size, mel_len, self.config.data.n_mels)),
                'mel_target': tf.random.normal((batch_size, mel_len, self.config.data.n_mels)),
                'stop_target': tf.random.uniform((batch_size, mel_len), 0, 2, dtype=tf.int32),
                'mel_lengths': tf.constant([mel_len-10, mel_len-20], dtype=tf.int32),
                'speaker_id': tf.zeros((batch_size,), dtype=tf.int32)
            }
            
            # Perform training step
            start_time = time.time()
            
            with tf.GradientTape() as tape:
                # Forward pass
                output = model(batch, training=True)
                
                # Compute loss
                y_true = {k: v for k, v in batch.items() if k.endswith('_target') or k.endswith('_lengths')}
                loss = loss_fn(y_true, output)
            
            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            step_time = time.time() - start_time
            
            self.results['training_step_time'] = step_time
            self.results['training_loss'] = float(loss)
            
            self.logger.info(f"Training step completed in {step_time:.3f}s")
            self.logger.info(f"Training loss: {loss:.6f}")
            
            # Validate gradients
            grad_norms = []
            zero_grads = 0
            
            for grad in gradients:
                if grad is not None:
                    grad_norm = tf.norm(grad)
                    grad_norms.append(grad_norm)
                    if grad_norm == 0:
                        zero_grads += 1
                else:
                    zero_grads += 1
            
            if grad_norms:
                avg_grad_norm = tf.reduce_mean(grad_norms)
                max_grad_norm = tf.reduce_max(grad_norms)
                
                self.results['avg_gradient_norm'] = float(avg_grad_norm)
                self.results['max_gradient_norm'] = float(max_grad_norm)
                
                self.logger.info(f"Gradient norms - Avg: {avg_grad_norm:.6f}, Max: {max_grad_norm:.6f}")
                
                if zero_grads > len(gradients) * 0.5:
                    self.issues.append(f"Many gradients are zero ({zero_grads}/{len(gradients)})")
                
                if max_grad_norm > 10:
                    self.recommendations.append(f"Large gradients detected - consider gradient clipping (current max: {max_grad_norm:.3f})")
            
            if step_time > 2.0:
                self.issues.append(f"Training step is slow ({step_time:.3f}s)")
                self.recommendations.append("Consider optimizing data pipeline or reducing model complexity")
            
            return True
            
        except Exception as e:
            self.issues.append(f"Training step validation failed: {str(e)}")
            self.logger.error(f"❌ Training step validation error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_quick_test(self) -> Dict[str, Any]:
        """Run a quick validation test."""
        self.logger.info("🚀 Running Quick Validation Test...")
        
        results = {
            'environment': self.validate_environment(),
            'dataset': self.validate_dataset(),
            'model': self.validate_model(),
            'losses': self.validate_losses()
        }
        
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation including training step test."""
        self.logger.info("🚀 Running Full Validation...")
        
        results = self.run_quick_test()
        
        # Add training step validation
        results['training_step'] = self.validate_training_step()
        
        return results
    
    def generate_report(self, validation_results: Dict[str, Any]) -> None:
        """Generate validation report."""
        self.logger.info("\n" + "="*80)
        self.logger.info("📋 VALIDATION REPORT")
        self.logger.info("="*80)
        
        # Summary
        total_tests = len(validation_results)
        passed_tests = sum(1 for result in validation_results.values() if result)
        
        self.logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            self.logger.info("🎉 ALL VALIDATIONS PASSED!")
        else:
            self.logger.info("⚠️ SOME VALIDATIONS FAILED")
        
        # Detailed results
        self.logger.info("\nDetailed Results:")
        for test_name, result in validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"  {test_name}: {status}")
        
        # Issues found
        if self.issues:
            self.logger.info(f"\n🚨 Issues Found ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                self.logger.info(f"  {i}. {issue}")
        
        # Recommendations
        if self.recommendations:
            self.logger.info(f"\n💡 Recommendations ({len(self.recommendations)}):")
            for i, rec in enumerate(self.recommendations, 1):
                self.logger.info(f"  {i}. {rec}")
        
        # Performance metrics
        if self.results:
            self.logger.info("\n📊 Performance Metrics:")
            for key, value in self.results.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.6f}")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        self.logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive MyXTTS Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-path", 
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run quick validation test (excludes training step test)"
    )
    
    parser.add_argument(
        "--full-validation", 
        action="store_true",
        help="Run full validation including training step test"
    )
    
    args = parser.parse_args()
    
    if not args.quick_test and not args.full_validation:
        print("Error: Must specify either --quick-test or --full-validation")
        sys.exit(1)
    
    # Create validator
    validator = ComprehensiveValidator(args.data_path, args.config)
    
    try:
        # Run validation
        if args.quick_test:
            results = validator.run_quick_test()
        else:
            results = validator.run_full_validation()
        
        # Generate report
        validator.generate_report(results)
        
        # Exit with appropriate code
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\nValidation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()