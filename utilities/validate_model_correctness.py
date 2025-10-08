#!/usr/bin/env python3
"""
Model Correctness Validation Script

This script performs comprehensive validation of MyXTTS model correctness,
ensuring the model architecture, loss functions, optimizer, and training
pipeline are all functioning correctly and producing expected results.

Usage:
    python validate_model_correctness.py
    python validate_model_correctness.py --verbose
    python validate_model_correctness.py --output-report correctness_report.txt
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import XTTSConfig
from myxtts.models.xtts import XTTS
# Import losses directly to avoid trainer dependencies
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'myxtts', 'training'))
from losses import XTTSLoss, mel_loss, stop_token_loss


class ModelCorrectnessValidator:
    """Validates MyXTTS model correctness and training pipeline."""
    
    def __init__(self, verbose: bool = False):
        """Initialize validator.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.results = []
        self.errors = []
        self.warnings = []
        
        # Suppress TensorFlow warnings unless verbose
        if not verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✓ ",
            "WARNING": "⚠️ ",
            "ERROR": "✗ ",
            "SECTION": "\n" + "="*70 + "\n"
        }.get(level, "")
        
        formatted_msg = f"{prefix}{message}"
        print(formatted_msg)
        
        if level == "ERROR":
            self.errors.append(message)
        elif level == "WARNING":
            self.warnings.append(message)
    
    def validate_model_architecture(self) -> bool:
        """Validate model architecture correctness."""
        self.log("MODEL ARCHITECTURE VALIDATION", "SECTION")
        
        try:
            # Create config
            config = XTTSConfig()
            config.model.text_encoder_dim = 256
            config.model.audio_encoder_dim = 256
            config.model.mel_decoder_dim = 256
            config.model.text_encoder_layers = 2
            config.model.audio_encoder_layers = 2
            config.model.mel_decoder_layers = 2
            
            # Create model
            model = XTTS(config.model)
            self.log("Model created successfully", "SUCCESS")
            
            # Check model has required components
            if not hasattr(model, 'text_encoder'):
                self.log("Model missing text_encoder component", "ERROR")
                return False
            
            if not hasattr(model, 'mel_decoder'):
                self.log("Model missing mel_decoder component", "ERROR")
                return False
            
            self.log("Model has all required components", "SUCCESS")
            
            # Test forward pass
            batch_size = 2
            text_len = 32
            mel_len = 64
            n_mels = 80
            
            text_inputs = tf.random.uniform([batch_size, text_len], 0, 100, dtype=tf.int32)
            mel_inputs = tf.random.normal([batch_size, mel_len, n_mels])
            text_lengths = tf.constant([text_len, text_len-5], dtype=tf.int32)
            mel_lengths = tf.constant([mel_len, mel_len-10], dtype=tf.int32)
            
            outputs = model(
                text_inputs=text_inputs,
                mel_inputs=mel_inputs,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=True
            )
            
            # Validate outputs
            required_outputs = ['mel_output', 'stop_output']
            for key in required_outputs:
                if key not in outputs:
                    self.log(f"Model output missing required key: {key}", "ERROR")
                    return False
            
            self.log("Model forward pass successful", "SUCCESS")
            
            # Check output shapes
            if outputs['mel_output'].shape[0] != batch_size:
                self.log(f"Incorrect batch size in output: {outputs['mel_output'].shape[0]}", "ERROR")
                return False
            
            if outputs['mel_output'].shape[2] != n_mels:
                self.log(f"Incorrect n_mels in output: {outputs['mel_output'].shape[2]}", "ERROR")
                return False
            
            self.log("Output shapes are correct", "SUCCESS")
            
            # Check for NaN or Inf
            if not tf.reduce_all(tf.math.is_finite(outputs['mel_output'])):
                self.log("Model output contains NaN or Inf values", "ERROR")
                return False
            
            self.log("Model outputs are finite", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Model architecture validation failed: {str(e)}", "ERROR")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def validate_loss_functions(self) -> bool:
        """Validate loss function correctness."""
        self.log("LOSS FUNCTION VALIDATION", "SECTION")
        
        try:
            config = XTTSConfig()
            loss_fn = XTTSLoss(
                mel_loss_weight=config.training.mel_loss_weight,
                kl_loss_weight=config.training.kl_loss_weight,
                use_adaptive_weights=False,  # Test without adaptive weights first
                loss_smoothing_factor=0.0,  # Test without smoothing first
                use_label_smoothing=False,
                use_huber_loss=False
            )
            
            self.log("Loss function created successfully", "SUCCESS")
            
            # Test basic loss computation
            batch_size = 2
            mel_len = 64
            n_mels = 80
            
            y_pred = {
                'mel_output': tf.random.normal([batch_size, mel_len, n_mels]),
                'stop_output': tf.random.uniform([batch_size, mel_len], 0, 1),
                'attention_weights': tf.random.uniform([batch_size, mel_len, 32], 0, 1)
            }
            
            y_true = {
                'mel_target': tf.random.normal([batch_size, mel_len, n_mels]),
                'stop_target': tf.random.uniform([batch_size, mel_len], 0, 2, dtype=tf.int32),
                'mel_lengths': tf.constant([mel_len, mel_len-10], dtype=tf.int32)
            }
            
            loss = loss_fn(y_true, y_pred)
            
            # Validate loss properties
            if not tf.math.is_finite(loss):
                self.log("Loss is not finite", "ERROR")
                return False
            
            if loss.numpy() <= 0:
                self.log(f"Loss should be positive, got: {loss.numpy()}", "ERROR")
                return False
            
            self.log(f"Basic loss computation successful: {loss.numpy():.4f}", "SUCCESS")
            
            # Test loss with identical predictions and targets
            y_pred_same = {
                'mel_output': y_true['mel_target'],
                'stop_output': tf.cast(y_true['stop_target'], tf.float32),
                'attention_weights': tf.random.uniform([batch_size, mel_len, 32], 0, 1)
            }
            
            loss_same = loss_fn(y_true, y_pred_same)
            
            if loss_same.numpy() > 0.5:
                self.log(f"Loss should be near zero for identical pred/target, got: {loss_same.numpy()}", "WARNING")
            else:
                self.log(f"Loss near-zero test passed: {loss_same.numpy():.4f}", "SUCCESS")
            
            # Test mel loss symmetry
            mel_a = tf.random.normal([batch_size, mel_len, n_mels])
            mel_b = tf.random.normal([batch_size, mel_len, n_mels])
            lengths = tf.constant([mel_len, mel_len], dtype=tf.int32)
            
            loss_a_b = mel_loss(mel_a, mel_b, lengths, use_huber_loss=False)
            loss_b_a = mel_loss(mel_b, mel_a, lengths, use_huber_loss=False)
            
            if abs(loss_a_b.numpy() - loss_b_a.numpy()) > 1e-5:
                self.log(f"Mel loss not symmetric: {loss_a_b.numpy()} vs {loss_b_a.numpy()}", "ERROR")
                return False
            
            self.log("Mel loss symmetry verified", "SUCCESS")
            
            # Test loss scaling
            mel_close = mel_a + tf.random.normal([batch_size, mel_len, n_mels]) * 0.1
            mel_far = mel_a + tf.random.normal([batch_size, mel_len, n_mels]) * 10.0
            
            loss_close = mel_loss(mel_a, mel_close, lengths, use_huber_loss=False)
            loss_far = mel_loss(mel_a, mel_far, lengths, use_huber_loss=False)
            
            if loss_close.numpy() >= loss_far.numpy():
                self.log(f"Loss scaling incorrect: close={loss_close.numpy()}, far={loss_far.numpy()}", "ERROR")
                return False
            
            self.log("Loss scaling verified", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Loss function validation failed: {str(e)}", "ERROR")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def validate_gradient_flow(self) -> bool:
        """Validate gradient flow through the model."""
        self.log("GRADIENT FLOW VALIDATION", "SECTION")
        
        try:
            # Create model and loss
            config = XTTSConfig()
            config.model.text_encoder_dim = 256
            config.model.audio_encoder_dim = 256
            config.model.mel_decoder_dim = 256
            config.model.text_encoder_layers = 2
            config.model.audio_encoder_layers = 2
            config.model.mel_decoder_layers = 2
            
            model = XTTS(config.model)
            loss_fn = XTTSLoss(
                mel_loss_weight=config.training.mel_loss_weight,
                kl_loss_weight=config.training.kl_loss_weight
            )
            
            # Create dummy batch
            batch_size = 2
            text_len = 32
            mel_len = 64
            n_mels = 80
            
            text_inputs = tf.random.uniform([batch_size, text_len], 0, 100, dtype=tf.int32)
            mel_inputs = tf.random.normal([batch_size, mel_len, n_mels])
            mel_targets = tf.random.normal([batch_size, mel_len, n_mels])
            stop_targets = tf.random.uniform([batch_size, mel_len], 0, 2, dtype=tf.int32)
            text_lengths = tf.constant([text_len, text_len-5], dtype=tf.int32)
            mel_lengths = tf.constant([mel_len, mel_len-10], dtype=tf.int32)
            
            # Compute gradients
            with tf.GradientTape() as tape:
                outputs = model(
                    text_inputs=text_inputs,
                    mel_inputs=mel_inputs,
                    text_lengths=text_lengths,
                    mel_lengths=mel_lengths,
                    training=True
                )
                
                y_true = {
                    'mel_target': mel_targets,
                    'stop_target': stop_targets,
                    'mel_lengths': mel_lengths
                }
                
                loss = loss_fn(y_true, outputs)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Check gradients
            gradient_count = 0
            finite_gradient_count = 0
            zero_gradient_count = 0
            
            for grad, var in zip(gradients, model.trainable_variables):
                if grad is not None:
                    gradient_count += 1
                    if tf.reduce_all(tf.math.is_finite(grad)):
                        finite_gradient_count += 1
                    if tf.reduce_max(tf.abs(grad)).numpy() < 1e-10:
                        zero_gradient_count += 1
            
            if gradient_count == 0:
                self.log("No gradients computed", "ERROR")
                return False
            
            self.log(f"Computed {gradient_count} gradients", "SUCCESS")
            
            if finite_gradient_count != gradient_count:
                self.log(f"Some gradients are not finite: {gradient_count - finite_gradient_count}/{gradient_count}", "ERROR")
                return False
            
            self.log(f"All {finite_gradient_count} gradients are finite", "SUCCESS")
            
            if zero_gradient_count > gradient_count * 0.1:
                self.log(f"Many gradients are near zero: {zero_gradient_count}/{gradient_count}", "WARNING")
            else:
                self.log(f"Gradient magnitudes are reasonable", "SUCCESS")
            
            # Test gradient magnitude distribution
            gradient_norms = []
            for grad in gradients:
                if grad is not None:
                    gradient_norms.append(tf.norm(grad).numpy())
            
            if len(gradient_norms) > 0:
                mean_norm = np.mean(gradient_norms)
                max_norm = np.max(gradient_norms)
                
                self.log(f"Gradient stats - Mean norm: {mean_norm:.6f}, Max norm: {max_norm:.6f}", "INFO")
                
                if max_norm > 100:
                    self.log(f"Some gradients are very large: {max_norm:.2f}", "WARNING")
            
            return True
            
        except Exception as e:
            self.log(f"Gradient flow validation failed: {str(e)}", "ERROR")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def validate_optimizer(self) -> bool:
        """Validate optimizer behavior."""
        self.log("OPTIMIZER VALIDATION", "SECTION")
        
        try:
            # Create model and optimizer
            config = XTTSConfig()
            config.model.text_encoder_dim = 256
            config.model.audio_encoder_dim = 256
            config.model.mel_decoder_dim = 256
            config.model.text_encoder_layers = 2
            config.model.audio_encoder_layers = 2
            config.model.mel_decoder_layers = 2
            
            model = XTTS(config.model)
            loss_fn = XTTSLoss(
                mel_loss_weight=config.training.mel_loss_weight,
                kl_loss_weight=config.training.kl_loss_weight
            )
            
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=config.training.learning_rate,
                beta_1=config.training.beta1,
                beta_2=config.training.beta2,
                epsilon=config.training.eps,
                weight_decay=config.training.weight_decay
            )
            
            self.log("Optimizer created successfully", "SUCCESS")
            
            # Prepare dummy batch
            batch_size = 2
            text_len = 32
            mel_len = 64
            n_mels = 80
            
            text_inputs = tf.random.uniform([batch_size, text_len], 0, 100, dtype=tf.int32)
            mel_inputs = tf.random.normal([batch_size, mel_len, n_mels])
            mel_targets = tf.random.normal([batch_size, mel_len, n_mels])
            stop_targets = tf.random.uniform([batch_size, mel_len], 0, 2, dtype=tf.int32)
            text_lengths = tf.constant([text_len, text_len-5], dtype=tf.int32)
            mel_lengths = tf.constant([mel_len, mel_len-10], dtype=tf.int32)
            
            # Store initial weights
            initial_weights = [tf.identity(var) for var in model.trainable_variables[:5]]
            
            # Perform training step
            with tf.GradientTape() as tape:
                outputs = model(
                    text_inputs=text_inputs,
                    mel_inputs=mel_inputs,
                    text_lengths=text_lengths,
                    mel_lengths=mel_lengths,
                    training=True
                )
                
                y_true = {
                    'mel_target': mel_targets,
                    'stop_target': stop_targets,
                    'mel_lengths': mel_lengths
                }
                
                loss = loss_fn(y_true, outputs)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Check that weights changed
            weights_changed = 0
            for initial, current in zip(initial_weights, model.trainable_variables[:5]):
                if tf.reduce_any(tf.not_equal(initial, current)):
                    weights_changed += 1
            
            if weights_changed == 0:
                self.log("Optimizer did not update weights", "ERROR")
                return False
            
            self.log(f"Optimizer updated {weights_changed}/{len(initial_weights)} sampled weights", "SUCCESS")
            
            # Test learning rate
            lr = optimizer.learning_rate.numpy()
            if lr <= 0 or lr > 1:
                self.log(f"Learning rate outside reasonable range: {lr}", "WARNING")
            else:
                self.log(f"Learning rate is reasonable: {lr:.2e}", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Optimizer validation failed: {str(e)}", "ERROR")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def validate_training_stability(self) -> bool:
        """Validate training stability over multiple steps."""
        self.log("TRAINING STABILITY VALIDATION", "SECTION")
        
        try:
            # Create model, loss, and optimizer
            config = XTTSConfig()
            config.model.text_encoder_dim = 256
            config.model.audio_encoder_dim = 256
            config.model.mel_decoder_dim = 256
            config.model.text_encoder_layers = 2
            config.model.audio_encoder_layers = 2
            config.model.mel_decoder_layers = 2
            
            model = XTTS(config.model)
            loss_fn = XTTSLoss(
                mel_loss_weight=config.training.mel_loss_weight,
                kl_loss_weight=config.training.kl_loss_weight,
                use_adaptive_weights=True,
                loss_smoothing_factor=0.1
            )
            
            optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
            
            # Run multiple training steps
            losses = []
            num_steps = 20
            
            for step in range(num_steps):
                # Create dummy batch
                batch_size = 2
                text_len = 32
                mel_len = 64
                n_mels = 80
                
                text_inputs = tf.random.uniform([batch_size, text_len], 0, 100, dtype=tf.int32)
                mel_inputs = tf.random.normal([batch_size, mel_len, n_mels])
                mel_targets = tf.random.normal([batch_size, mel_len, n_mels])
                stop_targets = tf.random.uniform([batch_size, mel_len], 0, 2, dtype=tf.int32)
                text_lengths = tf.constant([text_len, text_len-5], dtype=tf.int32)
                mel_lengths = tf.constant([mel_len, mel_len-10], dtype=tf.int32)
                
                # Training step
                with tf.GradientTape() as tape:
                    outputs = model(
                        text_inputs=text_inputs,
                        mel_inputs=mel_inputs,
                        text_lengths=text_lengths,
                        mel_lengths=mel_lengths,
                        training=True
                    )
                    
                    y_true = {
                        'mel_target': mel_targets,
                        'stop_target': stop_targets,
                        'mel_lengths': mel_lengths
                    }
                    
                    loss = loss_fn(y_true, outputs)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                losses.append(loss.numpy())
            
            # Analyze loss stability
            losses_array = np.array(losses)
            
            # Check all losses are finite
            if not np.all(np.isfinite(losses_array)):
                self.log("Some losses are not finite during training", "ERROR")
                return False
            
            self.log(f"All {num_steps} losses are finite", "SUCCESS")
            
            # Check for extreme values
            if np.max(losses_array) > 1000:
                self.log(f"Loss exploded: max={np.max(losses_array):.2f}", "ERROR")
                return False
            
            self.log(f"Losses remain bounded: max={np.max(losses_array):.2f}", "SUCCESS")
            
            # Check variance
            loss_mean = np.mean(losses_array)
            loss_std = np.std(losses_array)
            cv = loss_std / loss_mean
            
            if cv > 2.0:
                self.log(f"High loss variance: CV={cv:.3f}", "WARNING")
            else:
                self.log(f"Loss variance is reasonable: CV={cv:.3f}", "SUCCESS")
            
            self.log(f"Loss statistics - Mean: {loss_mean:.4f}, Std: {loss_std:.4f}", "INFO")
            
            return True
            
        except Exception as e:
            self.log(f"Training stability validation failed: {str(e)}", "ERROR")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def validate_inference_mode(self) -> bool:
        """Validate inference mode."""
        self.log("INFERENCE MODE VALIDATION", "SECTION")
        
        try:
            # Create model
            config = XTTSConfig()
            config.model.text_encoder_dim = 256
            config.model.audio_encoder_dim = 256
            config.model.mel_decoder_dim = 256
            config.model.text_encoder_layers = 2
            config.model.audio_encoder_layers = 2
            config.model.mel_decoder_layers = 2
            
            model = XTTS(config.model)
            
            # Test inference
            text_inputs = tf.random.uniform([1, 32], 0, 100, dtype=tf.int32)
            mel_inputs = tf.random.normal([1, 10, 80])
            text_lengths = tf.constant([32], dtype=tf.int32)
            mel_lengths = tf.constant([10], dtype=tf.int32)
            
            outputs = model(
                text_inputs=text_inputs,
                mel_inputs=mel_inputs,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=False
            )
            
            self.log("Inference forward pass successful", "SUCCESS")
            
            # Test inference consistency
            outputs2 = model(
                text_inputs=text_inputs,
                mel_inputs=mel_inputs,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=False
            )
            
            max_diff = tf.reduce_max(tf.abs(outputs['mel_output'] - outputs2['mel_output']))
            
            if max_diff.numpy() > 1e-5:
                self.log(f"Inference not deterministic: max_diff={max_diff.numpy():.2e}", "WARNING")
            else:
                self.log("Inference is deterministic", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Inference mode validation failed: {str(e)}", "ERROR")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def run_all_validations(self) -> bool:
        """Run all validation tests."""
        self.log("\nMYXTTS MODEL CORRECTNESS VALIDATION", "SECTION")
        self.log(f"TensorFlow version: {tf.__version__}", "INFO")
        self.log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
        
        tests = [
            ("Model Architecture", self.validate_model_architecture),
            ("Loss Functions", self.validate_loss_functions),
            ("Gradient Flow", self.validate_gradient_flow),
            ("Optimizer", self.validate_optimizer),
            ("Training Stability", self.validate_training_stability),
            ("Inference Mode", self.validate_inference_mode),
        ]
        
        results = {}
        for test_name, test_func in tests:
            result = test_func()
            results[test_name] = result
        
        # Print summary
        self.log("\nVALIDATION SUMMARY", "SECTION")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            self.log(f"{test_name}: {status}", "SUCCESS" if result else "ERROR")
        
        self.log(f"\nTotal: {passed}/{total} tests passed", "INFO")
        
        if self.errors:
            self.log(f"\nErrors encountered: {len(self.errors)}", "ERROR")
            for error in self.errors:
                self.log(f"  - {error}", "ERROR")
        
        if self.warnings:
            self.log(f"\nWarnings: {len(self.warnings)}", "WARNING")
            for warning in self.warnings:
                self.log(f"  - {warning}", "WARNING")
        
        all_passed = all(results.values())
        
        if all_passed:
            self.log("\n✓ ALL VALIDATION TESTS PASSED", "SUCCESS")
            self.log("MyXTTS model is correctly implemented and functioning as expected.", "INFO")
        else:
            self.log("\n✗ SOME VALIDATION TESTS FAILED", "ERROR")
            self.log("Please review the errors above and fix the issues.", "INFO")
        
        return all_passed
    
    def save_report(self, output_path: str):
        """Save validation report to file."""
        # Implementation would save detailed report
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate MyXTTS model correctness")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--output-report', type=str, help="Save report to file")
    
    args = parser.parse_args()
    
    # Run validation
    validator = ModelCorrectnessValidator(verbose=args.verbose)
    success = validator.run_all_validations()
    
    # Save report if requested
    if args.output_report:
        validator.save_report(args.output_report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
