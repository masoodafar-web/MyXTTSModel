#!/usr/bin/env python3
"""
End-to-End Validation Test for MyXTTS Model

This test validates the complete MyXTTS pipeline from input to output,
ensuring correctness and reliability at every stage.
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import XTTSConfig, ModelConfig, TrainingConfig, DataConfig
from myxtts.models.xtts import XTTS
from myxtts.training.losses import XTTSLoss, mel_loss, stop_token_loss


class TestEndToEndValidation(unittest.TestCase):
    """End-to-end validation tests for MyXTTS model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Create test configuration
        cls.config = XTTSConfig()
        cls.config.model.text_encoder_dim = 256
        cls.config.model.audio_encoder_dim = 256
        cls.config.model.mel_decoder_dim = 256
        cls.config.model.text_encoder_layers = 2
        cls.config.model.audio_encoder_layers = 2
        cls.config.model.mel_decoder_layers = 2
        cls.config.model.n_mels = 80
        
        # Create model
        cls.model = XTTS(cls.config.model)
        
        # Create loss function
        cls.loss_fn = XTTSLoss(
            mel_loss_weight=cls.config.training.mel_loss_weight,
            kl_loss_weight=cls.config.training.kl_loss_weight,
            use_adaptive_weights=cls.config.training.use_adaptive_loss_weights,
            loss_smoothing_factor=cls.config.training.loss_smoothing_factor,
            use_label_smoothing=cls.config.training.use_label_smoothing,
            use_huber_loss=cls.config.training.use_huber_loss
        )
        
        # Test batch dimensions
        cls.batch_size = 2
        cls.text_len = 32
        cls.mel_len = 64
        cls.n_mels = 80
    
    def test_01_model_initialization(self):
        """Test that model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, XTTS)
        print("✓ Model initialization successful")
    
    def test_02_model_forward_pass(self):
        """Test model forward pass with dummy data."""
        # Create dummy inputs
        text_inputs = tf.random.uniform(
            [self.batch_size, self.text_len], 0, 100, dtype=tf.int32
        )
        mel_inputs = tf.random.normal(
            [self.batch_size, self.mel_len, self.n_mels]
        )
        text_lengths = tf.constant([self.text_len, self.text_len-5], dtype=tf.int32)
        mel_lengths = tf.constant([self.mel_len, self.mel_len-10], dtype=tf.int32)
        
        # Forward pass
        outputs = self.model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True
        )
        
        # Validate outputs
        self.assertIn('mel_output', outputs)
        self.assertIn('stop_output', outputs)
        
        # Check output shapes
        mel_output_shape = outputs['mel_output'].shape
        self.assertEqual(mel_output_shape[0], self.batch_size)
        self.assertEqual(mel_output_shape[2], self.n_mels)
        
        # Check for NaN or Inf
        self.assertTrue(tf.reduce_all(tf.math.is_finite(outputs['mel_output'])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(outputs['stop_output'])))
        
        print(f"✓ Forward pass successful. Output shape: {mel_output_shape}")
    
    def test_03_loss_computation(self):
        """Test loss computation with dummy data."""
        # Create dummy predictions and targets
        y_pred = {
            'mel_output': tf.random.normal([self.batch_size, self.mel_len, self.n_mels]),
            'stop_output': tf.random.uniform([self.batch_size, self.mel_len], 0, 1),
            'attention_weights': tf.random.uniform([self.batch_size, self.mel_len, self.text_len], 0, 1)
        }
        
        y_true = {
            'mel_target': tf.random.normal([self.batch_size, self.mel_len, self.n_mels]),
            'stop_target': tf.random.uniform([self.batch_size, self.mel_len], 0, 2, dtype=tf.int32),
            'mel_lengths': tf.constant([self.mel_len, self.mel_len-10], dtype=tf.int32)
        }
        
        # Compute loss
        total_loss = self.loss_fn(y_true, y_pred)
        
        # Validate loss
        self.assertIsNotNone(total_loss)
        self.assertTrue(tf.math.is_finite(total_loss))
        self.assertGreater(total_loss.numpy(), 0)
        
        # Get individual losses
        losses = self.loss_fn.get_losses()
        self.assertIn('mel_loss', losses)
        
        print(f"✓ Loss computation successful. Total loss: {total_loss.numpy():.4f}")
    
    def test_04_gradient_flow(self):
        """Test that gradients flow correctly through the model."""
        # Create dummy batch
        text_inputs = tf.random.uniform(
            [self.batch_size, self.text_len], 0, 100, dtype=tf.int32
        )
        mel_inputs = tf.random.normal([self.batch_size, self.mel_len, self.n_mels])
        mel_targets = tf.random.normal([self.batch_size, self.mel_len, self.n_mels])
        stop_targets = tf.random.uniform(
            [self.batch_size, self.mel_len], 0, 2, dtype=tf.int32
        )
        text_lengths = tf.constant([self.text_len, self.text_len-5], dtype=tf.int32)
        mel_lengths = tf.constant([self.mel_len, self.mel_len-10], dtype=tf.int32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            outputs = self.model(
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
            
            loss = self.loss_fn(y_true, outputs)
        
        # Get gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Check that gradients exist and are finite
        gradient_count = 0
        finite_gradient_count = 0
        
        for grad, var in zip(gradients, self.model.trainable_variables):
            if grad is not None:
                gradient_count += 1
                if tf.reduce_all(tf.math.is_finite(grad)):
                    finite_gradient_count += 1
        
        self.assertGreater(gradient_count, 0, "No gradients computed")
        self.assertEqual(
            gradient_count, 
            finite_gradient_count, 
            f"Some gradients are not finite: {gradient_count - finite_gradient_count}/{gradient_count}"
        )
        
        print(f"✓ Gradient flow validated. {gradient_count} gradients computed, all finite")
    
    def test_05_training_step(self):
        """Test a complete training step."""
        # Create optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            weight_decay=0.01
        )
        
        # Create dummy batch
        text_inputs = tf.random.uniform(
            [self.batch_size, self.text_len], 0, 100, dtype=tf.int32
        )
        mel_inputs = tf.random.normal([self.batch_size, self.mel_len, self.n_mels])
        mel_targets = tf.random.normal([self.batch_size, self.mel_len, self.n_mels])
        stop_targets = tf.random.uniform(
            [self.batch_size, self.mel_len], 0, 2, dtype=tf.int32
        )
        text_lengths = tf.constant([self.text_len, self.text_len-5], dtype=tf.int32)
        mel_lengths = tf.constant([self.mel_len, self.mel_len-10], dtype=tf.int32)
        
        # Training step
        with tf.GradientTape() as tape:
            outputs = self.model(
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
            
            loss = self.loss_fn(y_true, outputs)
        
        # Get and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Verify loss is computed correctly
        self.assertTrue(tf.math.is_finite(loss))
        self.assertGreater(loss.numpy(), 0)
        
        print(f"✓ Training step successful. Loss: {loss.numpy():.4f}")
    
    def test_06_inference_mode(self):
        """Test inference mode (without training=True)."""
        # Create dummy inputs
        text_inputs = tf.random.uniform(
            [1, self.text_len], 0, 100, dtype=tf.int32
        )
        mel_inputs = tf.random.normal([1, 10, self.n_mels])  # Short sequence for inference
        text_lengths = tf.constant([self.text_len], dtype=tf.int32)
        mel_lengths = tf.constant([10], dtype=tf.int32)
        
        # Inference
        outputs = self.model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=False
        )
        
        # Validate outputs
        self.assertIn('mel_output', outputs)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(outputs['mel_output'])))
        
        print(f"✓ Inference mode successful. Output shape: {outputs['mel_output'].shape}")
    
    def test_07_loss_function_properties(self):
        """Test loss function mathematical properties."""
        # Test 1: Loss should be zero when prediction equals target
        batch_size = 2
        mel_len = 32
        n_mels = 80
        
        mel_target = tf.random.normal([batch_size, mel_len, n_mels])
        mel_lengths = tf.constant([mel_len, mel_len], dtype=tf.int32)
        
        # When prediction equals target, loss should be very small
        loss = mel_loss(mel_target, mel_target, mel_lengths, use_huber_loss=False)
        self.assertLess(loss.numpy(), 0.1, "Loss should be near zero when pred==target")
        
        # Test 2: Loss should increase with larger differences
        mel_pred_close = mel_target + tf.random.normal([batch_size, mel_len, n_mels]) * 0.1
        mel_pred_far = mel_target + tf.random.normal([batch_size, mel_len, n_mels]) * 10.0
        
        loss_close = mel_loss(mel_target, mel_pred_close, mel_lengths, use_huber_loss=False)
        loss_far = mel_loss(mel_target, mel_pred_far, mel_lengths, use_huber_loss=False)
        
        self.assertLess(loss_close.numpy(), loss_far.numpy(), 
                       "Loss should increase with larger prediction errors")
        
        # Test 3: Loss should be symmetric
        loss_a_b = mel_loss(mel_target, mel_pred_close, mel_lengths, use_huber_loss=False)
        loss_b_a = mel_loss(mel_pred_close, mel_target, mel_lengths, use_huber_loss=False)
        
        np.testing.assert_almost_equal(
            loss_a_b.numpy(), loss_b_a.numpy(), decimal=5,
            err_msg="Mel loss should be symmetric"
        )
        
        print("✓ Loss function mathematical properties validated")
    
    def test_08_loss_stability_over_iterations(self):
        """Test that loss remains stable over multiple iterations."""
        losses = []
        
        for i in range(10):
            # Create dummy data
            y_pred = {
                'mel_output': tf.random.normal([self.batch_size, self.mel_len, self.n_mels]),
                'stop_output': tf.random.uniform([self.batch_size, self.mel_len], 0, 1),
                'attention_weights': tf.random.uniform([self.batch_size, self.mel_len, self.text_len], 0, 1)
            }
            
            y_true = {
                'mel_target': tf.random.normal([self.batch_size, self.mel_len, self.n_mels]),
                'stop_target': tf.random.uniform([self.batch_size, self.mel_len], 0, 2, dtype=tf.int32),
                'mel_lengths': tf.constant([self.mel_len, self.mel_len-10], dtype=tf.int32)
            }
            
            loss = self.loss_fn(y_true, y_pred)
            losses.append(loss.numpy())
        
        # Check all losses are finite
        self.assertTrue(all(np.isfinite(l) for l in losses), "All losses should be finite")
        
        # Check loss variance is reasonable (not too high)
        loss_std = np.std(losses)
        loss_mean = np.mean(losses)
        cv = loss_std / loss_mean  # Coefficient of variation
        
        self.assertLess(cv, 1.0, f"Loss coefficient of variation too high: {cv:.3f}")
        
        print(f"✓ Loss stability validated. Mean: {loss_mean:.4f}, Std: {loss_std:.4f}, CV: {cv:.4f}")
    
    def test_09_model_output_consistency(self):
        """Test that model produces consistent outputs for the same input."""
        # Create dummy inputs
        text_inputs = tf.random.uniform(
            [1, self.text_len], 0, 100, dtype=tf.int32
        )
        mel_inputs = tf.random.normal([1, self.mel_len, self.n_mels])
        text_lengths = tf.constant([self.text_len], dtype=tf.int32)
        mel_lengths = tf.constant([self.mel_len], dtype=tf.int32)
        
        # Get two outputs with same input (in inference mode)
        outputs1 = self.model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=False
        )
        
        outputs2 = self.model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=False
        )
        
        # Outputs should be identical in inference mode
        mel_diff = tf.reduce_max(tf.abs(outputs1['mel_output'] - outputs2['mel_output']))
        self.assertLess(mel_diff.numpy(), 1e-5, 
                       f"Model outputs should be consistent in inference mode. Max diff: {mel_diff.numpy()}")
        
        print(f"✓ Model output consistency validated. Max difference: {mel_diff.numpy():.2e}")
    
    def test_10_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        print("\n=== End-to-End Pipeline Test ===")
        
        # 1. Data preparation
        text_inputs = tf.random.uniform([2, 50], 0, 100, dtype=tf.int32)
        mel_inputs = tf.random.normal([2, 100, 80])
        mel_targets = tf.random.normal([2, 100, 80])
        stop_targets = tf.zeros([2, 100], dtype=tf.int32)
        stop_targets = tf.concat([
            stop_targets[:, :-1],
            tf.ones([2, 1], dtype=tf.int32)
        ], axis=1)
        text_lengths = tf.constant([50, 45], dtype=tf.int32)
        mel_lengths = tf.constant([100, 90], dtype=tf.int32)
        
        print("  ✓ Step 1: Data preparation complete")
        
        # 2. Model forward pass
        outputs = self.model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True
        )
        
        print(f"  ✓ Step 2: Forward pass complete. Output shape: {outputs['mel_output'].shape}")
        
        # 3. Loss computation
        y_true = {
            'mel_target': mel_targets,
            'stop_target': stop_targets,
            'mel_lengths': mel_lengths
        }
        
        loss = self.loss_fn(y_true, outputs)
        print(f"  ✓ Step 3: Loss computation complete. Loss: {loss.numpy():.4f}")
        
        # 4. Gradient computation
        optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
        
        with tf.GradientTape() as tape:
            outputs = self.model(
                text_inputs=text_inputs,
                mel_inputs=mel_inputs,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=True
            )
            loss = self.loss_fn(y_true, outputs)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        print(f"  ✓ Step 4: Gradient computation complete. {len([g for g in gradients if g is not None])} gradients")
        
        # 5. Optimizer step
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        print("  ✓ Step 5: Optimizer step complete")
        
        # 6. Inference
        infer_outputs = self.model(
            text_inputs=text_inputs[:1],
            mel_inputs=mel_inputs[:1, :10, :],
            text_lengths=text_lengths[:1],
            mel_lengths=tf.constant([10], dtype=tf.int32),
            training=False
        )
        print(f"  ✓ Step 6: Inference complete. Output shape: {infer_outputs['mel_output'].shape}")
        
        print("\n✓ End-to-end pipeline validation successful!")


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEndValidation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
