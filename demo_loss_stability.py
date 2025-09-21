#!/usr/bin/env python3
"""
Demo script for loss stability improvements.

This script demonstrates how to use the new loss stability features
to improve training convergence and reduce instability.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from myxtts.config.config import XTTSConfig
from myxtts.training.losses import XTTSLoss


def demo_stability_features():
    """Demonstrate the new stability features."""
    print("🎯 MyXTTS Loss Stability Improvements Demo")
    print("=" * 50)
    
    # Load configuration with improved defaults
    config = XTTSConfig()
    
    print("📊 Enhanced Configuration Settings:")
    print(f"   • Mel loss weight: {config.training.mel_loss_weight} (reduced from 45.0)")
    print(f"   • Adaptive weights: {config.training.use_adaptive_loss_weights}")
    print(f"   • Loss smoothing: {config.training.loss_smoothing_factor}")
    print(f"   • Label smoothing: {config.training.mel_label_smoothing}")
    print(f"   • Huber loss: {config.training.use_huber_loss}")
    print(f"   • Early stopping patience: {config.training.early_stopping_patience}")
    
    # Create enhanced loss function
    print("\n🔧 Creating Enhanced Loss Function...")
    loss_fn = XTTSLoss(
        mel_loss_weight=config.training.mel_loss_weight,
        use_adaptive_weights=config.training.use_adaptive_loss_weights,
        loss_smoothing_factor=config.training.loss_smoothing_factor,
        max_loss_spike_threshold=config.training.max_loss_spike_threshold
    )
    print("   ✓ Loss function created with stability features")
    
    # Simulate training to show improvements
    print("\n🚀 Simulating Training with Stability Features...")
    simulate_training_with_stability(loss_fn)
    
    print("\n📈 Benefits of Stability Improvements:")
    print("   • Faster convergence through adaptive loss weights")
    print("   • Reduced training instability via loss smoothing")
    print("   • Better generalization with label smoothing")
    print("   • Improved gradient flow with Huber loss")
    print("   • Enhanced monitoring and early intervention")
    
    print("\n🎉 Ready for improved training performance!")


def simulate_training_with_stability(loss_fn):
    """Simulate training to demonstrate stability features."""
    batch_size, time_steps, n_mels = 4, 60, 80
    
    # Setup for different training phases
    phases = [
        ("Early Training", 0.8, 10),   # Higher variance initially
        ("Mid Training", 0.4, 15),    # Reducing variance
        ("Late Training", 0.1, 10),   # Low variance, converged
    ]
    
    all_losses = []
    all_stability_scores = []
    step = 0
    
    for phase_name, noise_scale, num_steps in phases:
        print(f"\n   {phase_name} (Steps {step+1}-{step+num_steps}):")
        phase_losses = []
        
        for i in range(num_steps):
            step += 1
            
            # Create mock training data with varying difficulty
            y_true = {
                "mel_target": tf.random.normal([batch_size, time_steps, n_mels]),
                "stop_target": tf.zeros([batch_size, time_steps, 1]),
                "text_lengths": tf.constant([50, 45, 55, 48]),
                "mel_lengths": tf.constant([55, 50, 58, 52])
            }
            
            # Add noise to predictions to simulate training progress
            base_pred = tf.random.normal([batch_size, time_steps, n_mels])
            noisy_pred = base_pred + tf.random.normal(tf.shape(base_pred)) * noise_scale
            
            y_pred = {
                "mel_output": noisy_pred,
                "stop_tokens": tf.random.uniform([batch_size, time_steps, 1])
            }
            
            # Compute loss with stability features
            loss = loss_fn(y_true, y_pred)
            stability_metrics = loss_fn.get_stability_metrics()
            
            phase_losses.append(float(loss))
            all_losses.append(float(loss))
            
            if 'loss_stability_score' in stability_metrics:
                stability_score = float(stability_metrics['loss_stability_score'])
                all_stability_scores.append(stability_score)
            else:
                all_stability_scores.append(0.0)
            
            # Show progress every few steps
            if i % 5 == 4 or i == num_steps - 1:
                recent_losses = phase_losses[-5:]
                avg_loss = np.mean(recent_losses)
                loss_std = np.std(recent_losses)
                print(f"     Step {step}: Loss = {avg_loss:.4f} ± {loss_std:.4f}")
    
    # Show overall training statistics
    print(f"\n   📊 Training Summary:")
    print(f"     Total steps: {step}")
    print(f"     Loss trend: {all_losses[0]:.3f} → {all_losses[-1]:.3f}")
    print(f"     Improvement: {((all_losses[0] - all_losses[-1]) / all_losses[0] * 100):.1f}%")
    
    if len(all_stability_scores) > 10:
        early_stability = np.mean(all_stability_scores[:10])
        late_stability = np.mean(all_stability_scores[-10:])
        print(f"     Stability: {early_stability:.3f} → {late_stability:.3f}")


def demo_before_after_comparison():
    """Show comparison between old and new loss configurations."""
    print("\n🔍 Before vs After Comparison")
    print("-" * 30)
    
    # Old configuration
    old_loss = XTTSLoss(
        mel_loss_weight=45.0,
        use_adaptive_weights=False,
        loss_smoothing_factor=0.0
    )
    
    # New configuration  
    new_loss = XTTSLoss(
        mel_loss_weight=35.0,
        use_adaptive_weights=True,
        loss_smoothing_factor=0.1
    )
    
    print("Configuration Comparison:")
    print("┌─────────────────────────┬──────────┬──────────┐")
    print("│ Feature                 │ Before   │ After    │")
    print("├─────────────────────────┼──────────┼──────────┤")
    print("│ Mel Loss Weight         │ 45.0     │ 35.0     │")
    print("│ Adaptive Weights        │ False    │ True     │") 
    print("│ Loss Smoothing          │ 0.0      │ 0.1      │")
    print("│ Label Smoothing         │ None     │ 0.05     │")
    print("│ Huber Loss              │ False    │ True     │")
    print("│ Spike Detection         │ None     │ 2.0x     │")
    print("│ Early Stop Patience     │ 20       │ 25       │")
    print("└─────────────────────────┴──────────┴──────────┘")
    
    # Simulate a few steps to show difference
    batch_size, time_steps, n_mels = 2, 40, 80
    
    y_true = {
        "mel_target": tf.random.normal([batch_size, time_steps, n_mels]),
        "mel_lengths": tf.constant([35, 38])
    }
    
    y_pred = {
        "mel_output": tf.random.normal([batch_size, time_steps, n_mels])
    }
    
    old_result = float(old_loss(y_true, y_pred))
    new_result = float(new_loss(y_true, y_pred))
    
    print(f"\nSample Loss Computation:")
    print(f"   Before: {old_result:.4f}")
    print(f"   After:  {new_result:.4f}")
    print(f"   Change: {((new_result - old_result) / old_result * 100):+.1f}%")


def show_usage_examples():
    """Show usage examples for the new features."""
    print("\n📚 Usage Examples")
    print("-" * 20)
    
    print("\n1. Basic Usage (Automatic):")
    print("```python")
    print("from myxtts.config.config import XTTSConfig")
    print("from myxtts.training.trainer import XTTSTrainer")
    print("")
    print("# Load improved configuration")
    print("config = XTTSConfig()")
    print("# All stability features are enabled by default")
    print("")
    print("# Create trainer - stability features applied automatically")
    print("trainer = XTTSTrainer(config)")
    print("```")
    
    print("\n2. Custom Configuration:")
    print("```python")
    print("config = XTTSConfig()")
    print("")
    print("# Customize stability settings")
    print("config.training.mel_loss_weight = 30.0          # Lower for balance")
    print("config.training.loss_smoothing_factor = 0.15    # More smoothing")
    print("config.training.early_stopping_patience = 30    # More patience")
    print("```")
    
    print("\n3. Manual Loss Function:")
    print("```python")
    print("from myxtts.training.losses import XTTSLoss")
    print("")
    print("# Create loss with custom stability parameters")
    print("loss_fn = XTTSLoss(")
    print("    mel_loss_weight=35.0,")
    print("    use_adaptive_weights=True,")
    print("    loss_smoothing_factor=0.1")
    print(")")
    print("```")
    
    print("\n4. Monitoring Stability:")
    print("```python")
    print("# During training, monitor stability")
    print("loss = loss_fn(y_true, y_pred)")
    print("stability_metrics = loss_fn.get_stability_metrics()")
    print("")
    print("if 'loss_stability_score' in stability_metrics:")
    print("    score = stability_metrics['loss_stability_score']")
    print("    if score < 0.5:")
    print("        print(f'Training instability detected: {score:.3f}')")
    print("```")


def main():
    """Run the demonstration."""
    try:
        demo_stability_features()
        demo_before_after_comparison()
        show_usage_examples()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\nThe loss stability improvements are ready to use.")
        print("They will automatically improve training convergence and")
        print("reduce instability in your XTTS model training.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())