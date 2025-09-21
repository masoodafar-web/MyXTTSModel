#!/usr/bin/env python3
"""
Enhanced Training Monitor and Loss Convergence Optimizer

This script provides advanced training monitoring and automatic optimization
to address slow loss convergence issues in MyXTTS training.

Usage:
    python enhanced_training_monitor.py --data-path ./data/ljspeech --monitor-training
    python enhanced_training_monitor.py --config config.yaml --optimize-losses
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from myxtts.config.config import XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.data.ljspeech import LJSpeechDataset
from myxtts.training.losses import XTTSLoss
from myxtts.training.trainer import XTTSTrainer
from myxtts.utils.commons import setup_logging


class EnhancedTrainingMonitor:
    """Enhanced training monitor with convergence optimization."""
    
    def __init__(self, config_path: Optional[str] = None, data_path: Optional[str] = None):
        """
        Initialize enhanced training monitor.
        
        Args:
            config_path: Optional path to config file
            data_path: Optional path to dataset
        """
        self.logger = setup_logging()
        self.metrics_history = []
        self.loss_history = []
        self.convergence_indicators = {}
        
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            self.config = XTTSConfig.from_yaml(config_path)
        else:
            self.config = XTTSConfig()
            if data_path:
                self.config.data.dataset_path = data_path
    
    def analyze_loss_convergence(self, loss_history: List[float], window_size: int = 100) -> Dict[str, Any]:
        """
        Analyze loss convergence patterns and identify issues.
        
        Args:
            loss_history: List of loss values
            window_size: Window size for moving averages
            
        Returns:
            Convergence analysis results
        """
        if len(loss_history) < window_size:
            return {"error": "Insufficient loss history for analysis"}
        
        # Convert to numpy array for easier manipulation
        losses = np.array(loss_history)
        
        # Calculate moving averages
        moving_avg = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        
        # Calculate convergence rate (negative slope indicates convergence)
        x = np.arange(len(moving_avg))
        if len(x) > 1:
            slope, intercept = np.polyfit(x, moving_avg, 1)
        else:
            slope = 0
        
        # Calculate variance and stability
        recent_losses = losses[-window_size:]
        variance = np.var(recent_losses)
        std_dev = np.std(recent_losses)
        
        # Detect oscillations (high frequency changes)
        if len(losses) > 2:
            diff = np.diff(losses)
            sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
            oscillation_rate = sign_changes / len(diff)
        else:
            oscillation_rate = 0
        
        # Plateau detection (loss not decreasing)
        recent_slope = 0
        if len(moving_avg) > 50:
            recent_x = np.arange(50)
            recent_losses_for_slope = moving_avg[-50:]
            recent_slope, _ = np.polyfit(recent_x, recent_losses_for_slope, 1)
        
        # Convergence metrics
        analysis = {
            'total_samples': len(losses),
            'current_loss': float(losses[-1]),
            'initial_loss': float(losses[0]),
            'improvement': float(losses[0] - losses[-1]),
            'improvement_ratio': float((losses[0] - losses[-1]) / losses[0]) if losses[0] != 0 else 0,
            'convergence_rate': float(slope),
            'recent_convergence_rate': float(recent_slope),
            'variance': float(variance),
            'std_dev': float(std_dev),
            'oscillation_rate': float(oscillation_rate),
            'is_converging': slope < -1e-6,
            'is_stable': variance < 0.01,
            'is_oscillating': oscillation_rate > 0.3,
            'is_plateaued': abs(recent_slope) < 1e-6
        }
        
        # Identify issues and recommendations
        issues = []
        recommendations = []
        
        if not analysis['is_converging']:
            issues.append("Loss is not converging (slope >= 0)")
            recommendations.append("Consider reducing learning rate or adjusting loss weights")
        
        if analysis['improvement_ratio'] < 0.1:
            issues.append(f"Poor overall improvement ({analysis['improvement_ratio']:.1%})")
            recommendations.append("Check model architecture and data quality")
        
        if not analysis['is_stable']:
            issues.append(f"Unstable training (variance: {analysis['variance']:.6f})")
            recommendations.append("Enable loss smoothing or reduce learning rate")
        
        if analysis['is_oscillating']:
            issues.append(f"Loss oscillating (rate: {analysis['oscillation_rate']:.3f})")
            recommendations.append("Reduce learning rate or enable gradient clipping")
        
        if analysis['is_plateaued']:
            issues.append("Loss has plateaued")
            recommendations.append("Consider learning rate scheduling or architecture changes")
        
        analysis['issues'] = issues
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def optimize_loss_weights(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        """
        Dynamically optimize loss weights based on current loss values.
        
        Args:
            current_losses: Dictionary of current loss component values
            
        Returns:
            Optimized loss weights
        """
        # Initialize with current weights
        optimized_weights = {
            'mel_loss_weight': self.config.training.mel_loss_weight,
            'kl_loss_weight': self.config.training.kl_loss_weight,
            'stop_loss_weight': getattr(self.config.training, 'stop_loss_weight', 1.0),
            'attention_loss_weight': getattr(self.config.training, 'attention_loss_weight', 1.0)
        }
        
        # Auto-balancing strategy: adjust weights based on relative loss magnitudes
        if 'mel_loss' in current_losses and 'kl_loss' in current_losses:
            mel_loss = current_losses['mel_loss']
            kl_loss = current_losses['kl_loss']
            
            # Target: mel loss should be 10-20x larger than KL loss for optimal balance
            ideal_ratio = 15.0
            current_ratio = mel_loss / max(kl_loss, 1e-8)
            
            if current_ratio > ideal_ratio * 2:  # Mel loss too dominant
                optimized_weights['mel_loss_weight'] *= 0.9
                optimized_weights['kl_loss_weight'] *= 1.1
                self.logger.info(f"Rebalancing: Mel loss too dominant (ratio: {current_ratio:.1f})")
            elif current_ratio < ideal_ratio / 2:  # KL loss too dominant
                optimized_weights['mel_loss_weight'] *= 1.1
                optimized_weights['kl_loss_weight'] *= 0.9
                self.logger.info(f"Rebalancing: KL loss too dominant (ratio: {current_ratio:.1f})")
        
        # Ensure weights stay within reasonable bounds
        optimized_weights['mel_loss_weight'] = np.clip(optimized_weights['mel_loss_weight'], 10.0, 100.0)
        optimized_weights['kl_loss_weight'] = np.clip(optimized_weights['kl_loss_weight'], 0.1, 10.0)
        
        return optimized_weights
    
    def suggest_learning_rate_schedule(self, current_epoch: int, loss_history: List[float]) -> float:
        """
        Suggest optimal learning rate based on training progress.
        
        Args:
            current_epoch: Current training epoch
            loss_history: Recent loss history
            
        Returns:
            Suggested learning rate
        """
        base_lr = self.config.training.learning_rate
        
        if len(loss_history) < 10:
            return base_lr
        
        # Analyze recent convergence
        recent_losses = loss_history[-10:]
        recent_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
        
        # Adaptive learning rate strategy
        if recent_improvement < 0.001:  # Very slow improvement
            suggested_lr = base_lr * 0.5  # Reduce LR
            reason = "slow convergence"
        elif recent_improvement < 0:  # Loss increasing
            suggested_lr = base_lr * 0.3  # Aggressively reduce LR
            reason = "loss increasing"
        elif recent_improvement > 0.05:  # Very fast improvement
            suggested_lr = base_lr * 1.2  # Slightly increase LR
            reason = "fast convergence"
        else:
            suggested_lr = base_lr  # Keep current LR
            reason = "stable"
        
        # Bound the learning rate
        suggested_lr = np.clip(suggested_lr, 1e-6, 1e-3)
        
        if suggested_lr != base_lr:
            self.logger.info(f"LR adjustment suggested: {base_lr:.2e} -> {suggested_lr:.2e} ({reason})")
        
        return suggested_lr
    
    def create_enhanced_loss_config(self) -> Dict[str, Any]:
        """
        Create enhanced loss configuration for better convergence.
        
        Returns:
            Enhanced loss configuration
        """
        enhanced_config = {
            # Adaptive loss weights
            'use_adaptive_loss_weights': True,
            'loss_smoothing_factor': 0.05,  # Stronger smoothing
            'max_loss_spike_threshold': 1.5,  # Lower threshold for spike detection
            'gradient_norm_threshold': 3.0,  # Lower threshold for gradient monitoring
            
            # Enhanced loss functions
            'use_label_smoothing': True,
            'mel_label_smoothing': 0.03,  # Reduced for better convergence
            'stop_label_smoothing': 0.08,
            'use_huber_loss': True,
            'huber_delta': 0.8,  # Slightly more sensitive
            
            # Improved loss weights (fine-tuned for convergence)
            'mel_loss_weight': 25.0,  # Reduced from 35.0 for better balance
            'kl_loss_weight': 1.5,    # Slightly increased for better regularization
            'stop_loss_weight': 2.0,   # Moderate weight for stop token loss
            'attention_loss_weight': 0.5,  # Light attention loss for alignment
            
            # Learning rate enhancements
            'use_warmup_cosine_schedule': True,
            'warmup_steps': 2000,  # Reduced for faster ramp-up
            'min_learning_rate': 5e-7,  # Lower minimum for fine-tuning
            'cosine_restarts': True,  # Enable periodic restarts
            'restart_period': 10000,  # Restart every 10k steps
            
            # Gradient optimization
            'gradient_clip_norm': 0.8,  # Tighter clipping for stability
            'gradient_accumulation_steps': 2,  # Accumulate gradients for larger effective batch
            
            # Training stability
            'early_stopping_patience': 50,  # Increased patience
            'early_stopping_min_delta': 0.0001,  # Smaller delta for fine-grained stopping
            'loss_scale_patience': 5,  # Patience for loss scaling adjustments
            
            # Monitoring and diagnostics
            'enable_loss_monitoring': True,
            'log_individual_losses': True,
            'track_gradient_norms': True,
            'save_loss_plots': True,
        }
        
        return enhanced_config
    
    def apply_training_optimizations(self, trainer: XTTSTrainer) -> XTTSTrainer:
        """
        Apply training optimizations to the trainer.
        
        Args:
            trainer: XTTSTrainer instance
            
        Returns:
            Optimized trainer
        """
        # Apply enhanced loss configuration
        enhanced_config = self.create_enhanced_loss_config()
        
        # Update trainer's configuration
        for key, value in enhanced_config.items():
            if hasattr(trainer.config.training, key):
                setattr(trainer.config.training, key, value)
                self.logger.info(f"Updated {key}: {value}")
        
        # Recreate loss function with enhanced settings
        if hasattr(trainer, 'loss_fn'):
            trainer.loss_fn = XTTSLoss(
                mel_loss_weight=enhanced_config['mel_loss_weight'],
                kl_loss_weight=enhanced_config['kl_loss_weight'],
                use_adaptive_weights=enhanced_config['use_adaptive_loss_weights'],
                loss_smoothing_factor=enhanced_config['loss_smoothing_factor'],
                use_label_smoothing=enhanced_config['use_label_smoothing'],
                use_huber_loss=enhanced_config['use_huber_loss']
            )
        
        # Update optimizer with new settings
        if hasattr(trainer, 'optimizer'):
            # Create new optimizer with enhanced settings
            trainer.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config.training.learning_rate,
                beta_1=getattr(self.config.training, 'beta1', 0.9),
                beta_2=getattr(self.config.training, 'beta2', 0.999),
                epsilon=getattr(self.config.training, 'eps', 1e-8),
                weight_decay=getattr(self.config.training, 'weight_decay', 1e-6),
                clipnorm=enhanced_config['gradient_clip_norm']
            )
        
        self.logger.info("✅ Applied training optimizations")
        return trainer
    
    def monitor_training_step(self, step: int, loss_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Monitor a single training step and provide real-time optimization.
        
        Args:
            step: Current training step
            loss_dict: Dictionary of loss values
            
        Returns:
            Monitoring results and recommendations
        """
        # Record metrics
        self.loss_history.append(loss_dict.get('total_loss', 0.0))
        self.metrics_history.append({
            'step': step,
            'timestamp': time.time(),
            **loss_dict
        })
        
        # Analyze convergence every 100 steps
        if step % 100 == 0 and len(self.loss_history) > 100:
            convergence_analysis = self.analyze_loss_convergence(self.loss_history)
            self.convergence_indicators = convergence_analysis
            
            # Log convergence status
            if convergence_analysis.get('issues'):
                self.logger.warning(f"Step {step} - Convergence issues detected:")
                for issue in convergence_analysis['issues']:
                    self.logger.warning(f"  • {issue}")
                
                if convergence_analysis.get('recommendations'):
                    self.logger.info("Recommendations:")
                    for rec in convergence_analysis['recommendations']:
                        self.logger.info(f"  • {rec}")
        
        # Real-time optimization suggestions
        suggestions = {}
        
        # Check for NaN or Inf values
        for loss_name, loss_value in loss_dict.items():
            if np.isnan(loss_value) or np.isinf(loss_value):
                suggestions['critical_issue'] = f"Loss {loss_name} is {loss_value}"
                self.logger.error(f"CRITICAL: {loss_name} = {loss_value}")
        
        # Check for loss explosion
        if loss_dict.get('total_loss', 0) > 100:
            suggestions['loss_explosion'] = True
            self.logger.warning(f"Warning: High loss value {loss_dict['total_loss']:.3f}")
        
        # Dynamic weight optimization
        if step % 500 == 0:  # Every 500 steps
            optimized_weights = self.optimize_loss_weights(loss_dict)
            suggestions['optimized_weights'] = optimized_weights
        
        # Learning rate suggestions
        if step % 1000 == 0:  # Every 1000 steps
            recent_losses = [m.get('total_loss', 0) for m in self.metrics_history[-50:]]
            suggested_lr = self.suggest_learning_rate_schedule(step // 1000, recent_losses)
            suggestions['suggested_lr'] = suggested_lr
        
        return {
            'convergence_analysis': self.convergence_indicators,
            'suggestions': suggestions,
            'step_metrics': loss_dict
        }
    
    def generate_training_report(self, output_path: str = "training_report.json") -> None:
        """
        Generate comprehensive training report.
        
        Args:
            output_path: Path to save the report
        """
        if not self.metrics_history:
            self.logger.warning("No training metrics available for report")
            return
        
        # Compile report data
        report = {
            'summary': {
                'total_steps': len(self.metrics_history),
                'training_duration': self.metrics_history[-1]['timestamp'] - self.metrics_history[0]['timestamp'],
                'final_loss': self.loss_history[-1] if self.loss_history else 0,
                'best_loss': min(self.loss_history) if self.loss_history else 0,
                'convergence_analysis': self.convergence_indicators
            },
            'metrics_history': self.metrics_history,
            'loss_history': self.loss_history,
            'configuration': {
                'mel_loss_weight': self.config.training.mel_loss_weight,
                'kl_loss_weight': self.config.training.kl_loss_weight,
                'learning_rate': self.config.training.learning_rate,
                'batch_size': self.config.data.batch_size
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"✅ Training report saved to: {output_path}")
    
    def visualize_training_progress(self, output_dir: str = "training_plots") -> None:
        """
        Create visualization plots for training progress.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.loss_history:
            self.logger.warning("No loss history available for visualization")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Loss progression plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history)
        plt.title('Total Loss Progression')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        
        # Moving average plot
        if len(self.loss_history) > 100:
            window_size = min(100, len(self.loss_history) // 10)
            moving_avg = np.convolve(self.loss_history, np.ones(window_size) / window_size, mode='valid')
            
            plt.subplot(2, 2, 2)
            plt.plot(moving_avg)
            plt.title(f'Moving Average (window={window_size})')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True)
        
        # Individual loss components
        if self.metrics_history:
            plt.subplot(2, 2, 3)
            mel_losses = [m.get('mel_loss', 0) for m in self.metrics_history]
            kl_losses = [m.get('kl_loss', 0) for m in self.metrics_history]
            
            if any(mel_losses):
                plt.plot(mel_losses, label='Mel Loss', alpha=0.7)
            if any(kl_losses):
                plt.plot(kl_losses, label='KL Loss', alpha=0.7)
            
            plt.title('Loss Components')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # Convergence rate analysis
        if len(self.loss_history) > 100:
            plt.subplot(2, 2, 4)
            window_size = 50
            convergence_rates = []
            
            for i in range(window_size, len(self.loss_history)):
                recent_losses = self.loss_history[i-window_size:i]
                x = np.arange(len(recent_losses))
                slope, _ = np.polyfit(x, recent_losses, 1)
                convergence_rates.append(-slope)  # Negative slope means convergence
            
            plt.plot(convergence_rates)
            plt.title('Convergence Rate')
            plt.xlabel('Step')
            plt.ylabel('Rate (higher = faster convergence)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Training plots saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Training Monitor and Loss Convergence Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data-path", 
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--monitor-training", 
        action="store_true",
        help="Monitor training and provide real-time optimization"
    )
    
    parser.add_argument(
        "--optimize-losses", 
        action="store_true",
        help="Create optimized loss configuration"
    )
    
    parser.add_argument(
        "--output", 
        help="Output path for optimized configuration"
    )
    
    args = parser.parse_args()
    
    if not args.monitor_training and not args.optimize_losses:
        print("Error: Must specify either --monitor-training or --optimize-losses")
        sys.exit(1)
    
    # Create monitor
    monitor = EnhancedTrainingMonitor(args.config, args.data_path)
    
    try:
        if args.optimize_losses:
            # Generate enhanced loss configuration
            enhanced_config = monitor.create_enhanced_loss_config()
            
            # Apply to current config
            for key, value in enhanced_config.items():
                if hasattr(monitor.config.training, key):
                    setattr(monitor.config.training, key, value)
            
            # Save optimized configuration
            output_path = args.output or "optimized_training_config.yaml"
            monitor.config.to_yaml(output_path)
            
            print(f"✅ Enhanced training configuration saved to: {output_path}")
            print("\nKey improvements:")
            print("• Adaptive loss weights for better balance")
            print("• Enhanced loss functions with Huber loss and label smoothing")
            print("• Optimized learning rate scheduling")
            print("• Improved gradient handling and monitoring")
            print("• Better training stability and convergence")
        
        if args.monitor_training:
            print("Enhanced training monitor is ready!")
            print("Use monitor.monitor_training_step(step, loss_dict) during training")
            print("Call monitor.generate_training_report() and monitor.visualize_training_progress() after training")
        
    except Exception as e:
        print(f"Operation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()