#!/usr/bin/env python3
"""
Loss Plateau Diagnostic Tool

This script helps diagnose why loss has plateaued and suggests remedies.
It analyzes training logs, configuration, and provides actionable recommendations.
"""

import sys
import os
import re
import argparse
from typing import List, Tuple, Dict
from pathlib import Path


def parse_loss_from_log(log_file: str) -> List[Tuple[int, float]]:
    """Parse loss values from training log."""
    loss_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match patterns like "Epoch 50: loss=2.7123"
            match = re.search(r'[Ee]poch\s+(\d+).*?loss[=:\s]+([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                loss_values.append((epoch, loss))
    
    return loss_values


def detect_plateau(loss_values: List[Tuple[int, float]], window: int = 10, threshold: float = 0.01) -> bool:
    """
    Detect if loss has plateaued.
    
    Args:
        loss_values: List of (epoch, loss) tuples
        window: Number of epochs to check
        threshold: Maximum change to consider a plateau
    
    Returns:
        True if plateau detected
    """
    if len(loss_values) < window:
        return False
    
    recent_losses = [loss for _, loss in loss_values[-window:]]
    
    # Calculate variance
    mean_loss = sum(recent_losses) / len(recent_losses)
    variance = sum((loss - mean_loss) ** 2 for loss in recent_losses) / len(recent_losses)
    std_dev = variance ** 0.5
    
    # Plateau if standard deviation is very small relative to mean
    relative_std = std_dev / (mean_loss + 1e-8)
    
    return relative_std < threshold


def get_loss_statistics(loss_values: List[Tuple[int, float]]) -> Dict[str, float]:
    """Get statistics about loss values."""
    if not loss_values:
        return {}
    
    losses = [loss for _, loss in loss_values]
    
    return {
        'initial_loss': losses[0],
        'current_loss': losses[-1],
        'min_loss': min(losses),
        'max_loss': max(losses),
        'reduction': losses[0] - losses[-1],
        'reduction_percent': ((losses[0] - losses[-1]) / losses[0]) * 100 if losses[0] > 0 else 0,
    }


def suggest_remedies(stats: Dict[str, float], current_config: Dict[str, any]) -> List[str]:
    """Suggest remedies based on loss statistics and current config."""
    suggestions = []
    
    current_loss = stats.get('current_loss', 3.0)
    reduction = stats.get('reduction', 0.0)
    
    # Check if loss is stuck around 2.5-2.7
    if 2.3 <= current_loss <= 2.9 and reduction < 0.2:
        suggestions.append(
            "🎯 **PLATEAU BREAKER RECOMMENDED**: Your loss is stuck around 2.5-2.7.\n"
            "   Run: bash breakthrough_training.sh\n"
            "   Or: python3 train_main.py --optimization-level plateau_breaker"
        )
    
    # Check learning rate
    lr = current_config.get('learning_rate', 1e-4)
    if lr > 5e-5 and current_loss < 3.0:
        suggestions.append(
            f"📉 **REDUCE LEARNING RATE**: Current LR ({lr:.2e}) may be too high for fine-tuning.\n"
            f"   Suggested: 1.5e-5 (plateau_breaker) or 1e-5 (basic)"
        )
    
    # Check mel loss weight
    mel_weight = current_config.get('mel_loss_weight', 2.5)
    if mel_weight > 5.0:
        suggestions.append(
            f"⚠️  **MEL LOSS WEIGHT TOO HIGH**: Current weight ({mel_weight}) exceeds safe range.\n"
            f"   Safe range: 1.0-5.0. Recommended: 2.0-2.5"
        )
    
    # Check gradient clipping
    grad_clip = current_config.get('gradient_clip_norm', 0.5)
    if grad_clip > 0.5 and current_loss < 3.0:
        suggestions.append(
            f"✂️  **TIGHTEN GRADIENT CLIPPING**: Current ({grad_clip}) may be too loose.\n"
            f"   Recommended: 0.3 for plateau breaking"
        )
    
    # Check if using adaptive features
    if not current_config.get('use_adaptive_loss_weights', True):
        suggestions.append(
            "🔧 **ENABLE ADAPTIVE WEIGHTS**: Adaptive loss weights can help balance components.\n"
            "   Set use_adaptive_loss_weights: true in config"
        )
    
    # General suggestions
    if not suggestions:
        if current_loss > 5.0:
            suggestions.append(
                "🚀 **START WITH BASIC**: Loss is still high. Use basic optimization level first.\n"
                "   Run: python3 train_main.py --optimization-level basic"
            )
        else:
            suggestions.append(
                "✅ **CONTINUE TRAINING**: Loss is decreasing well. Continue with current settings."
            )
    
    return suggestions


def analyze_training_progress(log_file: str = None) -> None:
    """Main analysis function."""
    print("=" * 70)
    print("🔍 Loss Plateau Diagnostic Tool")
    print("=" * 70)
    print()
    
    # Parse loss values
    if log_file and os.path.exists(log_file):
        print(f"📊 Analyzing training log: {log_file}")
        loss_values = parse_loss_from_log(log_file)
        
        if not loss_values:
            print("⚠️  No loss values found in log file.")
            print("   Make sure the log contains lines like 'Epoch X: loss=Y.ZZZ'")
            return
        
        print(f"   Found {len(loss_values)} loss measurements")
        print()
        
        # Get statistics
        stats = get_loss_statistics(loss_values)
        
        print("📈 Training Statistics:")
        print(f"   Initial loss: {stats['initial_loss']:.4f}")
        print(f"   Current loss: {stats['current_loss']:.4f}")
        print(f"   Minimum loss: {stats['min_loss']:.4f}")
        print(f"   Total reduction: {stats['reduction']:.4f} ({stats['reduction_percent']:.1f}%)")
        print()
        
        # Detect plateau
        is_plateau = detect_plateau(loss_values)
        
        if is_plateau:
            print("⚠️  **PLATEAU DETECTED**")
            print("   Loss has stopped improving significantly in recent epochs.")
            print()
        else:
            print("✅ Loss is still improving")
            print()
        
    else:
        print("ℹ️  No log file provided. Providing general guidance.")
        stats = {'current_loss': 2.7, 'reduction': 0.1}
        print()
    
    # Mock current config - in real usage, this would be parsed from config files
    current_config = {
        'learning_rate': 1e-4,
        'mel_loss_weight': 2.5,
        'gradient_clip_norm': 0.5,
        'use_adaptive_loss_weights': True,
    }
    
    # Get suggestions
    suggestions = suggest_remedies(stats, current_config)
    
    print("💡 Recommendations:")
    print()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
        print()
    
    print("=" * 70)
    print("📚 Additional Resources:")
    print("   - docs/PLATEAU_BREAKER_USAGE.md - Comprehensive usage guide")
    print("   - docs/PLATEAU_BREAKTHROUGH_GUIDE.md - Technical analysis")
    print("   - docs/FIX_SUMMARY.md - Previous loss fixes")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose loss plateau and suggest remedies"
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to training log file'
    )
    
    args = parser.parse_args()
    
    analyze_training_progress(args.log_file)


if __name__ == '__main__':
    main()
