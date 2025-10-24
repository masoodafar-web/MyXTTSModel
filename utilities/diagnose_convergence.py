#!/usr/bin/env python3
"""
Convergence Issue Diagnostic Tool for MyXTTS

This script analyzes training logs and checkpoints to detect convergence issues,
including loss plateaus, gradient problems, and training instabilities.

Usage:
    python utilities/diagnose_convergence.py --checkpoint path/to/checkpoint
    python utilities/diagnose_convergence.py --log-file training.log
"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# Color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class ConvergenceDiagnostic:
    """Diagnostic tool for convergence issues"""
    
    def __init__(self, checkpoint_path: Optional[str] = None, log_file: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.log_file = log_file
        self.issues = []
        self.loss_history = []
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    
    def print_issue(self, severity: str, message: str):
        """Print issue with color"""
        if severity == "ERROR":
            color = Colors.RED
            icon = "❌"
        elif severity == "WARNING":
            color = Colors.YELLOW
            icon = "⚠️ "
        else:
            color = Colors.GREEN
            icon = "✅"
        
        print(f"{color}{icon} [{severity}] {message}{Colors.RESET}")
    
    def parse_log_file(self) -> List[float]:
        """Parse training log file to extract loss values"""
        if not self.log_file or not os.path.exists(self.log_file):
            return []
        
        loss_values = []
        with open(self.log_file, 'r') as f:
            for line in f:
                # Try to extract loss values from common log formats
                # Format 1: "Loss: 2.345"
                match = re.search(r'Loss:\s*([0-9.]+)', line, re.IGNORECASE)
                if match:
                    loss_values.append(float(match.group(1)))
                    continue
                
                # Format 2: "loss=2.345"
                match = re.search(r'loss=([0-9.]+)', line, re.IGNORECASE)
                if match:
                    loss_values.append(float(match.group(1)))
                    continue
                
                # Format 3: JSON format
                if line.strip().startswith('{'):
                    try:
                        data = json.loads(line.strip())
                        if 'loss' in data:
                            loss_values.append(float(data['loss']))
                        elif 'train_loss' in data:
                            loss_values.append(float(data['train_loss']))
                    except:
                        pass
        
        return loss_values
    
    def detect_plateau(self, losses: List[float], window: int = 50, threshold: float = 0.01) -> bool:
        """Detect if loss has plateaued"""
        if len(losses) < window:
            return False
        
        recent_losses = losses[-window:]
        std_dev = self._std(recent_losses)
        mean_loss = sum(recent_losses) / len(recent_losses)
        
        # If standard deviation is very small relative to mean, it's a plateau
        coefficient_of_variation = std_dev / mean_loss if mean_loss > 0 else 0
        
        return coefficient_of_variation < threshold
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def detect_divergence(self, losses: List[float], window: int = 20) -> bool:
        """Detect if loss is diverging (increasing)"""
        if len(losses) < window:
            return False
        
        recent_losses = losses[-window:]
        first_half = sum(recent_losses[:len(recent_losses)//2]) / (len(recent_losses)//2)
        second_half = sum(recent_losses[len(recent_losses)//2:]) / (len(recent_losses) - len(recent_losses)//2)
        
        # If second half is significantly higher than first half
        return second_half > first_half * 1.2
    
    def detect_oscillation(self, losses: List[float], window: int = 30) -> bool:
        """Detect if loss is oscillating wildly"""
        if len(losses) < window:
            return False
        
        recent_losses = losses[-window:]
        std_dev = self._std(recent_losses)
        mean_loss = sum(recent_losses) / len(recent_losses)
        
        # High coefficient of variation indicates oscillation
        coefficient_of_variation = std_dev / mean_loss if mean_loss > 0 else 0
        
        return coefficient_of_variation > 0.3
    
    def analyze_convergence(self):
        """Analyze convergence patterns"""
        self.print_header("Convergence Analysis")
        
        # Parse log file
        if self.log_file:
            self.loss_history = self.parse_log_file()
            
            if not self.loss_history:
                self.print_issue("WARNING", "Could not parse loss values from log file")
                return
            
            print(f"Parsed {len(self.loss_history)} loss values from log file")
            print(f"Loss range: {min(self.loss_history):.4f} - {max(self.loss_history):.4f}")
            print()
            
            # Check for various issues
            
            # 1. High initial loss
            if self.loss_history and self.loss_history[0] > 100:
                self.print_issue(
                    "ERROR",
                    f"Initial loss is very high: {self.loss_history[0]:.2f} (>100)"
                )
                print(f"   {Colors.YELLOW}Likely cause: mel_loss_weight too high (should be 2.5-5.0){Colors.RESET}")
                self.issues.append("High initial loss - check mel_loss_weight")
            
            # 2. Loss plateau
            if self.detect_plateau(self.loss_history):
                recent_mean = sum(self.loss_history[-50:]) / min(50, len(self.loss_history[-50:]))
                self.print_issue(
                    "WARNING",
                    f"Loss has plateaued around {recent_mean:.4f}"
                )
                print(f"   {Colors.YELLOW}Suggestions:{Colors.RESET}")
                print(f"   - Use --optimization-level plateau_breaker")
                print(f"   - Reduce batch_size if using tiny/small model")
                print(f"   - Check if model capacity is sufficient")
                self.issues.append(f"Loss plateau at {recent_mean:.4f}")
            
            # 3. Loss divergence
            if self.detect_divergence(self.loss_history):
                self.print_issue(
                    "ERROR",
                    "Loss is diverging (increasing over time)"
                )
                print(f"   {Colors.YELLOW}Likely causes:{Colors.RESET}")
                print(f"   - Learning rate too high")
                print(f"   - No gradient clipping")
                print(f"   - Loss weights too high")
                self.issues.append("Loss divergence detected")
            
            # 4. Loss oscillation
            if self.detect_oscillation(self.loss_history):
                self.print_issue(
                    "WARNING",
                    "Loss is oscillating significantly"
                )
                print(f"   {Colors.YELLOW}Suggestions:{Colors.RESET}")
                print(f"   - Reduce learning rate")
                print(f"   - Increase batch_size")
                print(f"   - Enable gradient clipping")
                self.issues.append("High loss oscillation")
            
            # 5. NaN or Inf detection
            has_nan = any(x != x for x in self.loss_history)  # NaN check
            has_inf = any(x == float('inf') or x == float('-inf') for x in self.loss_history)
            
            if has_nan or has_inf:
                self.print_issue(
                    "ERROR",
                    "NaN or Inf values detected in loss"
                )
                print(f"   {Colors.YELLOW}Immediate actions:{Colors.RESET}")
                print(f"   - Add gradient clipping (gradient_clip_norm: 0.5)")
                print(f"   - Reduce learning rate")
                print(f"   - Check for division by zero in loss functions")
                self.issues.append("NaN/Inf in loss values")
            
            # 6. Loss stuck at high value
            if len(self.loss_history) > 100:
                recent_losses = self.loss_history[-100:]
                avg_recent_loss = sum(recent_losses) / len(recent_losses)
                
                if avg_recent_loss > 2.5:
                    self.print_issue(
                        "WARNING",
                        f"Loss stuck at high value: {avg_recent_loss:.4f} (>2.5)"
                    )
                    print(f"   {Colors.YELLOW}Suggestions:{Colors.RESET}")
                    print(f"   - Check mel_loss_weight (should be 2.5-5.0)")
                    print(f"   - Verify batch_size matches model_size")
                    print(f"   - Try --optimization-level plateau_breaker")
                    self.issues.append(f"Loss stuck at {avg_recent_loss:.4f}")
            
            # 7. Good convergence
            if (len(self.loss_history) > 50 and 
                self.loss_history[-1] < 2.0 and 
                not self.detect_plateau(self.loss_history) and
                not self.detect_divergence(self.loss_history)):
                self.print_issue(
                    "INFO",
                    f"Training is converging well! Current loss: {self.loss_history[-1]:.4f}"
                )
        
        else:
            self.print_issue("INFO", "No log file provided for convergence analysis")
    
    def generate_recommendations(self):
        """Generate recommendations based on detected issues"""
        self.print_header("Recommendations")
        
        if not self.issues:
            print(f"{Colors.GREEN}No convergence issues detected! Training looks healthy.{Colors.RESET}")
            return
        
        print(f"Based on detected issues, here are recommendations:\n")
        
        recommendations = []
        
        if any("High initial loss" in i for i in self.issues):
            recommendations.append(
                "1. Reduce mel_loss_weight to 2.5-5.0 in config.yaml"
            )
        
        if any("plateau" in i.lower() for i in self.issues):
            recommendations.append(
                "2. Use plateau breaker:\n"
                "   python train_main.py --optimization-level plateau_breaker"
            )
            recommendations.append(
                "3. Adjust batch size based on model size:\n"
                "   tiny: 8-16, small: 16-32, normal: 32-64"
            )
        
        if any("divergence" in i.lower() for i in self.issues):
            recommendations.append(
                "4. Add gradient clipping:\n"
                "   gradient_clip_norm: 0.5 in config.yaml"
            )
            recommendations.append(
                "5. Reduce learning rate by 50%"
            )
        
        if any("oscillation" in i.lower() for i in self.issues):
            recommendations.append(
                "6. Stabilize training:\n"
                "   - Increase batch_size\n"
                "   - Reduce learning_rate\n"
                "   - Enable --enable-static-shapes"
            )
        
        if any("NaN" in i or "Inf" in i for i in self.issues):
            recommendations.append(
                "7. Fix NaN/Inf issues:\n"
                "   - Set gradient_clip_norm: 0.5\n"
                "   - Reduce all loss weights\n"
                "   - Check data normalization"
            )
        
        for rec in recommendations:
            print(f"{Colors.CYAN}{rec}{Colors.RESET}\n")
    
    def generate_report(self):
        """Generate full diagnostic report"""
        self.print_header("MyXTTS Convergence Diagnostic Report")
        
        if self.log_file:
            print(f"{Colors.BOLD}Log File:{Colors.RESET} {self.log_file}")
        if self.checkpoint_path:
            print(f"{Colors.BOLD}Checkpoint:{Colors.RESET} {self.checkpoint_path}")
        print()
        
        # Run analysis
        self.analyze_convergence()
        self.generate_recommendations()
        
        # Summary
        self.print_header("Summary")
        print(f"Total issues detected: {len(self.issues)}")
        
        if len(self.issues) == 0:
            print(f"{Colors.GREEN}✅ No convergence issues detected!{Colors.RESET}")
            return 0
        else:
            print(f"{Colors.YELLOW}⚠️  Issues found - review recommendations above{Colors.RESET}")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose convergence issues in MyXTTS training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from log file
  python utilities/diagnose_convergence.py --log-file training.log
  
  # Analyze checkpoint
  python utilities/diagnose_convergence.py --checkpoint path/to/checkpoint
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for analysis'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to training log file'
    )
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.log_file:
        parser.error("At least one of --checkpoint or --log-file must be provided")
    
    # Create diagnostic tool
    diagnostic = ConvergenceDiagnostic(
        checkpoint_path=args.checkpoint,
        log_file=args.log_file
    )
    
    # Generate report
    exit_code = diagnostic.generate_report()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
