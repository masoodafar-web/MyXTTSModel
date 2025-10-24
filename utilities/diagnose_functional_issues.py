#!/usr/bin/env python3
"""
Automatic Functional Issue Diagnostic Tool for MyXTTS

This script automatically detects common functional issues in the MyXTTS project,
including convergence problems, model matching issues, and training instabilities.

Usage:
    python utilities/diagnose_functional_issues.py --config config.yaml
    python utilities/diagnose_functional_issues.py --checkpoint path/to/checkpoint --verbose
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class FunctionalIssueDiagnostic:
    """Diagnostic tool for detecting functional issues in MyXTTS"""
    
    def __init__(self, config_path: str = None, checkpoint_path: str = None, verbose: bool = False):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.issues = []
        self.warnings = []
        self.suggestions = []
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
    
    def print_issue(self, severity: str, message: str):
        """Print issue with appropriate color"""
        if severity == "ERROR":
            color = Colors.RED
            icon = "🔴"
        elif severity == "WARNING":
            color = Colors.YELLOW
            icon = "🟡"
        else:
            color = Colors.GREEN
            icon = "🟢"
        
        print(f"{color}{icon} [{severity}] {message}{Colors.RESET}")
    
    def check_convergence_issues(self) -> List[Tuple[str, str]]:
        """Check for potential convergence issues"""
        self.print_header("Checking Convergence Issues")
        issues = []
        
        # Check mel_loss_weight
        mel_weight = self.config.get('training', {}).get('mel_loss_weight', 2.5)
        mel_weight = float(mel_weight) if isinstance(mel_weight, (str, int)) else mel_weight
        if mel_weight > 10.0:
            issue = ("ERROR", f"mel_loss_weight is too high: {mel_weight} (should be 2.5-5.0)")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Set mel_loss_weight to 2.5-5.0 for stable convergence")
        elif mel_weight > 5.0:
            issue = ("WARNING", f"mel_loss_weight is high: {mel_weight} (recommended: 2.5-5.0)")
            issues.append(issue)
            self.print_issue(*issue)
        else:
            self.print_issue("INFO", f"mel_loss_weight is optimal: {mel_weight}")
        
        # Check kl_loss_weight
        kl_weight = self.config.get('training', {}).get('kl_loss_weight', 1.0)
        kl_weight = float(kl_weight) if isinstance(kl_weight, (str, int)) else kl_weight
        if kl_weight > 5.0:
            issue = ("ERROR", f"kl_loss_weight is too high: {kl_weight} (should be 0.5-2.0)")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Set kl_loss_weight to 0.5-2.0")
        elif kl_weight > 2.0:
            issue = ("WARNING", f"kl_loss_weight is high: {kl_weight} (recommended: 0.5-2.0)")
            issues.append(issue)
            self.print_issue(*issue)
        else:
            self.print_issue("INFO", f"kl_loss_weight is optimal: {kl_weight}")
        
        # Check gradient clipping
        grad_clip = self.config.get('training', {}).get('gradient_clip_norm', None)
        if grad_clip is not None:
            grad_clip = float(grad_clip) if isinstance(grad_clip, (str, int)) else grad_clip
        if grad_clip is None:
            issue = ("ERROR", "No gradient clipping configured - risk of gradient explosion!")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Add gradient_clip_norm: 0.5 to config")
        elif grad_clip > 2.0:
            issue = ("WARNING", f"gradient_clip_norm is high: {grad_clip} (recommended: 0.3-1.0)")
            issues.append(issue)
            self.print_issue(*issue)
        else:
            self.print_issue("INFO", f"gradient_clip_norm is optimal: {grad_clip}")
        
        # Check learning rate
        lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        # Convert to float if it's a string
        if isinstance(lr, str):
            lr = float(lr)
        model_size = self.config.get('model', {}).get('model_size', 'normal')
        
        recommended_lr = {
            'tiny': (2e-5, 5e-5),
            'small': (3e-5, 8e-5),
            'normal': (5e-5, 1e-4),
            'big': (3e-5, 8e-5)
        }
        
        if model_size in recommended_lr:
            min_lr, max_lr = recommended_lr[model_size]
            if lr < min_lr or lr > max_lr:
                issue = ("WARNING", f"learning_rate {lr} may not be optimal for {model_size} model (recommended: {min_lr}-{max_lr})")
                issues.append(issue)
                self.print_issue(*issue)
                self.suggestions.append(f"Adjust learning_rate to {min_lr}-{max_lr} for {model_size} model")
            else:
                self.print_issue("INFO", f"learning_rate {lr} is optimal for {model_size} model")
        
        return issues
    
    def check_model_matching_issues(self) -> List[Tuple[str, str]]:
        """Check for model matching and alignment issues"""
        self.print_header("Checking Model Matching Issues")
        issues = []
        
        # Check batch size vs model size
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        batch_size = int(batch_size) if isinstance(batch_size, str) else batch_size
        model_size = self.config.get('model', {}).get('model_size', 'normal')
        
        optimal_batch_sizes = {
            'tiny': (8, 16),
            'small': (16, 32),
            'normal': (32, 64),
            'big': (16, 32)
        }
        
        if model_size in optimal_batch_sizes:
            min_batch, max_batch = optimal_batch_sizes[model_size]
            if batch_size < min_batch or batch_size > max_batch:
                issue = ("WARNING", f"batch_size {batch_size} may not be optimal for {model_size} model (recommended: {min_batch}-{max_batch})")
                issues.append(issue)
                self.print_issue(*issue)
                self.suggestions.append(f"Adjust batch_size to {min_batch}-{max_batch} for {model_size} model")
            else:
                self.print_issue("INFO", f"batch_size {batch_size} is optimal for {model_size} model")
        
        # Check duration predictor
        use_duration = self.config.get('model', {}).get('use_duration_predictor', True)
        if not use_duration:
            issue = ("WARNING", "Duration predictor is disabled - may cause text-audio misalignment")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Enable use_duration_predictor for better alignment")
        else:
            self.print_issue("INFO", "Duration predictor is enabled")
        
        # Check speaker encoder settings
        speaker_dim = self.config.get('model', {}).get('speaker_embedding_dim', 256)
        speaker_dim = int(speaker_dim) if isinstance(speaker_dim, str) else speaker_dim
        if speaker_dim < 128:
            issue = ("WARNING", f"speaker_embedding_dim is too small: {speaker_dim} (recommended: 256-512)")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Increase speaker_embedding_dim to 256 or higher for better voice cloning")
        else:
            self.print_issue("INFO", f"speaker_embedding_dim is adequate: {speaker_dim}")
        
        # Check GST for prosody control
        enable_gst = self.config.get('model', {}).get('enable_gst', False)
        if not enable_gst:
            issue = ("INFO", "GST is disabled - limited prosody control")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Consider enabling GST for better prosody and emotion control")
        else:
            self.print_issue("INFO", "GST is enabled for prosody control")
        
        return issues
    
    def check_training_stability(self) -> List[Tuple[str, str]]:
        """Check for training stability issues"""
        self.print_header("Checking Training Stability")
        issues = []
        
        # Check static shapes
        enable_static = self.config.get('training', {}).get('enable_static_shapes', False)
        if not enable_static:
            issue = ("WARNING", "Static shapes not enabled - may cause GPU utilization issues")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Enable enable_static_shapes to prevent tf.function retracing")
        else:
            self.print_issue("INFO", "Static shapes enabled")
        
        # Check mixed precision
        use_mixed = self.config.get('training', {}).get('use_mixed_precision', False)
        if not use_mixed:
            issue = ("INFO", "Mixed precision not enabled - may use more memory")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Consider enabling use_mixed_precision for better memory efficiency")
        else:
            self.print_issue("INFO", "Mixed precision training enabled")
        
        # Check gradient checkpointing for large models
        model_size = self.config.get('model', {}).get('model_size', 'normal')
        enable_checkpointing = self.config.get('model', {}).get('enable_gradient_checkpointing', False)
        
        if model_size in ['normal', 'big'] and not enable_checkpointing:
            issue = ("INFO", "Gradient checkpointing not enabled for large model - may use more memory")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Consider enabling gradient_checkpointing for memory efficiency")
        
        # Check buffer size for data pipeline
        buffer_size = self.config.get('training', {}).get('buffer_size', 0)
        buffer_size = int(buffer_size) if isinstance(buffer_size, str) else buffer_size
        if buffer_size == 0:
            issue = ("INFO", "No data prefetch buffer configured")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Consider adding buffer_size (e.g., 100) for better GPU utilization")
        else:
            self.print_issue("INFO", f"Data prefetch buffer configured: {buffer_size}")
        
        return issues
    
    def check_output_quality(self) -> List[Tuple[str, str]]:
        """Check for output quality issues"""
        self.print_header("Checking Output Quality Configuration")
        issues = []
        
        # Check vocoder type
        vocoder_type = self.config.get('vocoder', {}).get('type', 'griffin_lim')
        if vocoder_type == 'griffin_lim':
            issue = ("WARNING", "Using griffin_lim vocoder - may produce lower quality audio")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Consider using HiFiGAN or UnivNet for better audio quality")
        else:
            self.print_issue("INFO", f"Using neural vocoder: {vocoder_type}")
        
        # Check mel spectrogram config
        n_mels = self.config.get('audio', {}).get('n_mels', 80)
        n_mels = int(n_mels) if isinstance(n_mels, str) else n_mels
        if n_mels < 80:
            issue = ("WARNING", f"n_mels is too low: {n_mels} (recommended: 80)")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Increase n_mels to 80 for standard TTS quality")
        else:
            self.print_issue("INFO", f"n_mels is adequate: {n_mels}")
        
        # Check hop length
        hop_length = self.config.get('audio', {}).get('hop_length', 256)
        hop_length = int(hop_length) if isinstance(hop_length, str) else hop_length
        if hop_length > 512:
            issue = ("WARNING", f"hop_length is too large: {hop_length} (recommended: 256)")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Reduce hop_length to 256 for better temporal resolution")
        else:
            self.print_issue("INFO", f"hop_length is optimal: {hop_length}")
        
        # Check prosody prediction
        enable_prosody = self.config.get('model', {}).get('enable_prosody_prediction', False)
        if not enable_prosody:
            issue = ("INFO", "Prosody prediction not enabled - output may be monotone")
            issues.append(issue)
            self.print_issue(*issue)
            self.suggestions.append("Consider enabling prosody_prediction for more natural speech")
        
        return issues
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        self.print_header("MyXTTS Functional Issue Diagnostic Report")
        
        print(f"{Colors.BOLD}Configuration File:{Colors.RESET} {self.config_path or 'Not provided'}")
        if self.checkpoint_path:
            print(f"{Colors.BOLD}Checkpoint:{Colors.RESET} {self.checkpoint_path}")
        print()
        
        # Run all checks
        all_issues = []
        all_issues.extend(self.check_convergence_issues())
        all_issues.extend(self.check_model_matching_issues())
        all_issues.extend(self.check_training_stability())
        all_issues.extend(self.check_output_quality())
        
        # Summary
        self.print_header("Diagnostic Summary")
        
        errors = [i for i in all_issues if i[0] == "ERROR"]
        warnings = [i for i in all_issues if i[0] == "WARNING"]
        infos = [i for i in all_issues if i[0] == "INFO"]
        
        print(f"{Colors.RED}🔴 Errors:   {len(errors)}{Colors.RESET}")
        print(f"{Colors.YELLOW}🟡 Warnings: {len(warnings)}{Colors.RESET}")
        print(f"{Colors.GREEN}🟢 Info:     {len(infos)}{Colors.RESET}")
        
        # Suggestions
        if self.suggestions:
            self.print_header("Recommended Actions")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"{Colors.CYAN}{i}. {suggestion}{Colors.RESET}")
        
        # Overall status
        print()
        if len(errors) == 0 and len(warnings) == 0:
            print(f"{Colors.BOLD}{Colors.GREEN}✅ Configuration looks good! No critical issues found.{Colors.RESET}")
        elif len(errors) == 0:
            print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  Some warnings found. Review recommendations above.{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}{Colors.RED}❌ Critical issues found! Please fix errors before training.{Colors.RESET}")
        
        print()
        return len(errors)


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose functional issues in MyXTTS configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic diagnostic
  python utilities/diagnose_functional_issues.py --config config.yaml
  
  # Verbose output
  python utilities/diagnose_functional_issues.py --config config.yaml --verbose
  
  # With checkpoint analysis
  python utilities/diagnose_functional_issues.py --config config.yaml --checkpoint path/to/checkpoint
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for analysis (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create diagnostic tool
    diagnostic = FunctionalIssueDiagnostic(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose
    )
    
    # Generate report
    error_count = diagnostic.generate_report()
    
    # Exit with appropriate code
    sys.exit(1 if error_count > 0 else 0)


if __name__ == '__main__':
    main()
