#!/usr/bin/env python3
"""
GPU Utilization and Performance Diagnostic Tool for MyXTTS

This script diagnoses GPU-related issues including low utilization, oscillations,
memory problems, and multi-GPU coordination issues.

Usage:
    python utilities/diagnose_gpu_issues.py --profile-steps 100
    python utilities/diagnose_gpu_issues.py --check-config config.yaml
"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class GPUDiagnostic:
    """Diagnostic tool for GPU-related issues"""
    
    def __init__(self, config_path: Optional[str] = None, profile_steps: int = 0):
        self.config_path = config_path
        self.profile_steps = profile_steps
        self.issues = []
        self.gpu_available = False
        self.gpu_info = []
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # Try to import GPU libraries
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check if GPUs are available"""
        try:
            import tensorflow as tf
            self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            if self.gpu_available:
                gpus = tf.config.list_physical_devices('GPU')
                self.gpu_info = [{"name": gpu.name, "device_type": gpu.device_type} for gpu in gpus]
        except:
            self.gpu_available = False
    
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
    
    def check_gpu_config(self) -> List[Tuple[str, str]]:
        """Check GPU-related configuration"""
        self.print_header("GPU Configuration Check")
        issues = []
        
        # Check if static shapes are enabled
        enable_static = self.config.get('training', {}).get('enable_static_shapes', False)
        if not enable_static:
            issue = ("ERROR", "Static shapes NOT enabled - will cause severe GPU utilization issues!")
            issues.append(issue)
            self.print_issue(*issue)
            print(f"   {Colors.YELLOW}Impact: GPU utilization will oscillate (90% → 5% → 90%){Colors.RESET}")
            print(f"   {Colors.YELLOW}Fix: Add 'enable_static_shapes: true' or use --enable-static-shapes{Colors.RESET}")
        else:
            self.print_issue("INFO", "Static shapes enabled - good for GPU stability")
        
        # Check multi-GPU configuration
        data_gpu = self.config.get('training', {}).get('data_gpu', None)
        model_gpu = self.config.get('training', {}).get('model_gpu', None)
        
        if data_gpu is not None and model_gpu is not None:
            self.print_issue("INFO", f"Multi-GPU mode detected: data_gpu={data_gpu}, model_gpu={model_gpu}")
            
            # Check if memory isolation is enabled
            memory_isolation = self.config.get('training', {}).get('enable_memory_isolation', False)
            if not memory_isolation:
                issue = ("WARNING", "Memory isolation not enabled for multi-GPU - may have bottlenecks")
                issues.append(issue)
                self.print_issue(*issue)
                print(f"   {Colors.YELLOW}Recommendation: Enable enable_memory_isolation for better performance{Colors.RESET}")
            else:
                self.print_issue("INFO", "Memory isolation enabled for multi-GPU")
            
            # Check if GPU memory limits are set
            data_gpu_mem = self.config.get('training', {}).get('data_gpu_memory', None)
            model_gpu_mem = self.config.get('training', {}).get('model_gpu_memory', None)
            
            if data_gpu_mem is None or model_gpu_mem is None:
                issue = ("WARNING", "GPU memory limits not set - may cause OOM errors")
                issues.append(issue)
                self.print_issue(*issue)
                print(f"   {Colors.YELLOW}Recommendation: Set data_gpu_memory and model_gpu_memory{Colors.RESET}")
        else:
            self.print_issue("INFO", "Single-GPU mode")
            
            # Check if buffer is configured for single GPU
            buffer_size = self.config.get('training', {}).get('buffer_size', 0)
            if isinstance(buffer_size, str):
                buffer_size = int(buffer_size)
            
            if buffer_size > 0:
                self.print_issue("INFO", f"Data prefetch buffer configured: {buffer_size}")
            else:
                issue = ("WARNING", "No data prefetch buffer - may have GPU starvation")
                issues.append(issue)
                self.print_issue(*issue)
                print(f"   {Colors.YELLOW}Recommendation: Add buffer_size: 100 for better GPU utilization{Colors.RESET}")
        
        # Check mixed precision
        use_mixed = self.config.get('training', {}).get('use_mixed_precision', False)
        if use_mixed:
            self.print_issue("INFO", "Mixed precision enabled - efficient GPU memory usage")
        else:
            issue = ("INFO", "Mixed precision not enabled - may use more GPU memory")
            issues.append(issue)
            self.print_issue(*issue)
            print(f"   {Colors.YELLOW}Consider: Enable use_mixed_precision for better efficiency{Colors.RESET}")
        
        return issues
    
    def check_gpu_availability_issues(self) -> List[Tuple[str, str]]:
        """Check GPU availability and driver issues"""
        self.print_header("GPU Availability Check")
        issues = []
        
        if not self.gpu_available:
            issue = ("ERROR", "No GPU detected! Training will be very slow on CPU")
            issues.append(issue)
            self.print_issue(*issue)
            print(f"   {Colors.YELLOW}Possible causes:{Colors.RESET}")
            print(f"   - CUDA/cuDNN not installed")
            print(f"   - TensorFlow GPU not installed")
            print(f"   - GPU driver issues")
            print(f"   {Colors.YELLOW}Fix: Install tensorflow-gpu and CUDA/cuDNN{Colors.RESET}")
        else:
            self.print_issue("INFO", f"GPU(s) detected: {len(self.gpu_info)} device(s)")
            for i, gpu in enumerate(self.gpu_info):
                print(f"   GPU {i}: {gpu['name']}")
        
        return issues
    
    def check_memory_issues(self) -> List[Tuple[str, str]]:
        """Check for potential memory issues"""
        self.print_header("Memory Configuration Check")
        issues = []
        
        # Check batch size
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        if isinstance(batch_size, str):
            batch_size = int(batch_size)
        model_size = self.config.get('model', {}).get('model_size', 'normal')
        
        # Estimate memory requirements
        estimated_memory = {
            'tiny': batch_size * 200,    # MB per sample
            'small': batch_size * 400,
            'normal': batch_size * 800,
            'big': batch_size * 1500
        }
        
        if model_size in estimated_memory:
            est_mem = estimated_memory[model_size]
            self.print_issue("INFO", f"Estimated GPU memory needed: ~{est_mem}MB for {model_size} model with batch_size {batch_size}")
            
            if est_mem > 16000:  # More than 16GB
                issue = ("WARNING", f"High memory requirement (~{est_mem}MB) - may cause OOM errors")
                issues.append(issue)
                self.print_issue(*issue)
                print(f"   {Colors.YELLOW}Recommendations:{Colors.RESET}")
                print(f"   - Reduce batch_size")
                print(f"   - Enable gradient_accumulation")
                print(f"   - Enable gradient_checkpointing")
        
        # Check gradient accumulation
        grad_accum = self.config.get('training', {}).get('gradient_accumulation_steps', 1)
        if isinstance(grad_accum, str):
            grad_accum = int(grad_accum)
        
        if grad_accum > 1:
            effective_batch = batch_size * grad_accum
            self.print_issue("INFO", f"Gradient accumulation enabled: effective batch_size = {effective_batch}")
        
        # Check gradient checkpointing
        checkpointing = self.config.get('model', {}).get('enable_gradient_checkpointing', False)
        if checkpointing:
            self.print_issue("INFO", "Gradient checkpointing enabled - saves memory at cost of speed")
        elif model_size in ['normal', 'big']:
            issue = ("WARNING", "Gradient checkpointing not enabled for large model")
            issues.append(issue)
            self.print_issue(*issue)
            print(f"   {Colors.YELLOW}Consider: Enable to reduce memory usage{Colors.RESET}")
        
        return issues
    
    def check_retracing_issues(self) -> List[Tuple[str, str]]:
        """Check for tf.function retracing issues"""
        self.print_header("TF.Function Retracing Check")
        issues = []
        
        enable_static = self.config.get('training', {}).get('enable_static_shapes', False)
        
        if not enable_static:
            issue = ("ERROR", "Static shapes disabled - WILL cause excessive retracing!")
            issues.append(issue)
            self.print_issue(*issue)
            print(f"   {Colors.YELLOW}Symptoms:{Colors.RESET}")
            print(f"   - GPU utilization oscillates: 90% → 5% → 90% → 5%")
            print(f"   - Training is 30x slower than expected")
            print(f"   - Lots of 'Tracing' warnings in logs")
            print(f"   {Colors.YELLOW}Fix: Enable enable_static_shapes immediately!{Colors.RESET}")
        else:
            self.print_issue("INFO", "Static shapes enabled - retracing should be minimal")
        
        return issues
    
    def profile_gpu_utilization(self):
        """Profile GPU utilization during training"""
        if not self.gpu_available:
            self.print_issue("WARNING", "Cannot profile GPU - no GPU detected")
            return
        
        self.print_header(f"GPU Utilization Profiling ({self.profile_steps} steps)")
        
        try:
            import GPUtil
            
            print("Monitoring GPU utilization...")
            print("(This is a placeholder - real profiling would run during training)")
            print()
            print("To profile during actual training:")
            print("1. Start training with --enable-profiling")
            print("2. Monitor GPU with: watch -n 1 nvidia-smi")
            print("3. Check for oscillation patterns")
            print()
            print(f"{Colors.YELLOW}Common patterns and their causes:{Colors.RESET}")
            print("• 90% → 5% → 90% → 5%  = Retracing issue (enable static shapes)")
            print("• Consistent 20-40%     = Data pipeline bottleneck (increase buffer)")
            print("• Spiky usage           = Batch size too small")
            print("• One GPU idle          = Multi-GPU not configured properly")
            
        except ImportError:
            self.print_issue("INFO", "GPUtil not installed - cannot profile GPU")
            print(f"   {Colors.YELLOW}Install with: pip install GPUtil{Colors.RESET}")
    
    def generate_recommendations(self):
        """Generate recommendations based on detected issues"""
        self.print_header("GPU Optimization Recommendations")
        
        recommendations = []
        
        # Always recommend static shapes if not enabled
        if not self.config.get('training', {}).get('enable_static_shapes', False):
            recommendations.append(
                "🔴 CRITICAL: Enable static shapes immediately!\n"
                "   Add to config: enable_static_shapes: true\n"
                "   Or use flag: --enable-static-shapes"
            )
        
        # Multi-GPU recommendations
        data_gpu = self.config.get('training', {}).get('data_gpu', None)
        model_gpu = self.config.get('training', {}).get('model_gpu', None)
        
        if data_gpu is not None and model_gpu is not None:
            if not self.config.get('training', {}).get('enable_memory_isolation', False):
                recommendations.append(
                    "🟡 For multi-GPU: Enable memory isolation\n"
                    "   Add: enable_memory_isolation: true\n"
                    "   Benefits: 80-95% GPU utilization vs 50-70%"
                )
        else:
            # Single GPU recommendations
            if not self.config.get('training', {}).get('buffer_size', 0):
                recommendations.append(
                    "🟡 For single-GPU: Add data prefetch buffer\n"
                    "   Add: buffer_size: 100\n"
                    "   Benefits: Better GPU utilization, less starvation"
                )
        
        # Memory optimization recommendations
        if not self.config.get('training', {}).get('use_mixed_precision', False):
            recommendations.append(
                "🟢 Consider: Enable mixed precision training\n"
                "   Add: use_mixed_precision: true\n"
                "   Benefits: 2x less memory, potentially faster"
            )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}\n")
        else:
            print(f"{Colors.GREEN}GPU configuration looks optimal!{Colors.RESET}")
    
    def generate_report(self):
        """Generate comprehensive GPU diagnostic report"""
        self.print_header("MyXTTS GPU Diagnostic Report")
        
        print(f"{Colors.BOLD}Configuration File:{Colors.RESET} {self.config_path or 'Not provided'}")
        if self.profile_steps > 0:
            print(f"{Colors.BOLD}Profile Steps:{Colors.RESET} {self.profile_steps}")
        print()
        
        # Run all checks
        all_issues = []
        all_issues.extend(self.check_gpu_availability_issues())
        all_issues.extend(self.check_gpu_config())
        all_issues.extend(self.check_memory_issues())
        all_issues.extend(self.check_retracing_issues())
        
        if self.profile_steps > 0:
            self.profile_gpu_utilization()
        
        # Recommendations
        self.generate_recommendations()
        
        # Summary
        self.print_header("Diagnostic Summary")
        
        errors = [i for i in all_issues if i[0] == "ERROR"]
        warnings = [i for i in all_issues if i[0] == "WARNING"]
        
        print(f"{Colors.RED}❌ Errors:   {len(errors)}{Colors.RESET}")
        print(f"{Colors.YELLOW}⚠️  Warnings: {len(warnings)}{Colors.RESET}")
        
        print()
        if len(errors) == 0 and len(warnings) == 0:
            print(f"{Colors.BOLD}{Colors.GREEN}✅ GPU configuration looks good!{Colors.RESET}")
        elif len(errors) == 0:
            print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  Some warnings - review recommendations{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}{Colors.RED}❌ Critical GPU issues found - fix before training!{Colors.RESET}")
        
        print()
        return len(errors)


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose GPU-related issues in MyXTTS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check GPU configuration
  python utilities/diagnose_gpu_issues.py --check-config config.yaml
  
  # Profile GPU utilization
  python utilities/diagnose_gpu_issues.py --profile-steps 100
  
  # Full diagnostic
  python utilities/diagnose_gpu_issues.py --check-config config.yaml --profile-steps 100
        """
    )
    
    parser.add_argument(
        '--check-config',
        type=str,
        help='Path to configuration file to check'
    )
    
    parser.add_argument(
        '--profile-steps',
        type=int,
        default=0,
        help='Number of steps to profile GPU utilization'
    )
    
    args = parser.parse_args()
    
    # Create diagnostic tool
    diagnostic = GPUDiagnostic(
        config_path=args.check_config,
        profile_steps=args.profile_steps
    )
    
    # Generate report
    error_count = diagnostic.generate_report()
    
    sys.exit(1 if error_count > 0 else 0)


if __name__ == '__main__':
    main()
