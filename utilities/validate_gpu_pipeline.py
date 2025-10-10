#!/usr/bin/env python3
"""
GPU Pipeline Configuration Validator

This tool validates that your configuration is properly set up to avoid
the GPU oscillation bottleneck caused by tf.numpy_function in the data pipeline.

It checks:
1. Config file settings (use_tf_native_loading, prefetch_to_gpu, etc.)
2. TF-native loader availability
3. Data pipeline code configuration
4. GPU availability

Usage:
    python utilities/validate_gpu_pipeline.py
    python utilities/validate_gpu_pipeline.py --config configs/config.yaml
    python utilities/validate_gpu_pipeline.py --fix  # Auto-fix config issues
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed")
    sys.exit(1)


class GPUPipelineValidator:
    """Validate GPU pipeline configuration."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize validator."""
        self.config_path = Path(config_path)
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("="*70)
        print("GPU PIPELINE CONFIGURATION VALIDATOR")
        print("="*70)
        print(f"Config file: {self.config_path}")
        print()
        
        all_passed = True
        
        # Check 1: Config file exists
        if not self._check_config_exists():
            all_passed = False
        
        # Check 2: GPU available
        if not self._check_gpu_available():
            all_passed = False
        
        # Check 3: TF-native loader module
        if not self._check_tf_native_loader():
            all_passed = False
        
        # Check 4: Config settings
        if not self._check_config_settings():
            all_passed = False
        
        # Check 5: Data pipeline code
        if not self._check_data_pipeline_code():
            all_passed = False
        
        # Print summary
        self._print_summary()
        
        return all_passed
    
    def _check_config_exists(self) -> bool:
        """Check if config file exists."""
        print("🔍 Checking config file...")
        
        if not self.config_path.exists():
            self.issues.append(f"Config file not found: {self.config_path}")
            print(f"   🔴 FAIL: Config file not found")
            return False
        
        self.successes.append("Config file exists")
        print(f"   ✅ PASS: Config file exists")
        return True
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        print("\n🔍 Checking GPU availability...")
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            self.warnings.append("No GPU detected - will run on CPU (slower)")
            print(f"   ⚠️  WARNING: No GPU detected")
            return True  # Not a failure, just slower
        
        self.successes.append(f"Found {len(gpus)} GPU(s)")
        print(f"   ✅ PASS: Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"      GPU {i}: {gpu.name}")
        return True
    
    def _check_tf_native_loader(self) -> bool:
        """Check if TF-native loader module exists and can be imported."""
        print("\n🔍 Checking TF-native loader module...")
        
        # Check file exists
        tf_native_file = Path(__file__).parent.parent / "myxtts" / "data" / "tf_native_loader.py"
        if not tf_native_file.exists():
            self.issues.append("tf_native_loader.py not found")
            print(f"   🔴 FAIL: tf_native_loader.py not found")
            return False
        
        print(f"   ✅ File exists: tf_native_loader.py")
        
        # Try to import
        try:
            from myxtts.data.tf_native_loader import TFNativeDataLoader
            self.successes.append("TFNativeDataLoader can be imported")
            print(f"   ✅ PASS: TFNativeDataLoader can be imported")
            return True
        except Exception as e:
            self.issues.append(f"Failed to import TFNativeDataLoader: {e}")
            print(f"   🔴 FAIL: Cannot import TFNativeDataLoader")
            print(f"      Error: {e}")
            return False
    
    def _check_config_settings(self) -> bool:
        """Check critical config settings."""
        print("\n🔍 Checking config settings...")
        
        if not self.config_path.exists():
            return False  # Already reported in _check_config_exists
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.issues.append(f"Failed to parse config: {e}")
            print(f"   🔴 FAIL: Cannot parse config file")
            return False
        
        all_passed = True
        
        # Check data section exists
        if 'data' not in config:
            self.issues.append("'data' section missing in config")
            print(f"   🔴 FAIL: 'data' section missing")
            return False
        
        data_config = config['data']
        
        # Critical settings to check
        critical_settings = {
            'use_tf_native_loading': (True, "CRITICAL: Enables GPU-optimized loading"),
            'prefetch_to_gpu': (True, "Important: Enables GPU prefetching"),
            'prefetch_buffer_size': (8, "Recommended: At least 8 for smooth pipeline"),
            'num_workers': (8, "Recommended: At least 8 for parallel loading"),
        }
        
        for setting, (recommended_value, description) in critical_settings.items():
            if setting in data_config:
                actual_value = data_config[setting]
                
                if setting in ['prefetch_buffer_size', 'num_workers']:
                    # For numeric settings, check minimum
                    if actual_value >= recommended_value:
                        print(f"   ✅ {setting}: {actual_value} (good)")
                        self.successes.append(f"{setting} = {actual_value}")
                    else:
                        print(f"   ⚠️  {setting}: {actual_value} (low, recommend >= {recommended_value})")
                        self.warnings.append(f"{setting} = {actual_value} (recommend >= {recommended_value})")
                else:
                    # For boolean settings, check exact match
                    if actual_value == recommended_value:
                        print(f"   ✅ {setting}: {actual_value}")
                        self.successes.append(f"{setting} = {actual_value}")
                    else:
                        print(f"   🔴 {setting}: {actual_value} (should be {recommended_value})")
                        self.issues.append(f"{setting} = {actual_value} (should be {recommended_value})")
                        self.issues.append(f"   Reason: {description}")
                        all_passed = False
            else:
                print(f"   ⚠️  {setting}: not set (will use default)")
                self.warnings.append(f"{setting} not explicitly set in config")
        
        return all_passed
    
    def _check_data_pipeline_code(self) -> bool:
        """Check data pipeline code for proper TF-native support."""
        print("\n🔍 Checking data pipeline code...")
        
        ljspeech_file = Path(__file__).parent.parent / "myxtts" / "data" / "ljspeech.py"
        if not ljspeech_file.exists():
            self.issues.append("ljspeech.py not found")
            print(f"   🔴 FAIL: ljspeech.py not found")
            return False
        
        with open(ljspeech_file, 'r') as f:
            content = f.read()
        
        all_passed = True
        
        # Check for TF-native code path
        if 'use_tf_native = getattr(self.config, \'use_tf_native_loading\'' in content:
            print(f"   ✅ TF-native loading code path exists")
            self.successes.append("TF-native code path in ljspeech.py")
        else:
            print(f"   🔴 TF-native loading code path NOT found")
            self.issues.append("TF-native code path missing in ljspeech.py")
            all_passed = False
        
        # Check for tf.numpy_function (should exist but be conditional)
        if 'tf.numpy_function' in content:
            print(f"   ⚠️  tf.numpy_function present (OK if used as fallback only)")
            self.warnings.append("tf.numpy_function exists (should only be fallback)")
        
        # Check for TFNativeDataLoader import
        if 'from .tf_native_loader import TFNativeDataLoader' in content:
            print(f"   ✅ TFNativeDataLoader import exists")
            self.successes.append("TFNativeDataLoader import in ljspeech.py")
        else:
            print(f"   🔴 TFNativeDataLoader import NOT found")
            self.issues.append("TFNativeDataLoader import missing in ljspeech.py")
            all_passed = False
        
        return all_passed
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\n✅ Successes: {len(self.successes)}")
        for success in self.successes:
            print(f"   • {success}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.issues:
            print(f"\n🔴 Issues: {len(self.issues)}")
            for issue in self.issues:
                print(f"   • {issue}")
        
        print("\n" + "="*70)
        if not self.issues:
            print("✅ VALIDATION PASSED")
            print("="*70)
            print("\nYour configuration is properly set up for GPU-optimized training!")
            print("You should NOT see GPU oscillation issues.")
            print("\nNext steps:")
            print("1. Start training: python train_main.py")
            print("2. Monitor GPU: watch -n 0.5 nvidia-smi")
            print("3. Expect stable 70-95% GPU utilization")
        else:
            print("🔴 VALIDATION FAILED")
            print("="*70)
            print("\nYour configuration has issues that will cause GPU bottleneck!")
            print("\nQuick fix:")
            print("1. Edit configs/config.yaml")
            print("2. Under 'data:' section, add/fix:")
            print("   use_tf_native_loading: true")
            print("   prefetch_to_gpu: true")
            print("   prefetch_buffer_size: 16")
            print("   num_workers: 16")
            print("3. Re-run validation: python utilities/validate_gpu_pipeline.py")
            print("\nOr run with --fix flag to auto-fix config:")
            print("   python utilities/validate_gpu_pipeline.py --fix")
        print("="*70)
    
    def auto_fix_config(self):
        """Attempt to auto-fix config issues."""
        print("\n" + "="*70)
        print("AUTO-FIXING CONFIG")
        print("="*70)
        
        if not self.config_path.exists():
            print("🔴 Cannot fix: Config file not found")
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"🔴 Cannot fix: Failed to parse config: {e}")
            return False
        
        # Ensure data section exists
        if 'data' not in config:
            config['data'] = {}
        
        # Fix critical settings
        fixes_applied = []
        
        if config['data'].get('use_tf_native_loading') != True:
            config['data']['use_tf_native_loading'] = True
            fixes_applied.append("use_tf_native_loading: true")
        
        if config['data'].get('prefetch_to_gpu') != True:
            config['data']['prefetch_to_gpu'] = True
            fixes_applied.append("prefetch_to_gpu: true")
        
        if config['data'].get('prefetch_buffer_size', 0) < 8:
            config['data']['prefetch_buffer_size'] = 16
            fixes_applied.append("prefetch_buffer_size: 16")
        
        if config['data'].get('num_workers', 0) < 8:
            config['data']['num_workers'] = 16
            fixes_applied.append("num_workers: 16")
        
        if not fixes_applied:
            print("✅ No fixes needed - config already optimal")
            return True
        
        # Backup original config
        backup_path = self.config_path.with_suffix('.yaml.backup')
        import shutil
        shutil.copy(self.config_path, backup_path)
        print(f"📁 Backup saved: {backup_path}")
        
        # Write fixed config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Fixed config saved: {self.config_path}")
        print(f"\nFixes applied:")
        for fix in fixes_applied:
            print(f"   • {fix}")
        
        print(f"\n✅ Config fixed successfully!")
        print(f"   Re-run validation to verify: python utilities/validate_gpu_pipeline.py")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GPU pipeline configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix config issues"
    )
    
    args = parser.parse_args()
    
    validator = GPUPipelineValidator(args.config)
    
    if args.fix:
        # Run validation first to identify issues
        validator.validate_all()
        print()
        # Then try to fix
        validator.auto_fix_config()
    else:
        # Just validate
        passed = validator.validate_all()
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
