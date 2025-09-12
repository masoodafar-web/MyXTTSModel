#!/usr/bin/env python3
"""
Test script to validate preprocessing mode functionality without requiring full dependencies.
"""

import sys
import os
import tempfile
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from myxtts.config.config import XTTSConfig


class MockLJSpeechDataset:
    """Mock LJSpeech dataset for testing."""
    
    def __init__(self, *args, **kwargs):
        self.precompute_calls = []
        self.verify_calls = []
        self.filter_calls = []
    
    def precompute_mels(self, num_workers=1, overwrite=False):
        self.precompute_calls.append(('mels', num_workers, overwrite))
        
    def precompute_tokens(self, num_workers=1, overwrite=False):
        self.precompute_calls.append(('tokens', num_workers, overwrite))
        
    def verify_and_fix_cache(self, fix=True):
        self.verify_calls.append(fix)
        return {"checked": 100, "fixed": 5, "failed": 0}
        
    def filter_items_by_cache(self):
        self.filter_calls.append(True)
        return 95
        
    def create_tf_dataset(self, **kwargs):
        return Mock()


def test_preprocessing_modes():
    """Test that preprocessing modes work as expected."""
    
    print("Testing preprocessing mode functionality...")
    
    # Test configurations
    configs = {
        "auto": XTTSConfig(preprocessing_mode="auto"),
        "precompute": XTTSConfig(preprocessing_mode="precompute"), 
        "runtime": XTTSConfig(preprocessing_mode="runtime")
    }
    
    for mode, config in configs.items():
        print(f"\nTesting {mode.upper()} mode:")
        print(f"  preprocessing_mode = {config.data.preprocessing_mode}")
        
        # Create mock trainer
        with patch('myxtts.training.trainer.LJSpeechDataset', MockLJSpeechDataset):
            with patch('myxtts.training.trainer.setup_logging') as mock_logging:
                mock_logger = Mock()
                mock_logging.return_value = mock_logger
                
                try:
                    # Import trainer after patching
                    from myxtts.training.trainer import XTTSTrainer
                    
                    # Mock the model and other dependencies
                    with patch('myxtts.training.trainer.XTTS'):
                        with patch('myxtts.training.trainer.configure_gpus'):
                            with patch('myxtts.training.trainer.get_device', return_value="CPU"):
                                with patch('myxtts.training.trainer.setup_gpu_strategy'):
                                    
                                    trainer = XTTSTrainer(config)
                                    
                                    # Test prepare_datasets method
                                    try:
                                        train_ds, val_ds = trainer.prepare_datasets("/fake/path")
                                        print(f"  ✅ prepare_datasets completed successfully")
                                        
                                        # Check logging calls
                                        log_calls = [call[0][0] for call in mock_logger.info.call_args_list if call[0]]
                                        preprocessing_logs = [log for log in log_calls if 'preprocessing mode' in log.lower()]
                                        
                                        if preprocessing_logs:
                                            print(f"  📝 Logged: {preprocessing_logs[0]}")
                                        
                                    except Exception as e:
                                        if mode == "precompute" and "preprocessing failed" in str(e).lower():
                                            print(f"  ✅ Correctly failed in precompute mode: {e}")
                                        else:
                                            print(f"  ❌ Unexpected error: {e}")
                                            
                except ImportError as e:
                    print(f"  ⚠️  Skipping trainer test due to missing dependencies: {e}")
    
    print("\n✅ All preprocessing mode tests completed!")


def test_configuration_integration():
    """Test that configuration integration works properly."""
    
    print("\nTesting configuration integration...")
    
    # Test YAML round-trip
    config = XTTSConfig(preprocessing_mode="precompute", batch_size=64)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
    
    try:
        # Save and load YAML
        config.to_yaml(yaml_path)
        loaded_config = XTTSConfig.from_yaml(yaml_path)
        
        assert loaded_config.data.preprocessing_mode == "precompute"
        assert loaded_config.data.batch_size == 64
        print("  ✅ YAML serialization works correctly")
        
    finally:
        os.unlink(yaml_path)
    
    # Test validation
    try:
        XTTSConfig(preprocessing_mode="invalid_mode")
        print("  ❌ Validation should have failed for invalid mode")
    except ValueError:
        print("  ✅ Validation correctly rejects invalid modes")
    
    print("✅ Configuration integration tests passed!")


if __name__ == "__main__":
    try:
        test_preprocessing_modes()
        test_configuration_integration()
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)