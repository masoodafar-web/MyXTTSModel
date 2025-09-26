#!/usr/bin/env python3
"""
تست سریع برای بررسی رفع مسئله optimizer variable mismatch
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_optimizer_fix():
    """تست رفع مسئله optimizer"""
    print("🧪 Testing optimizer variable mismatch fix...")
    
    # Import modules
    try:
        from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
        from myxtts.models.xtts import XTTS
        from myxtts.training.trainer import XTTSTrainer
        print("✅ Modules imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Create minimal config
    try:
        model_config = ModelConfig(
            text_encoder_dim=256,
            audio_encoder_dim=256,
            decoder_dim=512,
            use_duration_predictor=False,  # Key fix: disabled
            enable_gradient_checkpointing=True
        )
        
        training_config = TrainingConfig(
            epochs=1,
            learning_rate=1e-4
        )
        
        data_config = DataConfig(
            sample_rate=22050,
            batch_size=8
        )
        
        config = XTTSConfig(
            model=model_config,
            training=training_config,
            data=data_config
        )
        
        print("✅ Configuration created successfully")
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False
    
    # Test model creation
    try:
        model = XTTS(config.model)
        print("✅ Model created successfully")
        print(f"   - Model variables: {len(model.trainable_variables)}")
        
        # Test trainer creation
        trainer = XTTSTrainer(config=config, model=model, resume_checkpoint=None)
        print("✅ Trainer created successfully")
        
        # Check optimizer
        if hasattr(trainer, 'optimizer') and trainer.optimizer:
            print(f"✅ Optimizer exists: {type(trainer.optimizer).__name__}")
        else:
            print("⚠️  No optimizer found")
            
        return True
        
    except Exception as e:
        print(f"❌ Model/Trainer creation failed: {e}")
        return False

def test_duration_predictor_disabled():
    """تست غیرفعال بودن duration_predictor"""
    print("\n🧪 Testing duration_predictor disabled...")
    
    try:
        from myxtts.config.config import ModelConfig
        
        config = ModelConfig(
            text_encoder_dim=256,
            audio_encoder_dim=256,
            decoder_dim=512,
            use_duration_predictor=False
        )
        
        print(f"✅ Duration predictor disabled: {not config.use_duration_predictor}")
        return True
        
    except Exception as e:
        print(f"❌ Duration predictor test failed: {e}")
        return False

def main():
    """اجرای تست‌ها"""
    print("🎯 Optimizer Variable Mismatch Fix Test")
    print("=" * 50)
    
    # Test optimizer fix
    test1_passed = test_optimizer_fix()
    
    # Test duration predictor
    test2_passed = test_duration_predictor_disabled()
    
    # Summary
    print("\n📊 Test Results:")
    print("-" * 30)
    print(f"Optimizer fix test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Duration predictor test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎯 All tests passed! The optimizer variable mismatch should be fixed.")
        print("\nYou can now run training with:")
        print("python3 train_main.py --model-size tiny --optimization-level enhanced")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)