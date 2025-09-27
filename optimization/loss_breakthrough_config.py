#!/usr/bin/env python3
"""
Loss Breakthrough Configuration - راه‌حل برای عبور از plateau

این کانفیگ برای زمانی طراحی شده که لاس در یک نقطه گیر کرده:
- Learning rate schedule adjustment
- Loss components rebalancing  
- Advanced optimization techniques
"""

def apply_loss_breakthrough_optimizations(config):
    """
    تنظیمات پیشرفته برای عبور از loss plateau
    """
    
    # 1. LEARNING RATE BREAKTHROUGH
    # کاهش learning rate برای convergence بهتر
    if hasattr(config.training, 'learning_rate') and config.training.learning_rate > 2e-5:
        print("🔧 Reducing learning rate for plateau breakthrough...")
        config.training.learning_rate = 1.5e-5  # کاهش از 8e-5 به 1.5e-5
        
    # 2. SCHEDULER ADJUSTMENT FOR PLATEAU
    print("📈 Applying plateau-aware scheduler...")
    config.training.scheduler = "cosine"
    config.training.cosine_restarts = True
    config.training.scheduler_params = {
        "min_learning_rate": 5e-7,  # پایین‌تر برای fine-tuning
        "restart_period": 100,      # restart هر 100 epoch
        "restart_mult": 1.2         # افزایش تدریجی دوره
    }
    
    # 3. LOSS COMPONENTS REBALANCING
    # متعادل کردن وزن‌های loss برای بهبود
    print("⚖️ Rebalancing loss components...")
    config.training.mel_loss_weight = 2.0      # کاهش از 2.5
    config.training.kl_loss_weight = 1.2       # کاهش از 1.8
    config.training.stop_loss_weight = 0.8     # افزایش اهمیت stop token
    
    # 4. GRADIENT OPTIMIZATION
    print("🎯 Optimizing gradient flow...")
    config.training.gradient_clip_norm = 0.3   # سخت‌تر برای stability
    config.training.weight_decay = 2e-7        # کاهش regularization
    
    # 5. ADAPTIVE LEARNING STRATEGIES
    print("🤖 Enabling adaptive learning...")
    config.training.adaptive_loss_weights = True
    config.training.label_smoothing = 0.1
    config.training.huber_loss = True
    
    # 6. ADVANCED OPTIMIZER SETTINGS
    print("⚙️ Advanced optimizer configuration...")
    config.training.optimizer_params = {
        "betas": (0.9, 0.98),      # بهتر برای convergence
        "eps": 1e-9,               # numerical stability
        "amsgrad": True            # adaptive learning rate
    }
    
    # 7. PLATEAU DETECTION AND AUTO-ADJUSTMENT
    config.training.plateau_detection = {
        "patience": 10,            # صبر 10 epoch
        "min_delta": 0.01,         # حداقل تغییر معنادار
        "factor": 0.5,             # کاهش LR به نصف
        "cooldown": 5              # استراحت بعد تغییر
    }
    
    # 8. BATCH SIZE OPTIMIZATION FOR CONVERGENCE
    # بررسی اینکه آیا batch size مناسب است
    current_batch = getattr(config.training, 'batch_size', 32)
    if current_batch > 32:
        print(f"📊 Large batch size ({current_batch}) detected - this might cause plateau")
        print("   Consider using smaller batches for better gradient flow")
    
    print("\n✅ Loss breakthrough optimizations applied!")
    print("Expected improvements:")
    print("   • Better convergence below 2.5 loss")
    print("   • More stable training dynamics")
    print("   • Adaptive learning rate adjustment")
    print("   • Balanced loss components")
    
    return config


def create_plateau_breaking_config():
    """
    ایجاد کانفیگ خاص برای شکستن plateau
    """
    breakthrough_config = {
        # CORE TRAINING PARAMETERS
        "learning_rate": 1.5e-5,           # کاهش یافته
        "batch_size": 24,                  # کوچک‌تر برای gradient بهتر
        "gradient_accumulation": 2,
        
        # SCHEDULER FOR PLATEAU BREAKING
        "scheduler": "cosine",
        "cosine_restarts": True,
        "restart_period": 100,
        "min_learning_rate": 5e-7,
        
        # REBALANCED LOSS WEIGHTS
        "mel_loss_weight": 2.0,           # کاهش یافته
        "kl_loss_weight": 1.2,            # متعادل
        "stop_loss_weight": 0.8,          # تقویت شده
        
        # GRADIENT OPTIMIZATION
        "gradient_clip_norm": 0.3,        # محدودتر
        "weight_decay": 2e-7,             # کمتر
        
        # ADVANCED FEATURES
        "adaptive_loss_weights": True,
        "label_smoothing": 0.1,
        "huber_loss": True,
        
        # PLATEAU HANDLING
        "early_stopping_patience": 15,
        "reduce_lr_patience": 8,
        "reduce_lr_factor": 0.6
    }
    
    return breakthrough_config


if __name__ == "__main__":
    print("🚀 Loss Breakthrough Configuration")
    print("=" * 50)
    
    config = create_plateau_breaking_config()
    
    print("\n📋 Recommended settings for breaking loss plateau:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n💡 Usage:")
    print("   python3 train_main.py --optimization-level experimental --apply-loss-breakthrough")