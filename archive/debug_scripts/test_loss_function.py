#!/usr/bin/env python3
"""
Direct test of mel_loss function to verify clipping fix
"""
import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf

sys.path.insert(0, os.path.abspath('.'))

from myxtts.training.losses import mel_loss

def test_mel_loss_function():
    """Test the mel_loss function directly to see if clipping is fixed."""
    
    print("🧪 Testing Mel Loss Function Directly")
    print("=" * 50)
    
    # ایجاد داده‌های آزمایشی
    batch_size = 2
    time_steps = 100
    n_mels = 80
    
    # شبیه‌سازی mel target واقعی (با محدوده گسترده)
    mel_target = tf.random.normal([batch_size, time_steps, n_mels], mean=-5.0, stddev=2.0)
    
    # شبیه‌سازی mel prediction کنونی (نزدیک صفر)
    mel_pred_bad = tf.random.normal([batch_size, time_steps, n_mels], mean=0.0, stddev=0.1)
    
    # شبیه‌سازی mel prediction خوب
    mel_pred_good = tf.random.normal([batch_size, time_steps, n_mels], mean=-4.8, stddev=1.8)
    
    lengths = tf.constant([time_steps, time_steps-10], dtype=tf.int32)
    
    print(f"📊 Test Data Stats:")
    print(f"   Target: min={tf.reduce_min(mel_target):.3f}, max={tf.reduce_max(mel_target):.3f}, mean={tf.reduce_mean(mel_target):.3f}")
    print(f"   Bad Pred: min={tf.reduce_min(mel_pred_bad):.3f}, max={tf.reduce_max(mel_pred_bad):.3f}, mean={tf.reduce_mean(mel_pred_bad):.3f}")
    print(f"   Good Pred: min={tf.reduce_min(mel_pred_good):.3f}, max={tf.reduce_max(mel_pred_good):.3f}, mean={tf.reduce_mean(mel_pred_good):.3f}")
    
    # محاسبه loss برای هر سناریو
    print(f"\n🔍 Testing Loss Function:")
    
    # تست 1: bad prediction (مشابه مدل فعلی)
    loss_bad = mel_loss(
        y_true=mel_target,
        y_pred=mel_pred_bad,
        lengths=lengths,
        label_smoothing=0.0,
        use_huber_loss=False
    )
    
    print(f"   Bad prediction loss: {loss_bad:.6f}")
    
    # تست 2: good prediction
    loss_good = mel_loss(
        y_true=mel_target,
        y_pred=mel_pred_good,
        lengths=lengths,
        label_smoothing=0.0,
        use_huber_loss=False
    )
    
    print(f"   Good prediction loss: {loss_good:.6f}")
    
    # محاسبه دستی MAE برای مقایسه
    mae_bad = tf.reduce_mean(tf.abs(mel_target - mel_pred_bad))
    mae_good = tf.reduce_mean(tf.abs(mel_target - mel_pred_good))
    
    print(f"   Manual MAE (bad):  {mae_bad:.6f}")
    print(f"   Manual MAE (good): {mae_good:.6f}")
    
    # تست کلیپینگ
    print(f"\n🔧 Testing Loss Clipping Behavior:")
    
    # ایجاد خطای بزرگ
    mel_pred_very_bad = tf.zeros_like(mel_target)  # همه صفر
    loss_very_bad = mel_loss(
        y_true=mel_target,
        y_pred=mel_pred_very_bad,
        lengths=lengths,
        label_smoothing=0.0,
        use_huber_loss=False
    )
    
    mae_very_bad = tf.reduce_mean(tf.abs(mel_target - mel_pred_very_bad))
    
    print(f"   Very bad prediction loss: {loss_very_bad:.6f}")
    print(f"   Manual MAE (very bad): {mae_very_bad:.6f}")
    
    # بررسی کلیپینگ
    print(f"\n📈 Analysis:")
    
    if loss_very_bad < 50.0:
        print(f"   ⚠️  Loss is clipped too aggressively: {loss_very_bad:.3f}")
        print(f"   ⚠️  Original MAE was: {mae_very_bad:.3f}")
        print(f"   ⚠️  This may prevent learning from high-error states")
    else:
        print(f"   ✅ Loss allows high values: {loss_very_bad:.3f}")
        print(f"   ✅ This should allow proper gradient flow")
    
    # تست گرادیان
    print(f"\n🎯 Testing Gradient Flow:")
    
    # متغیر قابل آموزش
    mel_pred_var = tf.Variable(tf.zeros_like(mel_target))
    
    with tf.GradientTape() as tape:
        loss = mel_loss(
            y_true=mel_target,
            y_pred=mel_pred_var,
            lengths=lengths,
            label_smoothing=0.0,
            use_huber_loss=False
        )
    
    gradients = tape.gradient(loss, mel_pred_var)
    
    if gradients is not None:
        grad_norm = tf.norm(gradients)
        print(f"   Gradient norm: {grad_norm:.6f}")
        
        if grad_norm > 0.001:
            print(f"   ✅ Gradients are flowing properly")
        else:
            print(f"   ❌ Gradients are too small - may indicate clipping issues")
    else:
        print(f"   ❌ No gradients computed - major issue!")
    
    # نتیجه‌گیری
    print(f"\n🎯 Final Assessment:")
    
    if loss_very_bad > 10.0:
        print("   ✅ Loss clipping fix appears to be working")
        print("   ✅ High errors are no longer clipped to small values")
        print("   ✅ Model should be able to learn from mistakes")
    else:
        print("   ❌ Loss is still being clipped too aggressively")
        print("   ❌ Model may have difficulty learning mel spectrograms")
        
    return {
        'loss_bad': float(loss_bad),
        'loss_good': float(loss_good),
        'loss_very_bad': float(loss_very_bad),
        'gradient_norm': float(grad_norm) if gradients is not None else 0.0
    }

if __name__ == "__main__":
    test_mel_loss_function()