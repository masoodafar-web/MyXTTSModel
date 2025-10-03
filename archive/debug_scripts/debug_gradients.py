#!/usr/bin/env python3
"""
Debug gradient flow in XTTS model
"""
import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf

sys.path.insert(0, os.path.abspath('.'))

from train_main import build_config
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import find_latest_checkpoint

def debug_gradient_flow():
    """Debug why gradients are not flowing."""
    
    print("🔍 Debugging Gradient Flow")
    print("=" * 50)
    
    # تنظیم GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # بارگذاری مدل
    config = build_config(model_size='normal', checkpoint_dir='./checkpointsmain')
    latest_checkpoint = find_latest_checkpoint('./checkpointsmain')
    
    inf = XTTSInference(config=config, checkpoint_path=latest_checkpoint)
    
    # فعال کردن training mode
    inf.enable_training()
    
    # آماده‌سازی داده‌های کوچک
    text_tensor = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)  # simple sequence
    text_lengths = tf.constant([5], dtype=tf.int32)
    
    mel_target = tf.random.normal([1, 50, 80], mean=-5.0, stddev=2.0)  # کوچک‌تر
    mel_lengths = tf.constant([50], dtype=tf.int32)
    
    print(f"📝 Simple test data ready")
    
    # چک کردن trainable variables
    trainable_vars = inf.model.trainable_variables
    print(f"📊 Model has {len(trainable_vars)} trainable variables")
    
    if len(trainable_vars) == 0:
        print("❌ PROBLEM: No trainable variables found!")
        return
    
    # نمایش چند متغیر اول
    for i, var in enumerate(trainable_vars[:5]):
        print(f"   Var {i}: {var.name} shape={var.shape}")
    
    # تست gradient tape
    print(f"\n🧪 Testing gradient computation:")
    
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        outputs = inf.model(
            text_inputs=text_tensor,
            mel_inputs=mel_target,  # استفاده از mel_target به عنوان input
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True  # حتماً True
        )
        
        mel_pred = outputs['mel_output']
        
        # simple loss
        loss = tf.reduce_mean(tf.square(mel_target - mel_pred))
        
        print(f"   Forward pass successful")
        print(f"   Loss: {loss:.6f}")
        print(f"   Mel pred shape: {mel_pred.shape}")
    
    # محاسبه گرادیان‌ها
    print(f"\n📈 Computing gradients...")
    
    gradients = tape.gradient(loss, trainable_vars)
    
    # بررسی گرادیان‌ها
    none_grads = sum([1 for g in gradients if g is None])
    non_none_grads = len(gradients) - none_grads
    
    print(f"   Total variables: {len(gradients)}")
    print(f"   None gradients: {none_grads}")
    print(f"   Valid gradients: {non_none_grads}")
    
    if non_none_grads == 0:
        print(f"   ❌ CRITICAL: All gradients are None!")
        
        # تست ساده‌تر
        print(f"\n🔬 Testing with simple computation:")
        
        with tf.GradientTape() as simple_tape:
            simple_output = tf.reduce_sum(trainable_vars[0])  # خیلی ساده
            
        simple_grad = simple_tape.gradient(simple_output, trainable_vars[0])
        
        if simple_grad is not None:
            print(f"   ✅ Simple gradient works: {tf.norm(simple_grad):.6f}")
            print(f"   ❌ Problem is in model forward pass or loss computation")
        else:
            print(f"   ❌ Even simple gradient fails - major TF issue")
            
    else:
        print(f"   ✅ Some gradients computed successfully!")
        
        # نمایش نورم گرادیان‌ها
        for i, (var, grad) in enumerate(zip(trainable_vars[:10], gradients[:10])):
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                print(f"   Var {i} ({var.name}): grad_norm={grad_norm:.6f}")
    
    # تست مسیر forward pass
    print(f"\n🔍 Testing model components:")
    
    try:
        # تست encoder
        if hasattr(inf.model, 'text_encoder'):
            print(f"   Text encoder: OK")
        if hasattr(inf.model, 'audio_encoder'):  
            print(f"   Audio encoder: OK")
        if hasattr(inf.model, 'decoder'):
            print(f"   Decoder: OK")
            
        # تست intermediate outputs
        with tf.GradientTape() as component_tape:
            # فقط یک قسمت از مدل
            if hasattr(inf.model, 'decoder'):
                # تست مستقیم decoder
                fake_encoder_output = tf.random.normal([1, 50, 512])
                fake_attention = tf.random.normal([1, 50, 512])
                
                decoder_out = inf.model.decoder(
                    fake_encoder_output,
                    attention_context=fake_attention,
                    training=True
                )
                
                simple_decoder_loss = tf.reduce_mean(tf.square(decoder_out))
        
        decoder_grads = component_tape.gradient(simple_decoder_loss, inf.model.decoder.trainable_variables)
        decoder_valid_grads = sum([1 for g in decoder_grads if g is not None])
        
        print(f"   Decoder gradients: {decoder_valid_grads}/{len(decoder_grads)} valid")
        
    except Exception as e:
        print(f"   ❌ Component test failed: {e}")
    
    del tape  # cleanup

if __name__ == "__main__":
    debug_gradient_flow()