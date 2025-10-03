#!/usr/bin/env python3
"""
Manual training loop test - debugging mel learning
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
from myxtts.training.losses import mel_loss

def manual_training_test():
    """Test manual training loop to debug mel learning."""
    
    print("🧪 Manual Training Loop Test")
    print("=" * 50)
    
    # تنظیم GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # بارگذاری مدل
    config = build_config(model_size='normal', checkpoint_dir='./checkpointsmain')
    latest_checkpoint = find_latest_checkpoint('./checkpointsmain')
    
    print(f"📦 Loading: {latest_checkpoint}")
    
    # بارگذاری مدل برای inference ابتدایی
    inf = XTTSInference(config=config, checkpoint_path=latest_checkpoint)
    
    # فعال کردن training mode
    inf.enable_training()
    
    # آماده‌سازی داده تست
    text = "Hello world."
    processed = inf._preprocess_text(text, config.data.language)
    seq = inf.text_processor.text_to_sequence(processed)
    text_tensor = tf.constant([seq], dtype=tf.int32)
    text_lengths = tf.constant([len(seq)], dtype=tf.int32)
    
    # فایل صوتی
    audio, sr = sf.read('speaker.wav')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    audio_processed = inf.audio_processor.preprocess_audio(audio)
    mel_target = inf.audio_processor.wav_to_mel(audio_processed).T
    mel_tensor = tf.constant(mel_target[np.newaxis, ...], dtype=tf.float32)
    mel_lengths = tf.constant([mel_target.shape[0]], dtype=tf.int32)
    
    print(f"📝 Data ready: text_len={len(seq)}, mel_shape={mel_target.shape}")
    
    # تست اولیه
    print(f"\n🔍 Initial Test:")
    outputs = inf.model(
        text_inputs=text_tensor,
        mel_inputs=mel_tensor,
        text_lengths=text_lengths,
        mel_lengths=mel_lengths,
        training=True
    )
    
    mel_pred_initial = outputs['mel_output'].numpy()[0]
    mae_initial = np.mean(np.abs(mel_target - mel_pred_initial))
    
    print(f"   Target: min={mel_target.min():.3f}, max={mel_target.max():.3f}, mean={mel_target.mean():.3f}")
    print(f"   Initial: min={mel_pred_initial.min():.3f}, max={mel_pred_initial.max():.3f}, mean={mel_pred_initial.mean():.3f}")
    print(f"   Initial MAE: {mae_initial:.3f}")
    
    # ایجاد optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    print(f"\n🏋️ Manual Training Steps:")
    
    # training loop دستی
    for step in range(5):
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            outputs = inf.model(
                text_inputs=text_tensor,
                mel_inputs=mel_tensor,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=True
            )
            
            mel_pred = outputs['mel_output']
            
            # محاسبه loss
            loss = mel_loss(
                y_true=mel_tensor,
                y_pred=mel_pred,
                lengths=mel_lengths,
                label_smoothing=0.0,
                use_huber_loss=False
            )
        
        # Gradient computation
        gradients = tape.gradient(loss, inf.model.trainable_variables)
        
        # چک کردن گرادیان‌ها
        grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients]
        total_grad_norm = np.sqrt(sum([gn**2 for gn in grad_norms]))
        
        # Apply gradients (فیلتر کردن None gradients)
        grad_var_pairs = [(g, v) for g, v in zip(gradients, inf.model.trainable_variables) if g is not None]
        if grad_var_pairs:
            optimizer.apply_gradients(grad_var_pairs)
        else:
            print(f"             ❌ No gradients computed!")
        
        # محاسبه MAE برای نمایش پیشرفت
        mel_pred_np = mel_pred.numpy()[0]
        mae_current = np.mean(np.abs(mel_target - mel_pred_np))
        
        print(f"   Step {step+1}: loss={loss:.6f}, mae={mae_current:.3f}, grad_norm={total_grad_norm:.6f}")
        
        # بررسی تغییرات در range
        pred_range = mel_pred_np.max() - mel_pred_np.min()
        target_range = mel_target.max() - mel_target.min()
        
        if step == 0:
            initial_range = pred_range
            
        print(f"             pred_range={pred_range:.3f} (ratio={pred_range/target_range:.3f})")
        
        # چک کردن پیشرفت
        if pred_range > initial_range * 1.1:
            print(f"             ✅ Range is expanding!")
        
        del tape  # cleanup
    
    # تست نهایی
    print(f"\n🔍 Final Test:")
    outputs_final = inf.model(
        text_inputs=text_tensor,
        mel_inputs=mel_tensor,
        text_lengths=text_lengths,
        mel_lengths=mel_lengths,
        training=False
    )
    
    mel_pred_final = outputs_final['mel_output'].numpy()[0]
    mae_final = np.mean(np.abs(mel_target - mel_pred_final))
    
    print(f"   Final: min={mel_pred_final.min():.3f}, max={mel_pred_final.max():.3f}, mean={mel_pred_final.mean():.3f}")
    print(f"   Final MAE: {mae_final:.3f}")
    
    # تحلیل نهایی
    improvement = mae_initial - mae_final
    range_initial = mel_pred_initial.max() - mel_pred_initial.min()
    range_final = mel_pred_final.max() - mel_pred_final.min()
    
    print(f"\n📊 Summary:")
    print(f"   MAE improvement: {improvement:+.3f}")
    print(f"   Range change: {range_initial:.3f} → {range_final:.3f}")
    
    if improvement > 0.01 and range_final > range_initial * 1.1:
        print(f"   🎉 SUCCESS: Model is learning to generate proper mel spectrograms!")
    elif range_final > range_initial * 1.1:
        print(f"   ✅ PROGRESS: Range is expanding, continue training")
    else:
        print(f"   ❌ ISSUE: Model is not expanding its output range")

if __name__ == "__main__":
    manual_training_test()