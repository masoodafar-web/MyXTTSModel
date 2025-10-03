#!/usr/bin/env python3
"""
Simple test to check if mel_loss fix is working
"""
import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf

sys.path.insert(0, os.path.abspath('.'))

from train_main import build_config
from myxtts.training.trainer import XTTSTrainer
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import find_latest_checkpoint

def test_mel_generation():
    """Test if model can generate proper mel spectrograms after loss fix."""
    
    print("🧪 Testing Mel Generation After Loss Fix")
    print("=" * 50)
    
    # بارگذاری مدل
    config = build_config(
        model_size='normal',
        checkpoint_dir='./checkpointsmain',
        batch_size=1,  # خیلی کوچک برای تست
        lr=0.001      # نرخ یادگیری بالاتر برای تغییرات سریع
    )
    
    # تنظیم GPU برای همه دستگاه‌ها
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # بارگذاری checkpoint
    latest_checkpoint = find_latest_checkpoint('./checkpointsmain')
    print(f"📦 Loading: {latest_checkpoint}")
    
    # ایجاد inference برای تست اولیه
    inf = XTTSInference(config=config, checkpoint_path=latest_checkpoint)
    
    # آماده‌سازی داده تست
    text = "Hello world test."
    processed = inf._preprocess_text(text, config.data.language)
    seq = inf.text_processor.text_to_sequence(processed)
    text_tensor = tf.constant([seq], dtype=tf.int32)
    text_lengths = tf.constant([len(seq)], dtype=tf.int32)
    
    # فایل صوتی تست
    audio, sr = sf.read('speaker.wav')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    audio_processed = inf.audio_processor.preprocess_audio(audio)
    mel_target = inf.audio_processor.wav_to_mel(audio_processed).T
    mel_tensor = tf.constant(mel_target[np.newaxis, ...], dtype=tf.float32)
    mel_lengths = tf.constant([mel_target.shape[0]], dtype=tf.int32)
    
    print(f"📝 Text length: {len(seq)}")
    print(f"🎵 Mel target shape: {mel_target.shape}")
    
    # تست اولیه
    print("\n🔍 BEFORE Training Step:")
    outputs = inf.model(
        text_inputs=text_tensor,
        mel_inputs=mel_tensor,
        text_lengths=text_lengths,
        mel_lengths=mel_lengths,
        training=True
    )
    
    mel_pred_before = outputs['mel_output'].numpy()[0]
    
    print(f"   Target: min={mel_target.min():.3f}, max={mel_target.max():.3f}, mean={mel_target.mean():.3f}, std={mel_target.std():.3f}")
    print(f"   Pred:   min={mel_pred_before.min():.3f}, max={mel_pred_before.max():.3f}, mean={mel_pred_before.mean():.3f}, std={mel_pred_before.std():.3f}")
    
    mae_before = np.mean(np.abs(mel_target - mel_pred_before))
    print(f"   MAE: {mae_before:.3f}")
    
    # ایجاد trainer برای آموزش
    trainer = XTTSTrainer(config=config)
    trainer.load_checkpoint(latest_checkpoint)
    
    print(f"\n🏋️ Performing a few training steps...")
    
    # چند قدم آموزش
    for step in range(3):
        try:
            losses = trainer.train_step()
            mel_loss = losses.get('mel_loss', 0.0)
            total_loss = losses.get('total_loss', 0.0)
            print(f"   Step {step+1}: mel_loss={mel_loss:.4f}, total_loss={total_loss:.4f}")
            
            # چک کردن mel_loss value
            if mel_loss > 50.0:
                print(f"   ⚠️  Very high mel_loss: {mel_loss:.3f} - loss clipping may still be an issue")
            elif mel_loss > 10.0:
                print(f"   ✅ High mel_loss: {mel_loss:.3f} - model is trying to learn (good!)")
            elif mel_loss < 1.0:
                print(f"   ❌ Low mel_loss: {mel_loss:.3f} - model might not be learning properly")
                
        except Exception as e:
            print(f"   ❌ Training step failed: {e}")
            break
    
    # تست بعد از آموزش
    print(f"\n🔍 AFTER Training Steps:")
    outputs_after = trainer.model(
        text_inputs=text_tensor,
        mel_inputs=mel_tensor,
        text_lengths=text_lengths,
        mel_lengths=mel_lengths,
        training=True
    )
    
    mel_pred_after = outputs_after['mel_output'].numpy()[0]
    
    print(f"   Target: min={mel_target.min():.3f}, max={mel_target.max():.3f}, mean={mel_target.mean():.3f}, std={mel_target.std():.3f}")
    print(f"   Pred:   min={mel_pred_after.min():.3f}, max={mel_pred_after.max():.3f}, mean={mel_pred_after.mean():.3f}, std={mel_pred_after.std():.3f}")
    
    mae_after = np.mean(np.abs(mel_target - mel_pred_after))
    print(f"   MAE: {mae_after:.3f}")
    
    # تحلیل تغییرات
    print(f"\n📊 Analysis:")
    improvement = mae_before - mae_after
    pred_range_before = mel_pred_before.max() - mel_pred_before.min()
    pred_range_after = mel_pred_after.max() - mel_pred_after.min()
    target_range = mel_target.max() - mel_target.min()
    
    print(f"   MAE change: {improvement:+.3f} ({'✅ Better' if improvement > 0 else '❌ Worse' if improvement < -0.01 else '➖ No change'})")
    print(f"   Pred range before: {pred_range_before:.3f}")
    print(f"   Pred range after:  {pred_range_after:.3f}")
    print(f"   Target range:      {target_range:.3f}")
    print(f"   Range ratio after: {pred_range_after/target_range:.3f}")
    
    # نتیجه‌گیری
    print(f"\n🎯 Conclusion:")
    if pred_range_after > pred_range_before * 1.1:
        print("   ✅ Model is starting to expand its output range - GOOD!")
    else:
        print("   ❌ Model output range is still restricted")
        
    if improvement > 0.01:
        print("   ✅ MAE is improving - model is learning")
    else:
        print("   ❌ MAE is not improving significantly")
        
    if pred_range_after > 1.0:
        print("   ✅ Output has reasonable dynamic range")
        print("   🎉 Loss fix appears to be working!")
    else:
        print("   ❌ Output range is still too small")
        print("   ⚠️  May need more training steps or different approach")

if __name__ == "__main__":
    test_mel_generation()