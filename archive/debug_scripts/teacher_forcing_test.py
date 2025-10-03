#!/usr/bin/env python3
"""
Teacher Forcing Test - تشخیص مشکل اصلی مدل
"""
import numpy as np
import soundfile as sf
import tensorflow as tf
from train_main import build_config
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import find_latest_checkpoint

print("🔍 Teacher Forcing Diagnostic Test")
print("=" * 60)

# تنظیمات
ckpt = find_latest_checkpoint('./checkpointsmain')
config = build_config(model_size='normal', checkpoint_dir='./checkpointsmain')

# فقط CPU برای جلوگیری از OOM
tf.config.set_visible_devices([], 'GPU')

print(f"📦 Loading checkpoint: {ckpt}")
inf = XTTSInference(config=config, checkpoint_path=ckpt)

# متن نمونه از داده آموزشی
text = "The Chronicles of Newgate, Volume two. By Arthur Griffiths. Section thirteen: Newgate notorieties, part one."
processed = inf._preprocess_text(text, config.data.language)
seq = inf.text_processor.text_to_sequence(processed)
text_tensor = tf.constant([seq], dtype=tf.int32)
lengths = tf.constant([len(seq)], dtype=tf.int32)

print(f"📝 Text sequence length: {len(seq)}")

# بارگذاری فایل صوتی آموزشی
audio_path = '../dataset/dataset_train/wavs/LJ010-0001.wav'
try:
    audio, sr = sf.read(audio_path)
    print(f"🎵 Audio loaded: {audio_path} (sr={sr}, len={len(audio)})")
except FileNotFoundError:
    # اگر فایل در آن مسیر نیست، از speaker.wav استفاده کنیم
    audio_path = 'speaker.wav'
    audio, sr = sf.read(audio_path)
    print(f"🎵 Using alternative audio: {audio_path} (sr={sr}, len={len(audio)})")

if audio.ndim > 1:
    audio = audio.mean(axis=1)

# پردازش صوت و تولید mel
audio_processed = inf.audio_processor.preprocess_audio(audio)
mel_target = inf.audio_processor.wav_to_mel(audio_processed).T  # [time, mel_bins]
mel_tensor = tf.constant(mel_target[np.newaxis, ...], dtype=tf.float32)
mel_lengths = tf.constant([mel_target.shape[0]], dtype=tf.int32)

print(f"🎯 Target mel shape: {mel_target.shape}")

# اجرای مدل با Teacher Forcing
print("\n🔥 Running Teacher Forcing...")
outputs = inf.model(
    text_inputs=text_tensor, 
    mel_inputs=mel_tensor,
    text_lengths=lengths, 
    mel_lengths=mel_lengths, 
    training=True
)

mel_pred = outputs['mel_output'].numpy()[0]

print("\n📊 CRITICAL ANALYSIS:")
print("-" * 40)
print(f"🎯 Target mel stats:")
print(f"   Shape: {mel_target.shape}")
print(f"   Min: {mel_target.min():.3f}")
print(f"   Max: {mel_target.max():.3f}")
print(f"   Mean: {mel_target.mean():.3f}")
print(f"   Std: {mel_target.std():.3f}")

print(f"\n🤖 Model prediction stats:")
print(f"   Shape: {mel_pred.shape}")
print(f"   Min: {mel_pred.min():.3f}")
print(f"   Max: {mel_pred.max():.3f}")
print(f"   Mean: {mel_pred.mean():.3f}")
print(f"   Std: {mel_pred.std():.3f}")

# محاسبه خطاها
mae = np.mean(np.abs(mel_target - mel_pred))
rmse = np.sqrt(np.mean((mel_target - mel_pred)**2))

print(f"\n❌ Error metrics:")
print(f"   MAE: {mae:.3f}")
print(f"   RMSE: {rmse:.3f}")

# تحلیل توزیع
print(f"\n📈 Distribution analysis:")
target_range = mel_target.max() - mel_target.min()
pred_range = mel_pred.max() - mel_pred.min()
print(f"   Target range: {target_range:.3f}")
print(f"   Prediction range: {pred_range:.3f}")
print(f"   Range ratio: {pred_range/target_range:.4f}")

# نمونه فریم‌ها
print(f"\n🔬 Sample frames comparison:")
for i in [0, mel_target.shape[0]//4, mel_target.shape[0]//2, -1]:
    if i == -1:
        i = mel_target.shape[0] - 1
    print(f"   Frame {i:3d}: target={mel_target[i, :5].mean():.3f}, pred={mel_pred[i, :5].mean():.3f}")

# تشخیص نهایی
print(f"\n🚨 DIAGNOSIS:")
if pred_range < 1.0:
    print("   ❌ Model output is essentially flat (near zero)")
    print("   ❌ This explains why vocoder produces only noise")
    print("   ❌ Problem is in model training, not in vocoder")
else:
    print("   ✅ Model output has reasonable dynamic range")

if mae > 3.0:
    print("   ❌ Very high prediction error")
    print("   ❌ Model is not learning proper mel spectrogram mapping")
else:
    print("   ✅ Prediction error is reasonable")

print(f"\n💡 NEXT STEPS:")
print("   1. Check mel_loss clipping in losses.py")
print("   2. Review normalization consistency between training/inference")
print("   3. Add teacher forcing monitoring to training")
print("   4. Consider retraining with corrected loss function")