#!/usr/bin/env python3
"""
تست ساده کیفیت صدا بدون استفاده از GPU
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def simple_audio_quality_test(audio_file):
    """تست ساده کیفیت صدا"""
    
    print(f"🎵 تحلیل فایل: {audio_file}")
    
    # بارگذاری صدا
    try:
        y, sr = librosa.load(audio_file, sr=22050)
        print(f"✅ فایل بارگذاری شد: {len(y)} samples, {sr} Hz")
    except Exception as e:
        print(f"❌ خطا در بارگذاری: {e}")
        return
    
    # محاسبات بنیادی
    duration = len(y) / sr
    rms_energy = np.sqrt(np.mean(y**2))
    max_amplitude = np.max(np.abs(y))
    
    print(f"📊 مشخصات اولیه:")
    print(f"   • مدت زمان: {duration:.2f} ثانیه")
    print(f"   • RMS Energy: {rms_energy:.4f}")
    print(f"   • Max Amplitude: {max_amplitude:.4f}")
    
    # تشخیص نویز با روش‌های ساده
    
    # 1. بررسی سکوت زیاد (ممکن است نشانه مشکل باشد)
    silence_threshold = 0.01
    silence_ratio = np.sum(np.abs(y) < silence_threshold) / len(y)
    print(f"   • درصد سکوت: {silence_ratio*100:.1f}%")
    
    # 2. بررسی کلیپینگ (distortion)
    clipping_threshold = 0.99
    clipping_ratio = np.sum(np.abs(y) > clipping_threshold) / len(y)
    print(f"   • درصد کلیپینگ: {clipping_ratio*100:.3f}%")
    
    # 3. تحلیل فرکانسی ساده
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    magnitude = np.abs(fft)
    
    # فرکانس‌های اصلی صدای انسان: 80-8000 Hz
    human_voice_mask = (freqs >= 80) & (freqs <= 8000)
    voice_energy = np.sum(magnitude[human_voice_mask])
    total_energy = np.sum(magnitude)
    voice_ratio = voice_energy / total_energy if total_energy > 0 else 0
    
    print(f"   • انرژی در فرکانس صدای انسان: {voice_ratio*100:.1f}%")
    
    # 4. تشخیص نویز high-frequency
    high_freq_mask = freqs > 8000
    high_freq_energy = np.sum(magnitude[high_freq_mask])
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    print(f"   • انرژی در فرکانس‌های بالا (نویز): {high_freq_ratio*100:.1f}%")
    
    # نتیجه‌گیری
    print(f"\n🔍 تحلیل کیفیت:")
    
    quality_score = 0
    issues = []
    
    if silence_ratio > 0.7:
        issues.append("سکوت زیاد - احتمال مشکل در تولید صدا")
    else:
        quality_score += 1
        
    if clipping_ratio > 0.01:
        issues.append("کلیپینگ زیاد - اعوجاج در صدا")
    else:
        quality_score += 1
        
    if voice_ratio < 0.3:
        issues.append("انرژی کم در فرکانس صدای انسان")
    else:
        quality_score += 1
        
    if high_freq_ratio > 0.3:
        issues.append("نویز زیاد در فرکانس‌های بالا")
    else:
        quality_score += 1
    
    if rms_energy < 0.001:
        issues.append("انرژی کل خیلی کم")
    elif rms_energy > 0.5:
        issues.append("انرژی کل خیلی زیاد")
    else:
        quality_score += 1
    
    print(f"   • امتیاز کیفیت: {quality_score}/5")
    
    if issues:
        print(f"   • مشکلات شناسایی شده:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"   • ✅ هیچ مشکل جدی شناسایی نشد")
    
    # پیشنهاد نهایی
    if quality_score >= 4:
        print(f"\n🎉 نتیجه: کیفیت صدا خوب است!")
    elif quality_score >= 2:
        print(f"\n⚠️ نتیجه: کیفیت صدا متوسط - نیاز به بهبود")
    else:
        print(f"\n❌ نتیجه: کیفیت صدا ضعیف - مشکلات جدی وجود دارد")
    
    return quality_score, issues

if __name__ == "__main__":
    audio_file = "outputs/audio_samples/test_after_epoch1.wav"
    
    if os.path.exists(audio_file):
        simple_audio_quality_test(audio_file)
    else:
        print(f"❌ فایل {audio_file} پیدا نشد!")