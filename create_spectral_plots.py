#!/usr/bin/env python3
"""
ایجاد نمودار spectral analysis برای مقایسه کیفیت صدا
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# تنظیم فونت برای فارسی (اختیاری)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

def create_spectral_comparison():
    """ایجاد نمودار مقایسه‌ای spectral"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Audio Quality Comparison - After 1 Epoch Training', fontsize=16)
    
    # فایل جدید (بعد از epoch 1)
    new_file = "outputs/audio_samples/test_after_epoch1.wav"
    old_file = "outputs/audio_samples/output.wav"
    
    files = [
        (new_file, "After Epoch 1 Training", "green"),
        (old_file, "Before Training", "red")
    ]
    
    for i, (file_path, title, color) in enumerate(files):
        if os.path.exists(file_path):
            try:
                # بارگذاری صدا
                y, sr = librosa.load(file_path, sr=22050)
                
                # محاسبه Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # محاسبه STFT برای spectrogram
                stft = librosa.stft(y)
                spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
                
                # نمودار 1: Mel Spectrogram
                ax1 = axes[0, i] if i < 2 else axes[0, i-2]
                img1 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
                ax1.set_title(f'{title}\nMel Spectrogram')
                plt.colorbar(img1, ax=ax1, format='%+2.0f dB')
                
                # نمودار 2: Linear Spectrogram
                ax2 = axes[1, i] if i < 2 else axes[1, i-2]
                img2 = librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
                ax2.set_title(f'{title}\nLinear Spectrogram')
                plt.colorbar(img2, ax=ax2, format='%+2.0f dB')
                
                # محاسبه آمار
                rms = np.sqrt(np.mean(y**2))
                duration = len(y) / sr
                
                # اضافه کردن متن آمار
                stats_text = f'RMS: {rms:.4f}\nDuration: {duration:.2f}s'
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                print(f"✅ تحلیل شد: {title}")
                print(f"   - RMS Energy: {rms:.4f}")
                print(f"   - Duration: {duration:.2f}s")
                
            except Exception as e:
                print(f"❌ خطا در تحلیل {file_path}: {e}")
        else:
            print(f"❌ فایل پیدا نشد: {file_path}")
    
    plt.tight_layout()
    
    # ذخیره فایل
    output_path = "outputs/audio_samples/spectral_comparison_epoch1.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 نمودار ذخیره شد در: {output_path}")
    
    # نمایش آدرس کامل
    full_path = os.path.abspath(output_path)
    print(f"📍 مسیر کامل: {full_path}")
    
    plt.close()
    
    return output_path

def create_frequency_analysis():
    """تحلیل فرکانسی ساده"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Frequency Analysis Comparison', fontsize=14)
    
    new_file = "outputs/audio_samples/test_after_epoch1.wav"
    old_file = "outputs/audio_samples/output.wav"
    
    files = [
        (new_file, "After Epoch 1", "green"),
        (old_file, "Before Training", "red")
    ]
    
    for i, (file_path, label, color) in enumerate(files):
        if os.path.exists(file_path):
            try:
                y, sr = librosa.load(file_path, sr=22050)
                
                # FFT
                fft = np.fft.fft(y)
                freqs = np.fft.fftfreq(len(fft), 1/sr)
                magnitude = np.abs(fft)
                
                # فقط فرکانس‌های مثبت
                positive_freqs = freqs[:len(freqs)//2]
                positive_magnitude = magnitude[:len(magnitude)//2]
                
                # نمودار
                axes[i].semilogy(positive_freqs, positive_magnitude, color=color, alpha=0.7)
                axes[i].set_title(f'{label}')
                axes[i].set_xlabel('Frequency (Hz)')
                axes[i].set_ylabel('Magnitude')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim(0, 8000)  # فقط تا 8kHz
                
            except Exception as e:
                print(f"❌ خطا در تحلیل فرکانسی {file_path}: {e}")
    
    plt.tight_layout()
    
    # ذخیره
    freq_output = "outputs/audio_samples/frequency_analysis_epoch1.png"
    plt.savefig(freq_output, dpi=150, bbox_inches='tight')
    print(f"📊 تحلیل فرکانسی ذخیره شد در: {freq_output}")
    
    plt.close()
    
    return freq_output

if __name__ == "__main__":
    print("🎵 ایجاد نمودارهای تحلیل صدا...")
    
    # ایجاد نمودارهای مقایسه‌ای
    spectral_path = create_spectral_comparison()
    freq_path = create_frequency_analysis()
    
    print("\n📋 نمودارهای ایجاد شده:")
    print(f"1. 🎼 Spectral Comparison: {os.path.abspath(spectral_path)}")
    print(f"2. 📊 Frequency Analysis: {os.path.abspath(freq_path)}")
    
    # نمودار قدیمی
    old_spectral = "outputs/audio_samples/spectral_analysis.png"
    if os.path.exists(old_spectral):
        print(f"3. 📈 Previous Analysis: {os.path.abspath(old_spectral)}")
    
    print("\n✅ همه آماده! می‌توانید فایل‌ها را باز کنید و مقایسه کنید.")