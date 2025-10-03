#!/usr/bin/env python3
"""
Advanced Spectral Analysis for MyXTTS Model
تحلیل پیشرفته کیفیت طیفی برای تشخیص دقیق مشکل
"""
import os
import sys
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # choose GPU
sys.path.insert(0, os.path.abspath('.'))

from train_main import build_config
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import find_latest_checkpoint

def load_mel_spectrogram(audio_path, sr=22050, n_fft=2048, hop_length=256, n_mels=80):
    """Load audio and convert to mel spectrogram."""
    try:
        audio, sr_file = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr_file != sr:
            audio = librosa.resample(audio, orig_sr=sr_file, target_sr=sr)
        
        # محاسبه mel spectrogram
        S = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            fmin=0, fmax=8000
        )
        
        # تبدیل به log scale (مشابه training)
        mel_spec = np.log(np.maximum(S, 1e-5)).T  # [time, n_mels]
        
        return mel_spec, audio
        
    except Exception as e:
        print(f"خطا در بارگذاری {audio_path}: {e}")
        return None, None

def spectral_convergence(spec_pred, spec_target):
    """محاسبه Spectral Convergence."""
    num = np.linalg.norm(np.abs(spec_pred) - np.abs(spec_target), ord='fro')
    den = np.linalg.norm(np.abs(spec_target), ord='fro')
    return num / (den + 1e-8)

def log_spectral_distance(spec_pred, spec_target):
    """محاسبه Log Spectral Distance."""
    return np.sqrt(np.mean((spec_pred - spec_target)**2))

def spectral_distortion(spec_pred, spec_target):
    """محاسبه Spectral Distortion (dB)."""
    # تبدیل به magnitude spectrum
    mag_pred = np.abs(spec_pred)
    mag_target = np.abs(spec_target)
    
    # محاسبه در فضای dB
    db_pred = 20 * np.log10(np.maximum(mag_pred, 1e-7))
    db_target = 20 * np.log10(np.maximum(mag_target, 1e-7))
    
    return np.sqrt(np.mean((db_pred - db_target)**2))

def mel_cepstral_distortion(spec_pred, spec_target, n_mfcc=13):
    """محاسبه Mel-Cepstral Distortion."""
    try:
        # تبدیل به linear scale برای MFCC
        mel_pred_lin = np.exp(spec_pred)
        mel_target_lin = np.exp(spec_target)
        
        # محاسبه MFCC (نیاز به transpose برای librosa)
        mfcc_pred = librosa.feature.mfcc(S=mel_pred_lin.T, n_mfcc=n_mfcc)
        mfcc_target = librosa.feature.mfcc(S=mel_target_lin.T, n_mfcc=n_mfcc)
        
        # برگرداندن به format اصلی
        mfcc_pred = mfcc_pred.T  # [time, n_mfcc]
        mfcc_target = mfcc_target.T
        
        # تطبیق طول
        min_len = min(mfcc_pred.shape[0], mfcc_target.shape[0])
        mfcc_pred = mfcc_pred[:min_len]
        mfcc_target = mfcc_target[:min_len]
        
        # محاسبه MCD
        diff = mfcc_pred - mfcc_target
        mcd = np.sqrt(np.mean(diff**2, axis=0))
        
        return np.mean(mcd[1:])  # حذف C0 (انرژی کل)
        
    except Exception as e:
        print(f"خطا در محاسبه MCD: {e}")
        return float('inf')

def analyze_spectral_quality():
    """تحلیل جامع کیفیت طیفی."""
    
    print("🔍 Advanced Spectral Quality Analysis")
    print("=" * 60)
    
    # بارگذاری مدل
    config = build_config(model_size='normal', checkpoint_dir='./checkpointsmain')
    latest_checkpoint = find_latest_checkpoint('./checkpointsmain')
    
    print(f"📦 Loading model: {latest_checkpoint}")
    
    # تنظیم GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    inf = XTTSInference(config=config, checkpoint_path=latest_checkpoint)
    inf.enable_training()  # برای teacher forcing
    
    # بارگذاری فایل هدف
    target_audio_paths = [
        '../dataset/dataset_train/wavs/LJ010-0001.wav',
        'speaker.wav'
    ]
    
    target_path = None
    for path in target_audio_paths:
        if Path(path).exists():
            target_path = path
            break
    
    if not target_path:
        print("❌ هیچ فایل صوتی هدف پیدا نشد!")
        return
    
    print(f"🎵 Target audio: {target_path}")
    
    # بارگذاری mel spectrogram هدف
    mel_target_librosa, audio_target = load_mel_spectrogram(target_path)
    
    if mel_target_librosa is None:
        print("❌ خطا در بارگذاری mel spectrogram")
        return
    
    # تولید mel با مدل (Teacher Forcing)
    text = "The Chronicles of Newgate Volume two by Arthur Griffiths."
    processed = inf._preprocess_text(text, config.data.language)
    seq = inf.text_processor.text_to_sequence(processed)
    text_tensor = tf.constant([seq], dtype=tf.int32)
    text_lengths = tf.constant([len(seq)], dtype=tf.int32)
    
    # پردازش صوت مشابه training
    audio_processed = inf.audio_processor.preprocess_audio(audio_target)
    mel_target_model = inf.audio_processor.wav_to_mel(audio_processed).T
    mel_tensor = tf.constant(mel_target_model[np.newaxis, ...], dtype=tf.float32)
    mel_lengths = tf.constant([mel_target_model.shape[0]], dtype=tf.int32)
    
    print(f"📊 Data shapes:")
    print(f"   Librosa mel: {mel_target_librosa.shape}")
    print(f"   Model target: {mel_target_model.shape}")
    
    # اجرای مدل
    print(f"\n🤖 Running model inference...")
    outputs = inf.model(
        text_inputs=text_tensor,
        mel_inputs=mel_tensor,
        text_lengths=text_lengths,
        mel_lengths=mel_lengths,
        training=True
    )
    
    mel_pred = outputs['mel_output'].numpy()[0]
    
    print(f"   Model output shape: {mel_pred.shape}")
    
    # تطبیق اندازه‌ها
    min_len = min(mel_target_model.shape[0], mel_pred.shape[0], mel_target_librosa.shape[0])
    mel_target_model = mel_target_model[:min_len]
    mel_pred = mel_pred[:min_len]
    mel_target_librosa = mel_target_librosa[:min_len]
    
    print(f"\n📈 Spectral Analysis Results:")
    print("-" * 40)
    
    # 1. آمارهای پایه
    print(f"🎯 Target (Model Processing):")
    print(f"   Shape: {mel_target_model.shape}")
    print(f"   Range: [{mel_target_model.min():.3f}, {mel_target_model.max():.3f}]")
    print(f"   Mean: {mel_target_model.mean():.3f}, Std: {mel_target_model.std():.3f}")
    
    print(f"\n🤖 Model Prediction:")
    print(f"   Shape: {mel_pred.shape}")
    print(f"   Range: [{mel_pred.min():.3f}, {mel_pred.max():.3f}]")
    print(f"   Mean: {mel_pred.mean():.3f}, Std: {mel_pred.std():.3f}")
    
    print(f"\n📚 Librosa Reference:")
    print(f"   Shape: {mel_target_librosa.shape}")
    print(f"   Range: [{mel_target_librosa.min():.3f}, {mel_target_librosa.max():.3f}]")
    print(f"   Mean: {mel_target_librosa.mean():.3f}, Std: {mel_target_librosa.std():.3f}")
    
    # 2. محاسبه معیارهای طیفی
    print(f"\n📊 Spectral Quality Metrics:")
    print("-" * 40)
    
    # Model vs Target
    mae_model = np.mean(np.abs(mel_target_model - mel_pred))
    rmse_model = np.sqrt(np.mean((mel_target_model - mel_pred)**2))
    sc_model = spectral_convergence(mel_pred, mel_target_model)
    lsd_model = log_spectral_distance(mel_pred, mel_target_model)
    
    print(f"🔥 Model vs Training Target:")
    print(f"   MAE: {mae_model:.4f}")
    print(f"   RMSE: {rmse_model:.4f}")
    print(f"   Spectral Convergence: {sc_model:.4f}")
    print(f"   Log Spectral Distance: {lsd_model:.4f}")
    
    # Librosa vs Model target (برای مقایسه preprocessing)
    mae_lib = np.mean(np.abs(mel_target_librosa - mel_target_model))
    rmse_lib = np.sqrt(np.mean((mel_target_librosa - mel_target_model)**2))
    sc_lib = spectral_convergence(mel_target_model, mel_target_librosa)
    
    print(f"\n📚 Librosa vs Model Preprocessing:")
    print(f"   MAE: {mae_lib:.4f}")
    print(f"   RMSE: {rmse_lib:.4f}")
    print(f"   Spectral Convergence: {sc_lib:.4f}")
    
    # 3. Mel-Cepstral Distortion
    print(f"\n🎼 Advanced Metrics:")
    print("-" * 40)
    
    try:
        mcd = mel_cepstral_distortion(mel_pred, mel_target_model)
        print(f"   Mel-Cepstral Distortion: {mcd:.4f}")
    except:
        print(f"   Mel-Cepstral Distortion: محاسبه نشد")
    
    # 4. تحلیل frequency-wise
    freq_wise_error = np.mean(np.abs(mel_target_model - mel_pred), axis=0)
    worst_freq_bins = np.argsort(freq_wise_error)[-5:]
    best_freq_bins = np.argsort(freq_wise_error)[:5]
    
    print(f"\n🔊 Frequency Analysis:")
    print(f"   Worst frequency bins: {worst_freq_bins} (error: {freq_wise_error[worst_freq_bins].mean():.3f})")
    print(f"   Best frequency bins: {best_freq_bins} (error: {freq_wise_error[best_freq_bins].mean():.3f})")
    
    # 5. تحلیل temporal
    temporal_error = np.mean(np.abs(mel_target_model - mel_pred), axis=1)
    print(f"\n⏰ Temporal Analysis:")
    print(f"   Temporal error range: [{temporal_error.min():.3f}, {temporal_error.max():.3f}]")
    print(f"   Mean temporal error: {temporal_error.mean():.3f}")
    
    # 6. نتیجه‌گیری
    print(f"\n🎯 Assessment:")
    print("=" * 40)
    
    if mae_model > 3.0:
        print("❌ Very poor spectral quality - model not learning mel generation")
    elif mae_model > 1.5:
        print("⚠️  Poor spectral quality - significant training needed")
    elif mae_model > 0.5:
        print("✅ Moderate spectral quality - some improvement needed")
    else:
        print("🎉 Good spectral quality - model generating proper mels")
    
    if sc_model > 0.8:
        print("❌ Very high spectral convergence - poor spectral structure")
    elif sc_model > 0.5:
        print("⚠️  High spectral convergence - spectral structure needs work") 
    elif sc_model > 0.2:
        print("✅ Moderate spectral convergence - reasonable structure")
    else:
        print("🎉 Low spectral convergence - excellent spectral structure")
    
    # چک کردن dynamic range
    pred_range = mel_pred.max() - mel_pred.min()
    target_range = mel_target_model.max() - mel_target_model.min() 
    range_ratio = pred_range / target_range
    
    print(f"\n📏 Dynamic Range Analysis:")
    print(f"   Target range: {target_range:.3f}")
    print(f"   Prediction range: {pred_range:.3f}")
    print(f"   Range ratio: {range_ratio:.3f}")
    
    if range_ratio < 0.1:
        print("❌ Model output is essentially flat - major training issue")
    elif range_ratio < 0.3:
        print("⚠️  Model output has very limited dynamic range")
    elif range_ratio < 0.7:
        print("✅ Model output has reasonable dynamic range")
    else:
        print("🎉 Model output has excellent dynamic range")
    
    # ذخیره نتایج برای بررسی بصری
    print(f"\n💾 Saving analysis results...")
    
    np.save('spectral_analysis_target.npy', mel_target_model)
    np.save('spectral_analysis_pred.npy', mel_pred)
    np.save('spectral_analysis_librosa.npy', mel_target_librosa)
    
    # ایجاد visualization ساده
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Target
        im1 = axes[0,0].imshow(mel_target_model.T, aspect='auto', origin='lower')
        axes[0,0].set_title('Target Mel (Model Processing)')
        axes[0,0].set_ylabel('Mel Bins')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Prediction
        im2 = axes[0,1].imshow(mel_pred.T, aspect='auto', origin='lower')
        axes[0,1].set_title('Model Prediction')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Difference
        diff = np.abs(mel_target_model - mel_pred)
        im3 = axes[1,0].imshow(diff.T, aspect='auto', origin='lower')
        axes[1,0].set_title('Absolute Difference')
        axes[1,0].set_ylabel('Mel Bins')
        axes[1,0].set_xlabel('Time Frames')
        plt.colorbar(im3, ax=axes[1,0])
        
        # Error distribution
        axes[1,1].hist(temporal_error, bins=50, alpha=0.7, label='Temporal Error')
        axes[1,1].set_xlabel('Error')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Error Distribution')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('spectral_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'spectral_analysis.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
    
    print(f"\n📋 Summary saved to analysis files")
    
    return {
        'mae': mae_model,
        'rmse': rmse_model,
        'spectral_convergence': sc_model,
        'log_spectral_distance': lsd_model,
        'range_ratio': range_ratio,
        'temporal_error_mean': temporal_error.mean()
    }

if __name__ == "__main__":
    results = analyze_spectral_quality()
    
    if results:
        print(f"\n🎊 Analysis Complete!")
        print(f"Key metrics: MAE={results['mae']:.3f}, SC={results['spectral_convergence']:.3f}, Range={results['range_ratio']:.3f}")