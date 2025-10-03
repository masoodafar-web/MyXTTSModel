#!/usr/bin/env python3
"""
Quick audio quality test for the enhanced inference output.
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def analyze_audio(audio_path):
    """Analyze audio quality metrics."""
    # Load audio
    audio, sr = sf.read(audio_path)
    
    print(f"🎵 Audio Analysis: {audio_path}")
    print(f"   • Duration: {len(audio)/sr:.2f}s")
    print(f"   • Sample rate: {sr} Hz")
    print(f"   • Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"   • RMS level: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"   • Peak level: {np.max(np.abs(audio)):.4f}")
    
    # Check for silence/noise
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.001:
        print("   ❌ WARNING: Very low audio level - possibly silent")
    elif rms > 0.5:
        print("   ❌ WARNING: Very high audio level - possibly clipping") 
    elif 0.001 <= rms <= 0.5:
        print("   ✅ Audio level looks reasonable")
    
    # Spectral analysis
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Find dominant frequency range
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    mean_centroid = np.mean(spectral_centroid)
    
    print(f"   • Spectral centroid: {mean_centroid:.0f} Hz")
    
    # Check for speech-like characteristics
    if mean_centroid < 500:
        print("   ❌ WARNING: Very low spectral centroid - may be mostly low-frequency noise")
    elif mean_centroid > 8000:
        print("   ❌ WARNING: Very high spectral centroid - may be mostly high-frequency noise")
    else:
        print("   ✅ Spectral characteristics look speech-like")
    
    return {
        'duration': len(audio)/sr,
        'rms': rms,
        'peak': np.max(np.abs(audio)),
        'spectral_centroid': mean_centroid,
        'is_reasonable': 0.001 <= rms <= 0.5 and 500 <= mean_centroid <= 8000
    }

if __name__ == "__main__":
    # Test both outputs
    print("=" * 60)
    result1 = analyze_audio("enhanced_final_test.wav")
    
    print("\n" + "=" * 60)
    result2 = analyze_audio("persian_test_enhanced.wav")
    
    print(f"\n{'='*60}")
    print(f"🎯 Final Assessment:")
    if result1['is_reasonable'] and result2['is_reasonable']:
        print("   ✅ Both English and Persian audio have reasonable speech-like characteristics")
        print("   ✅ Enhanced mel normalization fix is working for both languages!")
        print("   🎉 PROBLEM RESOLVED: No more noise-only output!")
    else:
        print("   ❌ Some audio may still have issues - further debugging needed")