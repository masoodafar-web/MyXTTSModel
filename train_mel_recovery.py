#!/usr/bin/env python3
"""
Training with Teacher Forcing Monitoring
تریننگ با نظارت بر Teacher Forcing
"""
import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
from datetime import datetime

# اضافه کردن مسیرهای لازم
sys.path.insert(0, os.path.abspath('.'))

from train_main import build_config
from myxtts.training.trainer import XTTSTrainer
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import find_latest_checkpoint

class TeacherForcingMonitor:
    """Monitor teacher forcing performance during training."""
    
    def __init__(self, config, test_audio_path='speaker.wav'):
        self.config = config
        self.test_audio_path = test_audio_path
        self.test_text = "The Chronicles of Newgate, Volume two by Arthur Griffiths."
        
        # بارگذاری فایل تست
        try:
            self.test_audio, sr = sf.read(test_audio_path)
            if self.test_audio.ndim > 1:
                self.test_audio = self.test_audio.mean(axis=1)
            print(f"✅ Test audio loaded: {test_audio_path}")
        except Exception as e:
            print(f"❌ Could not load test audio: {e}")
            self.test_audio = None
    
    def evaluate_model(self, model, audio_processor, text_processor, step):
        """Evaluate model using teacher forcing."""
        if self.test_audio is None:
            return None
            
        try:
            # پردازش متن
            processed_text = XTTSInference._preprocess_text(None, self.test_text, self.config.data.language)
            seq = text_processor.text_to_sequence(processed_text)
            text_tensor = tf.constant([seq], dtype=tf.int32)
            text_lengths = tf.constant([len(seq)], dtype=tf.int32)
            
            # پردازش صوت
            audio_processed = audio_processor.preprocess_audio(self.test_audio)
            mel_target = audio_processor.wav_to_mel(audio_processed).T
            mel_tensor = tf.constant(mel_target[np.newaxis, ...], dtype=tf.float32)
            mel_lengths = tf.constant([mel_target.shape[0]], dtype=tf.int32)
            
            # اجرای مدل
            outputs = model(
                text_inputs=text_tensor,
                mel_inputs=mel_tensor, 
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=False  # evaluation mode
            )
            
            mel_pred = outputs['mel_output'].numpy()[0]
            
            # محاسبه آمارها
            target_stats = {
                'min': float(mel_target.min()),
                'max': float(mel_target.max()),
                'mean': float(mel_target.mean()),
                'std': float(mel_target.std())
            }
            
            pred_stats = {
                'min': float(mel_pred.min()),
                'max': float(mel_pred.max()),
                'mean': float(mel_pred.mean()),
                'std': float(mel_pred.std())
            }
            
            # محاسبه خطاها
            mae = float(np.mean(np.abs(mel_target - mel_pred)))
            rmse = float(np.sqrt(np.mean((mel_target - mel_pred)**2)))
            
            # محاسبه نسبت range
            target_range = target_stats['max'] - target_stats['min']
            pred_range = pred_stats['max'] - pred_stats['min']
            range_ratio = pred_range / max(target_range, 1e-8)
            
            results = {
                'step': step,
                'target_stats': target_stats,
                'pred_stats': pred_stats,
                'mae': mae,
                'rmse': rmse,
                'range_ratio': range_ratio
            }
            
            # نمایش نتایج
            print(f"\n📊 Teacher Forcing Results at Step {step}:")
            print(f"   🎯 Target: min={target_stats['min']:.3f}, max={target_stats['max']:.3f}, mean={target_stats['mean']:.3f}, std={target_stats['std']:.3f}")
            print(f"   🤖 Pred:   min={pred_stats['min']:.3f}, max={pred_stats['max']:.3f}, mean={pred_stats['mean']:.3f}, std={pred_stats['std']:.3f}")
            print(f"   📈 Errors: MAE={mae:.3f}, RMSE={rmse:.3f}")
            print(f"   📏 Range ratio: {range_ratio:.3f} ({'✅ Good' if range_ratio > 0.5 else '❌ Poor'})")
            
            return results
            
        except Exception as e:
            print(f"❌ Teacher forcing evaluation failed: {e}")
            return None

def train_with_monitoring():
    """Main training function with teacher forcing monitoring."""
    
    print("🚀 Starting Training with Teacher Forcing Monitoring")
    print("=" * 60)
    
    # بارگذاری کانفیگ ساده
    config = build_config(
        model_size='normal',
        checkpoint_dir='./checkpointsmain',
        batch_size=8,
        lr=0.0001,
        epochs=20
    )
    
    # تنظیم GPU
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices('GPU')[0], True
    )
    
    # پیدا کردن آخرین checkpoint
    latest_checkpoint = find_latest_checkpoint('./checkpointsmain')
    print(f"📦 Starting from checkpoint: {latest_checkpoint}")
    
    # ایجاد trainer
    trainer = XTTSTrainer(config=config)
    
    # بارگذاری checkpoint اگر وجود دارد
    if latest_checkpoint and os.path.exists(f"{latest_checkpoint}_model.weights.h5"):
        trainer.load_checkpoint(latest_checkpoint)
        print(f"✅ Loaded checkpoint: {latest_checkpoint}")
    else:
        print("🆕 Starting fresh training")
    
    # ایجاد teacher forcing monitor
    monitor = TeacherForcingMonitor(config)
    
    # تنظیمات تریننگ
    max_steps = 2000  # کم برای تست سریع
    eval_interval = 200  # هر 200 قدم ارزیابی کن
    save_interval = 500  # هر 500 قدم ذخیره کن
    
    print(f"🎯 Training Settings:")
    print(f"   • Max steps: {max_steps}")
    print(f"   • Eval interval: {eval_interval}")
    print(f"   • Save interval: {save_interval}")
    print(f"   • Batch size: {getattr(config.training, 'batch_size', 8)}")
    print(f"   • Learning rate: {getattr(config.training, 'learning_rate', 0.0001)}")
    
    # شروع حلقه تریننگ - شروع مجدد برای تست teacher forcing
    step = 0  # شروع از صفر برای monitoring
    
    try:
        while step < max_steps:
            # قدم تریننگ
            train_losses = trainer.train_step()
            step += 1
            
            # لاگ کردن loss
            if step % 50 == 0:
                mel_loss = train_losses.get('mel_loss', 0.0)
                total_loss = train_losses.get('total_loss', 0.0)
                print(f"Step {step:5d}: mel_loss={mel_loss:.4f}, total_loss={total_loss:.4f}")
            
            # ارزیابی teacher forcing
            if step % eval_interval == 0:
                print(f"\n🔍 Evaluating Teacher Forcing at Step {step}")
                results = monitor.evaluate_model(
                    trainer.model,
                    trainer.audio_processor,
                    trainer.text_processor,
                    step
                )
                
                # چک کردن پیشرفت
                if results:
                    if results['range_ratio'] > 0.3:
                        print("🎉 Model is starting to learn mel spectrograms!")
                    if results['mae'] < 3.0:
                        print("🎉 Error is decreasing significantly!")
            
            # ذخیره checkpoint
            if step % save_interval == 0:
                checkpoint_path = f"./checkpointsmain/recovery_checkpoint_{step}"
                trainer.save_checkpoint(checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    
    # ارزیابی نهایی
    print(f"\n🏁 Final Evaluation at Step {step}")
    final_results = monitor.evaluate_model(
        trainer.model,
        trainer.audio_processor, 
        trainer.text_processor,
        step
    )
    
    if final_results:
        if final_results['range_ratio'] > 0.5 and final_results['mae'] < 2.0:
            print("🎉 SUCCESS: Model has learned to generate proper mel spectrograms!")
            print("   Ready for inference testing.")
        else:
            print("⚠️  Model still needs more training.")
            print("   Consider continuing training or adjusting hyperparameters.")

if __name__ == "__main__":
    train_with_monitoring()