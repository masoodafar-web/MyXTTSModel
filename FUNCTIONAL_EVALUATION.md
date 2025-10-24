# ارزیابی عملکردی پروژه MyXTTS - Functional Evaluation of MyXTTS Project

## خلاصه / Executive Summary

این سند یک ارزیابی جامع از مشکلات احتمالی عملکردی پروژه MyXTTS ارائه می‌دهد، با تمرکز بر مسائل همگرایی (convergence) و تطبیق مدل (model matching).

This document provides a comprehensive evaluation of potential functional issues in the MyXTTS project, focusing on convergence and model matching problems.

---

## 1. مشکلات همگرایی مدل / Model Convergence Issues

### 1.1 مشکل: همگرایی کند یا توقف در سطح بالای Loss
**Problem: Slow Convergence or Loss Plateau**

#### علائم / Symptoms:
- Loss در مقادیر بالا (2.5-2.8) متوقف می‌شود
- Loss is stuck at high values (2.5-2.8)
- کاهش بسیار کند loss در طول epochs
- Very slow loss reduction across epochs
- نوسانات زیاد در loss بدون روند کاهشی واضح
- High loss oscillations without clear decreasing trend

#### علل ریشه‌ای / Root Causes:

**الف) وزن‌های نادرست loss:**
```python
# ❌ مقادیر خطرناک / Dangerous values
mel_loss_weight: 35.0 - 45.0  # خیلی زیاد!
kl_loss_weight: 10.0          # بیش از حد

# ✅ مقادیر ایمن / Safe values  
mel_loss_weight: 2.5 - 5.0
kl_loss_weight: 0.5 - 2.0
```

**ب) اندازه batch نامناسب برای اندازه مدل:**
```python
# ❌ Mismatches
tiny model + batch_size 64    # مدل کوچک، batch بزرگ
big model + batch_size 4      # مدل بزرگ، batch کوچک

# ✅ Optimal matches
tiny model  → batch_size 8-16
small model → batch_size 16-32
normal model → batch_size 32-64
big model → batch_size 16-32
```

**ج) learning rate نامناسب:**
- learning rate خیلی بالا → نوسانات و عدم همگرایی
- learning rate خیلی پایین → همگرایی بسیار کند
- نیاز به تنظیم بر اساس اندازه مدل

#### راه‌حل‌های پیشنهادی / Recommended Solutions:

1. **استفاده از optimization level مناسب:**
```bash
# برای همگرایی سریع‌تر
python train_main.py --optimization-level enhanced

# برای حل plateau
python train_main.py --optimization-level plateau_breaker
```

2. **تنظیم batch size با توجه به model size:**
```bash
# Tiny model
python train_main.py --model-size tiny --batch-size 16

# Normal model  
python train_main.py --model-size normal --batch-size 32
```

3. **فعال‌سازی adaptive loss weights:**
```python
enable_adaptive_loss_weights: true
adaptive_weight_update_interval: 50
```

### 1.2 مشکل: انفجار یا محو گرادیان‌ها
**Problem: Gradient Explosion or Vanishing**

#### علائم / Symptoms:
- Loss به طور ناگهانی به NaN یا Inf می‌رسد
- Loss suddenly becomes NaN or Inf
- پارامترهای مدل بسیار بزرگ یا صفر می‌شوند
- Model parameters become extremely large or zero
- هشدارهای gradient overflow در لاگ
- Gradient overflow warnings in logs

#### علل ریشه‌ای / Root Causes:
- عدم gradient clipping یا clipping ناکافی
- learning rate بسیار بالا
- وزن‌های loss بسیار بزرگ
- معماری مدل ناپایدار (لایه‌های بسیار عمیق بدون normalization)

#### راه‌حل‌ها / Solutions:

```python
# در config.yaml
gradient_clip_norm: 0.5      # محدود کردن گرادیان‌ها
gradient_clip_value: 1.0     # جلوگیری از مقادیر خیلی بزرگ

# استفاده از gradient checkpointing برای مدل‌های بزرگ
enable_gradient_checkpointing: true
```

---

## 2. مشکلات تطبیق مدل / Model Matching Issues

### 2.1 مشکل: عدم همخوانی Text و Audio
**Problem: Text-Audio Misalignment**

#### علائم / Symptoms:
- خروجی صوتی با متن ورودی همخوانی ندارد
- Audio output doesn't match input text
- تأخیر یا تعجیل در pronunciation
- Delays or rushing in pronunciation
- کلمات اضافی یا حذف شده در خروجی
- Extra or missing words in output

#### علل ریشه‌ای / Root Causes:

**الف) Duration Predictor ضعیف:**
```python
# مشکل: predictor خیلی ساده
duration_predictor = Dense(1, activation='relu')  # خیلی ساده!

# بهتر: معماری قوی‌تر
class DurationPredictor:
    def __init__(self):
        self.conv_layers = [Conv1D(...) for _ in range(3)]
        self.lstm = LSTM(128)
        self.output = Dense(1, activation='softplus')
```

**ب) Attention Mechanism ناکارآمد:**
- استفاده از vanilla attention بدون monotonic constraint
- عدم استفاده از location-sensitive attention
- نداشتن forward attention mechanism

#### راه‌حل‌ها / Solutions:

1. **بهبود Duration Prediction:**
```python
# فعال‌سازی duration predictor پیشرفته
use_duration_predictor: true
duration_predictor_layers: 3
duration_predictor_hidden_dim: 256
```

2. **استفاده از Guided Attention:**
```python
use_guided_attention: true
guided_attention_sigma: 0.2
guided_attention_loss_weight: 1.0
```

3. **افزودن Forward Attention:**
```python
use_forward_attention: true
forward_attention_mask: true
```

### 2.2 مشکل: عدم انتقال سبک گوینده
**Problem: Poor Speaker Style Transfer**

#### علائم / Symptoms:
- خروجی شبیه speaker reference نیست
- Output doesn't resemble reference speaker
- عدم انتقال prosody و emotion
- Lack of prosody and emotion transfer
- صدای همه speaker ها یکسان است
- All speakers sound the same

#### علل ریشه‌ای / Root Causes:

**الف) Speaker Encoder ضعیف:**
```python
# ❌ مشکل: embedding خیلی کوچک
speaker_embedding_dim: 64  # خیلی کم!

# ✅ بهتر
speaker_embedding_dim: 256  # استاندارد
speaker_embedding_dim: 512  # برای کیفیت بالاتر
```

**ب) عدم استفاده از Global Style Tokens (GST):**
```python
# بدون GST → محدودیت در کنترل prosody
enable_gst: false

# با GST → کنترل بهتر سبک
enable_gst: true
gst_num_style_tokens: 10
gst_num_heads: 4
```

#### راه‌حل‌ها / Solutions:

1. **فعال‌سازی و تنظیم GST:**
```bash
python train_main.py \
    --enable-gst \
    --gst-num-style-tokens 12 \
    --gst-style-token-dim 128
```

2. **بهبود Speaker Encoding:**
```python
# استفاده از pretrained speaker encoder
use_pretrained_speaker_encoder: true
speaker_encoder_model: "resemblyzer"  # یا "deep_speaker"

# افزایش ظرفیت embedding
speaker_embedding_dim: 512
```

3. **استفاده از Contrastive Learning:**
```python
use_contrastive_speaker_loss: true
contrastive_loss_weight: 0.5
contrastive_loss_temperature: 0.07
```

---

## 3. مشکلات پایداری آموزش / Training Stability Issues

### 3.1 مشکل: نوسانات شدید GPU Utilization
**Problem: Severe GPU Utilization Oscillations**

#### علائم / Symptoms:
```
GPU Utilization Pattern:
90% → 5% → 90% → 5% → ...  # نوسان شدید
```

#### علل / Causes:
- عدم استفاده مناسب از data pipeline
- Retracing مکرر tf.function
- عدم استفاده از static shapes

#### راه‌حل‌ها / Solutions:

```bash
# استفاده از static shapes
python train_main.py --enable-static-shapes

# فعال‌سازی data prefetching
python train_main.py --buffer-size 100

# Multi-GPU با memory isolation
python train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-memory-isolation
```

### 3.2 مشکل: مصرف بیش از حد حافظه
**Problem: Excessive Memory Consumption**

#### علائم / Symptoms:
- OOM (Out of Memory) errors
- کاهش سرعت آموزش به دلیل memory swapping
- Slow training due to memory swapping

#### راه‌حل‌ها / Solutions:

```python
# Gradient accumulation برای batch های بزرگتر با حافظه کم
gradient_accumulation_steps: 4

# Mixed precision training
use_mixed_precision: true

# Gradient checkpointing
enable_gradient_checkpointing: true
```

---

## 4. مشکلات کیفیت خروجی / Output Quality Issues

### 4.1 مشکل: کیفیت پایین صوتی خروجی
**Problem: Poor Audio Quality Output**

#### علائم / Symptoms:
- نویز در خروجی
- Noisy output
- تحریف در صدا
- Audio distortion
- کیفیت پایین mel spectrogram
- Low quality mel spectrograms

#### علل / Causes:

**الف) Vocoder نامناسب:**
```python
# استفاده از vocoder ساده
vocoder_type: "griffin_lim"  # کیفیت پایین

# بهتر: استفاده از neural vocoder
vocoder_type: "hifigan"      # کیفیت بالا
vocoder_type: "univnet"      # بهترین کیفیت
```

**ب) تنظیمات Mel Spectrogram:**
```python
# ❌ تنظیمات نامناسب
n_mels: 40          # خیلی کم
hop_length: 512     # خیلی بزرگ

# ✅ تنظیمات بهینه
n_mels: 80          # استاندارد برای TTS
hop_length: 256     # وضوح زمانی بهتر
```

#### راه‌حل‌ها / Solutions:

1. **استفاده از Neural Vocoder:**
```python
vocoder:
  type: "hifigan"
  checkpoint: "path/to/pretrained/hifigan"
  
# یا در training
python train_main.py --vocoder-type hifigan
```

2. **بهینه‌سازی Mel Spectrogram:**
```python
audio:
  n_mels: 80
  n_fft: 1024
  hop_length: 256
  win_length: 1024
  fmin: 0
  fmax: 8000
```

### 4.2 مشکل: عدم تنوع در خروجی
**Problem: Lack of Variation in Output**

#### علائم / Symptoms:
- تمام خروجی‌ها monotone هستند
- All outputs sound monotone
- عدم تنوع در prosody
- Lack of prosody variation

#### راه‌حل‌ها / Solutions:

```python
# فعال‌سازی prosody modeling
enable_prosody_prediction: true

# استفاده از variational components
use_variational_encoder: true
vae_latent_dim: 16

# GST برای کنترل سبک
enable_gst: true
```

---

## 5. جدول خلاصه مشکلات و راه‌حل‌ها / Summary Table

| مشکل / Issue | علامت / Symptom | راه‌حل / Solution | اولویت / Priority |
|--------------|----------------|-------------------|-------------------|
| Loss Plateau | Loss stuck at 2.5-2.8 | Use `--optimization-level plateau_breaker` | 🔴 High |
| Gradient Explosion | NaN/Inf loss | Add gradient clipping | 🔴 High |
| Text-Audio Misalignment | Wrong pronunciation | Improve duration predictor | 🟡 Medium |
| Poor Speaker Transfer | All voices same | Enable GST + larger embeddings | 🟡 Medium |
| GPU Oscillation | 90%→5%→90% | Enable static shapes | 🔴 High |
| OOM Errors | Memory overflow | Gradient accumulation + checkpointing | 🟡 Medium |
| Poor Audio Quality | Noisy output | Use HiFiGAN vocoder | 🟢 Low |
| Monotone Output | No variation | Enable prosody prediction + GST | 🟢 Low |

---

## 6. اسکریپت تشخیص خودکار مشکلات / Automatic Issue Detection Script

برای تشخیص خودکار این مشکلات، از اسکریپت زیر استفاده کنید:

To automatically detect these issues, use the following script:

```bash
# اجرای تشخیص جامع
python utilities/diagnose_functional_issues.py --config config.yaml

# تشخیص مشکلات همگرایی
python utilities/diagnose_convergence.py --checkpoint path/to/checkpoint

# تشخیص مشکلات GPU
python utilities/diagnose_gpu_issues.py --profile-steps 100
```

---

## 7. توصیه‌های کلی / General Recommendations

### برای آموزش بهینه / For Optimal Training:

```bash
# پیکربندی پیشنهادی برای شروع
python train_main.py \
    --model-size normal \
    --optimization-level enhanced \
    --batch-size 32 \
    --enable-gst \
    --enable-static-shapes \
    --gradient-clip-norm 0.5 \
    --enable-evaluation \
    --evaluation-interval 25
```

### چک‌لیست قبل از شروع آموزش / Pre-Training Checklist:

- [ ] تنظیم `mel_loss_weight` بین 2.5-5.0
- [ ] انتخاب `batch_size` متناسب با `model_size`
- [ ] فعال‌سازی `gradient_clipping`
- [ ] استفاده از `--enable-static-shapes`
- [ ] بررسی dataset normalization
- [ ] انتخاب `vocoder` مناسب
- [ ] فعال‌سازی `GST` برای voice cloning
- [ ] تنظیم `learning_rate` بر اساس model size

---

## 8. نتیجه‌گیری / Conclusion

### مشکلات اصلی شناسایی شده / Main Issues Identified:

1. **مشکلات همگرایی** ناشی از وزن‌های نامناسب loss و batch size
2. **مشکلات تطبیق** به دلیل duration predictor و attention mechanism ضعیف  
3. **ناپایداری آموزش** از GPU utilization و memory management
4. **کیفیت پایین خروجی** ناشی از vocoder و mel spectrogram config

### اقدامات توصیه شده / Recommended Actions:

1. ✅ استفاده از `--optimization-level enhanced` به عنوان پیش‌فرض
2. ✅ فعال‌سازی `--enable-static-shapes` برای پایداری GPU
3. ✅ استفاده از `--enable-gst` برای بهبود voice cloning
4. ✅ اجرای اسکریپت‌های diagnostic قبل و حین آموزش
5. ✅ نظارت مداوم بر metrics و تنظیم hyperparameters

با رعایت این توصیه‌ها، می‌توان اکثر مشکلات عملکردی را پیش‌بینی و حل کرد.

By following these recommendations, most functional issues can be anticipated and resolved.

---

**تاریخ ارزیابی / Evaluation Date:** 2025-10-24  
**نسخه / Version:** 1.0  
**وضعیت / Status:** ✅ Complete
