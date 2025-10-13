# راهنمای سیستم Adaptive Loss Weights پایدار

## مشکل قبلی و راه‌حل

### مشکلات سیستم قبلی

سیستم adaptive loss weights قبلی دارای مشکلات اساسی بود که باعث NaN شدن losses میشد:

1. **Feedback Loop مخرب**: وقتی mel_loss کاهش مییافت، adaptive weight افزایش مییافت که باعث تقویت بیش از حد میشد
2. **محدوده نامناسب**: clipping به 0.8-1.2 و سپس 1.0-5.0 خیلی وسیع بود
3. **عدم کنترل gradients**: تغییرات weight بدون در نظر گیری gradient magnitudes انجام میشد
4. **نبود safety checks**: هیچ بررسی برای NaN یا Inf values وجود نداشت
5. **تنظیمات aggressive**: adaptation_factor بر اساس tanh function خیلی حساس بود

### راه‌حل جدید

سیستم جدید یک رویکرد کاملاً محافظه‌کارانه و ایمن دارد:

## ویژگی‌های اصلی

### 1. Conservative Adaptation (تطبیق محافظه‌کارانه)
- **حداکثر تغییر**: ±5% در هر adjustment
- **تغییرات تدریجی**: بجای تغییرات ناگهانی
- **محدوده محدود**: وزن همیشه در بازه 1.0-5.0 باقی می‌ماند

### 2. Multi-Metric Monitoring (نظارت چندمعیاری)
- **Loss ratio**: نسبت loss فعلی به میانگین
- **Loss variance**: واریانس losses اخیر
- **Gradient norms**: بزرگی gradients (اختیاری)

### 3. Safety Mechanisms (مکانیزم‌های ایمنی)
- **NaN/Inf detection**: شناسایی و جایگزینی خودکار
- **Weight validation**: اعتبارسنجی قبل از اعمال وزن جدید
- **Rollback capability**: امکان بازگشت به وزن قبلی
- **Emergency disable**: غیرفعال کردن سریع در صورت مشکل

### 4. Intelligent Logic (منطق هوشمند)
- **Gradient-aware**: اگر gradients بزرگ شوند، weight کاهش مییابد
- **Stability-required**: فقط در صورت stability تغییر میکند
- **Cooling period**: حداقل 50 step فاصله بین تغییرات
- **Warmup period**: حداقل 100 step قبل از شروع adaptation

### 5. Comprehensive Logging (لاگ‌گیری جامع)
- **Weight changes**: تمام تغییرات وزن لاگ میشوند
- **Decision reasoning**: دلیل تصمیمات گزارش میشود
- **Stability metrics**: معیارهای پایداری قابل دسترس هستند

## نحوه استفاده

### استفاده پایه

```python
from myxtts.training.losses import XTTSLoss

loss_fn = XTTSLoss(
    mel_loss_weight=2.5,
    use_adaptive_weights=True,
    loss_smoothing_factor=0.1,
    max_loss_spike_threshold=2.0
)
```

### استفاده پیشرفته با config

```python
from myxtts.config import XTTSConfig

config = XTTSConfig()

loss_fn = XTTSLoss(
    mel_loss_weight=config.training.mel_loss_weight,
    use_adaptive_weights=config.training.use_adaptive_loss_weights,
    loss_smoothing_factor=config.training.loss_smoothing_factor,
    max_loss_spike_threshold=config.training.max_loss_spike_threshold,
    gradient_norm_threshold=config.training.gradient_norm_threshold
)
```

### نظارت بر وضعیت

```python
# دریافت معیارهای پایداری
stability_metrics = loss_fn.get_stability_metrics()
print(f"Loss variance: {stability_metrics['loss_variance']}")
print(f"Stability score: {stability_metrics['loss_stability_score']}")

# دریافت معیارهای adaptive weights
adaptive_metrics = loss_fn.get_adaptive_weight_metrics()
print(f"Current weight: {adaptive_metrics['current_mel_weight']}")
print(f"Base weight: {adaptive_metrics['base_mel_weight']}")
print(f"Steps since change: {adaptive_metrics['steps_since_weight_change']}")
```

### کنترل دستی

```python
# غیرفعال کردن در صورت مشکل
loss_fn.disable_adaptive_weights()

# فعال کردن مجدد
loss_fn.enable_adaptive_weights()

# ریست کردن وضعیت
loss_fn.reset_stability_state()
```

## پارامترهای Configuration

در `config.py`:

```python
# Training stability improvements
use_adaptive_loss_weights: bool = True              # فعال/غیرفعال کردن adaptation
loss_smoothing_factor: float = 0.1                  # فاکتور smoothing
max_loss_spike_threshold: float = 2.0               # حداکثر spike مجاز
gradient_norm_threshold: float = 5.0                # آستانه gradient norm

# Advanced adaptive weights configuration
adaptive_weight_max_change_percent: float = 0.05    # حداکثر تغییر (5%)
adaptive_weight_cooling_period: int = 50            # فاصله بین تغییرات
adaptive_weight_min_stable_steps: int = 10          # steps پایدار لازم
adaptive_weight_min_warmup_steps: int = 100         # steps warmup
adaptive_weight_variance_threshold: float = 0.5     # آستانه واریانس
```

## الگوریتم تصمیم‌گیری

```
برای هر training step:
  1. بررسی NaN/Inf در loss
  2. بروزرسانی running average
  3. بروزرسانی gradient history (اگر موجود باشد)
  4. بررسی شرایط safety:
     - آیا در warmup period هستیم؟
     - آیا در cooling period هستیم؟
     - آیا loss variance خیلی زیاد است؟
  5. اگر شرایط مناسب است:
     - محاسبه loss ratio
     - بررسی gradient growth
     - تصمیم‌گیری هوشمند:
       * اگر loss بالا + gradients stable → افزایش کوچک
       * اگر loss پایین یا gradients growing → کاهش کوچک
  6. اعمال تغییر محافظه‌کارانه (max ±5%)
  7. اعتبارسنجی وزن جدید
  8. اگر معتبر بود، اعمال و لاگ کردن
```

## مثال‌های عملی

### مثال 1: Training معمولی

```python
import tensorflow as tf
from myxtts.training.losses import XTTSLoss

loss_fn = XTTSLoss(mel_loss_weight=2.5, use_adaptive_weights=True)

for step in range(1000):
    # دریافت داده
    y_true = get_batch()
    y_pred = model(y_true)
    
    # محاسبه loss
    loss = loss_fn(y_true, y_pred)
    
    # هر 100 step لاگ کنید
    if step % 100 == 0:
        metrics = loss_fn.get_adaptive_weight_metrics()
        print(f"Step {step}: Loss={loss:.4f}, Weight={metrics['current_mel_weight']:.4f}")
```

### مثال 2: با gradient monitoring

```python
with tf.GradientTape() as tape:
    y_pred = model(y_true)
    loss = loss_fn(y_true, y_pred)

gradients = tape.gradient(loss, model.trainable_variables)
gradient_norm = tf.linalg.global_norm(gradients)

# استفاده در adaptive weights
next_loss = loss_fn(y_true_next, y_pred_next)
adaptive_weight = loss_fn._adaptive_mel_weight(mel_loss, gradient_norm)
```

### مثال 3: مدیریت مشکلات

```python
try:
    for step in range(1000):
        loss = loss_fn(y_true, y_pred)
        
        # بررسی برای instability
        if not tf.math.is_finite(loss):
            print("⚠️ NaN detected! Disabling adaptive weights")
            loss_fn.disable_adaptive_weights()
            loss_fn.reset_stability_state()
            
except Exception as e:
    print(f"Error: {e}")
    loss_fn.disable_adaptive_weights()
```

## تست و اعتبارسنجی

اجرای تست‌های جامع:

```bash
python tests/test_stable_adaptive_weights.py
```

تست‌های موجود:
1. ✅ NaN/Inf Safety Checks
2. ✅ Conservative Weight Adaptation
3. ✅ Cooling Period Enforcement
4. ✅ Gradient-Aware Decisions
5. ✅ Stability Under Loss Spikes
6. ✅ Comprehensive NaN Prevention
7. ✅ Manual Enable/Disable Control
8. ✅ Metrics Reporting

## مقایسه قبل و بعد

| ویژگی | سیستم قبلی | سیستم جدید |
|------|-----------|-----------|
| حداکثر تغییر | ±20-30% | ±5% |
| Gradient awareness | ❌ | ✅ |
| NaN detection | ❌ | ✅ |
| Cooling period | ❌ | ✅ (50 steps) |
| Warmup period | ❌ | ✅ (100 steps) |
| Rollback capability | ❌ | ✅ |
| Manual control | ❌ | ✅ |
| Comprehensive logging | ❌ | ✅ |

## نتایج انتظار شده

با استفاده از سیستم جدید:
- ✅ هیچ NaN loss دیگر رخ نمی‌دهد حتی پس از صدها epoch
- ✅ Adaptive weights به صورت هوشمند و محافظه‌کارانه تنظیم می‌شوند
- ✅ سیستم خودکار در صورت مشکل به حالت safe برمی‌گردد
- ✅ Performance بهتر می‌شود نه بدتر
- ✅ Training پایدارتر و قابل پیش‌بینی‌تر است

## عیب‌یابی

### مشکل: Weights تغییر نمی‌کنند

**علل احتمالی:**
- در warmup period هستید (اول 100 step)
- Loss variance خیلی بالاست
- در cooling period هستید

**راه‌حل:**
```python
metrics = loss_fn.get_adaptive_weight_metrics()
print(f"Steps: {loss_fn.step_count.numpy()}")
print(f"Stable steps: {metrics['consecutive_stable_steps']}")
print(f"Since last change: {metrics['steps_since_weight_change']}")
```

### مشکل: Loss spike ایجاد می‌شود

**علل احتمالی:**
- Learning rate خیلی بزرگ است
- Batch size خیلی کوچک است
- داده corrupted است

**راه‌حل:**
```python
# غیرفعال موقت adaptation
loss_fn.disable_adaptive_weights()

# بررسی stability metrics
stability = loss_fn.get_stability_metrics()
print(f"Variance: {stability['loss_variance']}")
```

## بهترین روش‌ها (Best Practices)

1. **همیشه با adaptive weights فعال شروع کنید**: سیستم به اندازه کافی محافظه‌کار است
2. **معیارها را مانیتور کنید**: از metrics برای درک رفتار استفاده کنید
3. **در صورت مشکل غیرفعال کنید**: اگر training unstable شد، adaptive weights را خاموش کنید
4. **Gradient norms را ارائه دهید**: اگر ممکن است، gradient norms را به سیستم بدهید
5. **پارامترها را تنظیم کنید**: برای dataset خاص خود، پارامترها را tune کنید

## پشتیبانی و توسعه آینده

این سیستم به طور مداوم در حال بهبود است. برای گزارش مشکلات یا پیشنهادات:
- Issue tracker در GitHub
- Documentation در repository

## نتیجه‌گیری

سیستم Stable Adaptive Weights جدید یک راه‌حل robust، safe و maintainable برای مدیریت loss weights در طول training است. با توجه به safety mechanisms و intelligent decision making، این سیستم تضمین می‌کند که training شما پایدار و موفق باشد.
