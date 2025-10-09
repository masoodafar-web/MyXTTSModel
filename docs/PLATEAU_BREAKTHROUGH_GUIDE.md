# راه‌حل فوری برای Loss Plateau

> **✅ Implementation Status**: The `plateau_breaker` optimization level is now fully implemented in `train_main.py` and ready to use. All configuration changes mentioned in this guide have been applied.

## آنالیز مشکل

### 🔍 علت‌های اصلی plateau در 2.5:

1. **Learning Rate بالا**: 8e-05 → 1.5e-05 (کاهش 80%)
2. **Loss weights نامتعادل**: mel_loss=2.5, stop_loss=0.64
3. **Scheduler plateau**: cosine restart در فاز flat
4. **Gradient clipping loose**: 0.8 → 0.3 (control بهتر)

## ✅ راه‌حل‌های اعمال شده:

### 1. **PLATEAU_BREAKER Configuration**
```bash
python3 train_main.py --optimization-level plateau_breaker
```

**تغییرات کلیدی:**
- Learning rate: 8e-05 → 1.5e-05 (80% کاهش)
- Mel loss weight: 2.5 → 2.0 (متعادل‌تر)
- KL loss weight: 1.8 → 1.2 (کاهش)
- Gradient clip: 0.8 → 0.3 (control سخت‌تر)
- Scheduler restart: هر 100 epoch (بیشتر)

### 2. **اسکریپت فوری:**
```bash
bash breakthrough_training.sh
```

## 🎯 انتظارات:

1. **Loss کاهش تا 2.2-2.3** در 10 epoch
2. **Validation loss بهبود** و convergence پایدار
3. **تعادل بهتر** بین mel_loss و stop_loss
4. **کاهش نوسانات** و training stable

## 📊 نکات مهم:

- **صبر کنید**: نتایج در 5-10 epoch اول قابل مشاهده
- **Monitor کنید**: validation loss مهم‌تر از train loss
- **Balance توجه**: mel_loss باید به سمت stop_loss متعادل شه

## 🚀 اجرای فوری:
```bash
# متوقف کردن training فعلی
pkill -f "python3 train_main.py"

# شروع با تنظیمات جدید
bash breakthrough_training.sh
```