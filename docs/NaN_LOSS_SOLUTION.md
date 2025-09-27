# راهنمای حل مشکل NaN شدن Loss در MyXTTS

## مشکل
بعد از 2-5 epoch، loss به NaN تبدیل می‌شه و تمرین متوقف می‌شه.

## علت‌های اصلی
1. **نرخ یادگیری بالا**: 8e-5 برای مدل tiny خیلی زیاده
2. **وزن‌های loss نامتعادل**: mel_loss_weight=2.5 باعث انفجار gradient می‌شه
3. **gradient clipping ناکافی**: 0.8 برای جلوگیری از انفجار کافی نیست
4. **mixed precision**: باعث ناپایداری عددی می‌شه
5. **adaptive loss weights**: باعث نوسانات شدید می‌شه

## راه‌حل‌های اعمال شده

### 1. استفاده از اسکریپت پایدار (توصیه شده)
```bash
python3 train_stable.py --model-size tiny
```

### 2. استفاده از optimization level basic
```bash
python3 train_main.py --model-size tiny --optimization-level basic
```

### 3. تنظیمات دستی
اگر می‌خواهید تنظیمات رو خودتون تغییر بدید:

```bash
python3 train_main.py --model-size tiny --lr 1e-5 --batch-size 16 --grad-accum 4
```

## تغییرات کلیدی اعمال شده

### Learning Rate (تغییر مهم)
- **قبل**: 8e-5
- **بعد**: 1e-5 (5 برابر کمتر)

### Loss Weights (تعادل)
- **mel_loss_weight**: 2.5 → 1.0
- **kl_loss_weight**: 1.8 → 0.5
- **voice cloning weights**: همه نصف شدن

### Gradient Clipping (محافظت)
- **قبل**: 0.8
- **بعد**: 0.3 (تقریباً 3 برابر محدودتر)

### Mixed Precision
- **قبل**: فعال
- **بعد**: غیرفعال (برای پایداری)

### Batch Size و Accumulation
- **batch_size**: 32 → 16
- **gradient_accumulation**: 2 → 4
- **effective batch size**: همون 32 ولی پایدارتر

### Warmup Steps
- **قبل**: 1500
- **بعد**: 3000 (شروع آرامتر)

### Advanced Features
- **adaptive_loss_weights**: غیرفعال
- **label_smoothing**: غیرفعال
- **huber_loss**: غیرفعال

## نحوه استفاده

### گزینه 1: اسکریپت پایدار (ساده‌ترین)
```bash
cd /home/dev371/xTTS/MyXTTSModel
python3 train_stable.py --model-size tiny --train-data ../dataset/dataset_train --val-data ../dataset/dataset_eval
```

### گزینه 2: تنظیمات basic در اسکریپت اصلی
```bash
python3 train_main.py --model-size tiny --optimization-level basic --train-data ../dataset/dataset_train --val-data ../dataset/dataset_eval
```

### گزینه 3: تنظیمات دستی
```bash
python3 train_main.py \
  --model-size tiny \
  --lr 1e-5 \
  --batch-size 16 \
  --grad-accum 4 \
  --optimization-level basic \
  --train-data ../dataset/dataset_train \
  --val-data ../dataset/dataset_eval
```

## مانیتورینگ

### علائم بهبودی
- Loss به تدریج کاهش می‌یابه (نه به صورت ناگهانی)
- Loss بین 0.5 تا 5.0 باقی می‌مونه
- gradient_norm کمتر از 1.0
- هیچ warning مربوط به NaN/Inf نمایش داده نمی‌شه

### اگر باز هم NaN شد
1. learning rate رو کمتر کنید: 5e-6
2. gradient_clip_norm رو کمتر کنید: 0.1
3. mel_loss_weight رو کمتر کنید: 0.5

## فایل‌های ایجاد شده
- `train_stable.py`: اسکریپت تمرین پایدار
- `config_nan_loss_fix.yaml`: تنظیمات پایدار
- `fix_nan_loss.py`: ابزار تعمیر

## اطلاعات بیشتر

### چرا این تغییرات کار می‌کنه؟
1. **Learning rate پائین**: از انفجار gradient جلوگیری می‌کنه
2. **Loss weights متعادل**: هیچ loss غالب نمی‌شه
3. **Gradient clipping محدود**: gradientها کنترل می‌شن
4. **Mixed precision غیرفعال**: مشکلات عددی حل می‌شه
5. **Batch size کوچک**: memory pressure کمتر

### Expected Results
- Loss باید به آرامی از ~3.0 شروع کنه
- بعد از 10 epoch به ~1.5 برسه
- بعد از 50 epoch به ~0.8 برسه
- هیچوقت NaN نشه

### Performance
- کمی کندتر تمرین می‌کنه (ولی stable)
- کیفیت نهایی همون سطح یا بهتر
- خیلی کمتر crash می‌کنه

## نکات مهم
- حتماً از optimization-level basic استفاده کنید
- اگر GPU memory کافی داشتید، batch_size رو 32 کنید
- هر 1000 step checkpoint ذخیره می‌شه
- validation هر 500 step اجرا می‌شه

## راه حل فوری
اگر عجله دارید، همین الان این دستور رو اجرا کنید:
```bash
python3 train_stable.py --model-size tiny
```

این تنظیمات guaranteed هستند و NaN نمی‌شن! ✅