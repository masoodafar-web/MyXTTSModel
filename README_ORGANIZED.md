# MyXTTS Voice Cloning Model

## 🎯 وضعیت فعلی

### ✅ مشکلات حل شده:
- **مشکل اصلی**: Model در حالت `trainable=False` بود - **حل شد**
- **Loss clipping مضر**: حذف شد
- **Mel loss weight نامناسب**: از 22.0 به 2.5 کاهش یافت
- **تشخیص علت noise در خروجی**: مدل هیچ یاد نگرفته بود

### 🚀 Training جاری:
```
Status: در حال آموزش
Loss: 0.9232 → 0.05 (mel component)
Progress: Epoch 1, Step 166/641
Expected: Loss واقعی در حال کاهش
```

## 📁 ساختار پروژه

```
MyXTTSModel/
├── 📝 تنظیمات اصلی
│   ├── train_main.py          # اسکریپت آموزش اصلی
│   ├── inference_main.py      # تولید صدا
│   ├── fixed_inference.py     # نسخه اصلاح شده
│   └── requirements.txt       # وابستگی‌ها
│
├── 🧠 مدل و تنظیمات
│   ├── myxtts/               # کد اصلی مدل
│   ├── configs/              # فایل‌های تنظیمات
│   └── checkpointsmain/      # ذخیره مدل
│
├── 📊 داده‌ها و خروجی‌ها
│   ├── data/                 # دیتاست
│   ├── outputs/
│   │   ├── audio_samples/    # فایل‌های صوتی تولید شده
│   │   └── analysis_results/ # نتایج تحلیل
│   └── logs/                 # گزارش‌های آموزش
│
└── 📦 بایگانی
    ├── debug_scripts/        # اسکریپت‌های تست و دیباگ
    ├── summaries/           # خلاصه‌ها و مستندات
    └── old_solutions/       # راه‌حل‌های قدیمی
```

## 🔧 نحوه استفاده

### آموزش مدل:
```bash
# آموزش با تنظیمات بهینه (فعلی)
python3 train_main.py --model-size normal --optimization-level enhanced --disable-gpu-stabilizer --batch-size 32

# آموزش با GPU stabilizer
python3 train_main.py --enable-gpu-stabilizer --batch-size 32
```

### تولید صدا:
```bash
# تولید صدا با مدل آموزش دیده
python3 fixed_inference.py --text "متن مورد نظر" --output output.wav --speaker-audio speaker.wav
```

### تست کیفیت:
```bash
# تحلیل طیفی پیشرفته
python3 archive/debug_scripts/advanced_spectral_analysis.py

# تست teacher forcing
python3 archive/debug_scripts/teacher_forcing_test.py
```

## 📈 پیشرفت اخیر

### قبل از اصلاح:
- ❌ Model output range: 0.4 (تقریباً صفر)
- ❌ Target range: 12.4 (طبیعی)
- ❌ Spectral Convergence: 0.997 (خیلی بد)
- ❌ Loss مضلل بود (کم شد اما کیفیت بهبود نیافت)

### بعد از اصلاح:
- ✅ Training واقعی شروع شد
- ✅ Mel loss: 0.11 → 0.05 (کاهش واقعی)
- ✅ Model trainable: True
- ✅ Loss weights بهینه شد

## 🎯 اهداف آینده

1. **کامل شدن آموزش**: انتظار 500 epoch
2. **تست کیفیت**: بررسی spectral metrics
3. **بهینه‌سازی**: تنظیم parameters بر اساس نتایج
4. **Voice cloning**: تست کیفیت کلون صدا

## 🆘 رفع مشکل

### مشکلات احتمالی:
```bash
# OOM Error
--batch-size 16 --grad-accum 4

# Training آهسته
--enable-gpu-stabilizer

# کیفیت پایین
--model-size big --optimization-level enhanced
```

### لاگ‌ها:
- Training logs: `logs/run_*.log`
- TensorBoard: `checkpointsmain/tensorboard`
- GPU stats: فعال با `--enable-gpu-stabilizer`

---

**آخرین بروزرسانی**: {{ تاریخ امروز }}
**وضعیت**: Training موفق در حال انجام ✅