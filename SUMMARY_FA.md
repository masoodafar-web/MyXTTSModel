# خلاصه پیاده‌سازی - لاگ تصاویر و صداها در تنسوربورد

## مشکل اصلی
الان یسری تصویر هست توی فولدر training_samples که من میخوام توی tensorboard ببینمش و همچنین sample صوت هایی که تولید میشه هم میخوام توی تنسوربورد ببینم الان نمیبینمش آدرس تنسوربوردم هم logs

## راه‌حل

### ✅ کارهایی که انجام شد:

1. **لاگ کردن تصاویر از پوشه training_samples**
   - تصاویر PNG, JPG, JPEG, BMP, GIF پشتیبانی می‌شوند
   - در تنسوربورد زیر `training_samples/image/*` نمایش داده می‌شوند

2. **لاگ کردن صداها از پوشه training_samples**
   - فایل‌های WAV, MP3, FLAC, OGG پشتیبانی می‌شوند
   - در تنسوربورد زیر `training_samples/audio/*` نمایش داده می‌شوند

3. **لاگ کردن صداهای تولید شده**
   - این قابلیت قبلاً وجود داشت و حالا کامل شده
   - صداهای تولید شده زیر `text2audio_eval/*` نمایش داده می‌شوند

4. **تنظیمات قابل شخصی‌سازی**
   - آدرس پوشه samples قابل تغییر است
   - فاصله لاگ کردن قابل تنظیم است
   - آدرس تنسوربورد قابل تنظیم است (مثلاً logs)

## نحوه استفاده

### روش ساده:

```bash
# 1. پوشه را بسازید
mkdir training_samples

# 2. فایل‌های خود را کپی کنید
cp تصاویر_شما/*.png training_samples/
cp صداهای_شما/*.wav training_samples/

# 3. آموزش را شروع کنید
python train_main.py --train-data ../dataset/dataset_train

# 4. تنسوربورد را باز کنید
tensorboard --logdir=logs
```

سپس در مرورگر به `http://localhost:6006` بروید.

### روش پیشرفته:

```bash
# با تنظیمات سفارشی
python train_main.py \
    --train-data ../dataset/dataset_train \
    --training-samples-dir ./my_samples \
    --training-samples-log-interval 50 \
    --tensorboard-log-dir ./logs

# باز کردن تنسوربورد
tensorboard --logdir=./logs
```

## مشاهده در تنسوربورد

وقتی تنسوربورد را باز می‌کنید:

1. **برای دیدن تصاویر:**
   - به تب **IMAGES** بروید
   - دنبال `training_samples/image/*` بگردید

2. **برای دیدن صداها:**
   - به تب **AUDIO** بروید
   - دنبال `training_samples/audio/*` بگردید

3. **برای دیدن صداهای تولید شده:**
   - به تب **AUDIO** بروید
   - دنبال `text2audio_eval/*` بگردید

4. **برای دیدن اسپکتروگرام‌ها:**
   - به تب **IMAGES** بروید
   - دنبال `train/spectrogram_*` بگردید

## تنظیمات

### از طریق خط فرمان:

```bash
# تغییر آدرس پوشه samples
--training-samples-dir ./my_samples

# تغییر فاصله لاگ (هر چند قدم یک بار)
--training-samples-log-interval 100

# تغییر آدرس تنسوربورد
--tensorboard-log-dir ./logs

# غیرفعال کردن
--training-samples-log-interval 0
```

## فایل‌های پشتیبانی شده

### تصاویر:
- ✅ PNG (پیشنهادی)
- ✅ JPG/JPEG
- ✅ BMP
- ✅ GIF

### صوت:
- ✅ WAV (پیشنهادی، نیاز به نصب اضافی ندارد)
- ✅ MP3, FLAC, OGG (نیاز به نصب soundfile)

برای نصب soundfile:
```bash
pip install soundfile
```

## مثال کامل

### مرحله ۱: آماده‌سازی
```bash
# ساخت پوشه
mkdir training_samples

# کپی فایل‌ها
cp spectrograms/*.png training_samples/
cp reference_audio/*.wav training_samples/
```

### مرحله ۲: آموزش
```bash
python train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --batch-size 16 \
    --epochs 500
```

### مرحله ۳: مشاهده
```bash
# در ترمینال دیگر
tensorboard --logdir=./checkpointsmain/tensorboard

# یا اگر آدرس سفارشی دادید
tensorboard --logdir=./logs
```

### مرحله ۴: باز کردن در مرورگر
به آدرس زیر بروید:
```
http://localhost:6006
```

## اسکریپت خودکار

یک اسکریپت آماده برای ایجاد فایل‌های نمونه وجود دارد:

```bash
bash examples/tensorboard_samples_example.sh
```

این اسکریپت:
- پوشه training_samples را می‌سازد
- تصاویر نمونه ایجاد می‌کند
- فایل‌های صوتی نمونه ایجاد می‌کند
- دستورات لازم را نشان می‌دهد

## راهنماها

### راهنماهای انگلیسی:
- 📖 **راهنمای کامل**: [TENSORBOARD_SAMPLES_GUIDE.md](TENSORBOARD_SAMPLES_GUIDE.md)
- 📝 **خلاصه پیاده‌سازی**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

### راهنماهای فارسی:
- 🇮🇷 **راهنمای فارسی**: [TENSORBOARD_SAMPLES_GUIDE_FA.md](TENSORBOARD_SAMPLES_GUIDE_FA.md)
- 📁 **راهنمای پوشه**: [training_samples/README.md](training_samples/README.md)

## عیب‌یابی

### تصاویر نمایش داده نمی‌شوند؟

1. **بررسی پوشه:**
   ```bash
   ls -la training_samples/
   ```

2. **فرمت فایل را چک کنید** - باید PNG, JPG, JPEG, BMP یا GIF باشد

3. **فاصله لاگ را بررسی کنید** - پیش‌فرض هر ۱۰۰ قدم است

### صداها نمایش داده نمی‌شوند؟

1. **برای WAV نیاز به نصب اضافی نیست**

2. **برای MP3, FLAC, OGG باید soundfile نصب شود:**
   ```bash
   pip install soundfile
   ```

3. **فایل صوتی را تست کنید:**
   ```python
   import tensorflow as tf
   audio_bytes = tf.io.read_file('training_samples/sample.wav')
   audio, sr = tf.audio.decode_wav(audio_bytes)
   print(f"نرخ نمونه‌برداری: {sr}, شکل: {audio.shape}")
   ```

### تنسوربورد چیزی نشان نمی‌دهد؟

1. **آدرس صحیح را بررسی کنید:**
   ```bash
   # پیش‌فرض
   tensorboard --logdir=./checkpointsmain/tensorboard
   
   # سفارشی
   tensorboard --logdir=./logs
   ```

2. **صبر کنید** - اولین لاگ در قدم ۱۰۰ اتفاق می‌افتد، نه قدم ۰

3. **رفرش کنید** - در مرورگر دکمه رفرش را بزنید

## نکات مهم

1. ✅ **فرمت PNG**: برای تصاویر بهترین کیفیت
2. ✅ **فرمت WAV**: برای صوت ساده‌ترین است
3. ✅ **نام فایل‌ها**: نام‌های واضح بگذارید
4. ✅ **حجم فایل**: فایل‌های سنگین کند هستند
5. ✅ **آپدیت**: می‌توانید در حین آموزش فایل اضافه/حذف کنید

## خلاصه تغییرات فنی

### فایل‌های تغییر یافته:
1. `myxtts/training/trainer.py` - پیاده‌سازی اصلی
2. `myxtts/config/config.py` - تنظیمات
3. `train_main.py` - رابط خط فرمان
4. `README.md` - بروزرسانی مستندات
5. `.gitignore` - نادیده گرفتن فایل‌های sample

### فایل‌های جدید:
1. `TENSORBOARD_SAMPLES_GUIDE.md` - راهنمای انگلیسی
2. `TENSORBOARD_SAMPLES_GUIDE_FA.md` - راهنمای فارسی
3. `training_samples/README.md` - راهنمای پوشه
4. `examples/tensorboard_samples_example.sh` - اسکریپت مثال
5. `test_tensorboard_logging.py` - اسکریپت تست
6. `IMPLEMENTATION_SUMMARY.md` - خلاصه پیاده‌سازی

## نتیجه

✅ **مشکل حل شد!**

حالا می‌توانید:
1. تصاویر خود را در `training_samples` بگذارید
2. صداهای خود را در `training_samples` بگذارید
3. آموزش را شروع کنید
4. همه چیز را در تنسوربورد ببینید

آدرس تنسوربورد هم قابل تنظیم است - می‌توانید از `logs` یا هر آدرس دیگری استفاده کنید.

## پشتیبانی

اگر سوالی دارید یا مشکلی پیش آمد:
1. راهنماهای بالا را بخوانید
2. بخش عیب‌یابی را چک کنید
3. در مخزن پروژه issue باز کنید

---

**موفق باشید! 🎉**
