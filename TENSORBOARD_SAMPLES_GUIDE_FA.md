# راهنمای فارسی - لاگ کردن تصاویر و صداها در تنسوربورد

## خلاصه
با این قابلیت جدید، می‌توانید تصاویر و فایل‌های صوتی موجود در پوشه `training_samples` را در تنسوربورد ببینید. همچنین صداهایی که مدل تولید می‌کند هم در تنسوربورد نمایش داده می‌شوند.

## مراحل استفاده

### ۱. ایجاد پوشه training_samples

```bash
mkdir training_samples
```

### ۲. کپی کردن فایل‌های خود

```bash
# کپی تصاویر (مانند اسپکتروگرام‌ها)
cp your_images/*.png training_samples/

# کپی فایل‌های صوتی
cp your_audio/*.wav training_samples/
```

### ۳. شروع آموزش

```bash
python train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval
```

### ۴. باز کردن تنسوربورد

```bash
# اگر آدرس لاگ شما logs است:
tensorboard --logdir=logs

# اگر آدرس پیش‌فرض را استفاده می‌کنید:
tensorboard --logdir=./checkpointsmain/tensorboard
```

سپس در مرورگر خود به آدرس زیر بروید:
```
http://localhost:6006
```

## مشاهده محتوا در تنسوربورد

### تصاویر
1. به تب **IMAGES** بروید
2. موارد زیر `training_samples/image/` را پیدا کنید
3. همچنین می‌توانید اسپکتروگرام‌های تولید شده را در `train/spectrogram_*` ببینید

### صداها
1. به تب **AUDIO** بروید
2. موارد زیر `training_samples/audio/` را پیدا کنید
3. صداهای تولید شده توسط مدل در `text2audio_eval/` هستند

## تنظیمات اضافی

### تغییر آدرس پوشه samples
```bash
python train_main.py \
    --training-samples-dir ./my_samples \
    --train-data ../dataset/dataset_train
```

### تغییر فاصله لاگ کردن
به طور پیش‌فرض، هر ۱۰۰ قدم یک بار لاگ می‌شود. برای تغییر:

```bash
# لاگ هر ۵۰ قدم
python train_main.py \
    --training-samples-log-interval 50 \
    --train-data ../dataset/dataset_train

# لاگ هر ۲۰۰ قدم
python train_main.py \
    --training-samples-log-interval 200 \
    --train-data ../dataset/dataset_train
```

### غیرفعال کردن این قابلیت
```bash
python train_main.py \
    --training-samples-log-interval 0 \
    --train-data ../dataset/dataset_train
```

### تغییر آدرس لاگ‌های تنسوربورد
```bash
python train_main.py \
    --tensorboard-log-dir ./logs \
    --train-data ../dataset/dataset_train
```

## فرمت‌های پشتیبانی شده

### تصاویر
- PNG (پیشنهادی)
- JPG/JPEG
- BMP
- GIF

### صوت
- WAV (پیشنهادی - نیاز به کتابخانه اضافی ندارد)
- MP3, FLAC, OGG (نیاز به کتابخانه soundfile)

برای نصب soundfile:
```bash
pip install soundfile
```

## مثال کامل

```bash
# ۱. ایجاد پوشه
mkdir training_samples

# ۲. کپی فایل‌ها
cp my_spectrograms/*.png training_samples/
cp my_audio/*.wav training_samples/

# ۳. شروع آموزش
python train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --training-samples-dir training_samples \
    --training-samples-log-interval 100 \
    --tensorboard-log-dir logs \
    --batch-size 16 \
    --epochs 500

# ۴. باز کردن تنسوربورد (در ترمینال دیگر)
tensorboard --logdir=logs
```

سپس در مرورگر به `http://localhost:6006` بروید.

## عیب‌یابی

### تصاویر نمایش داده نمی‌شوند

1. مطمئن شوید پوشه `training_samples` وجود دارد:
   ```bash
   ls -la training_samples/
   ```

2. فرمت فایل‌ها را چک کنید (باید یکی از png, jpg, jpeg, bmp, gif باشد)

3. فاصله لاگ را بررسی کنید - لاگ اول در قدم ۱۰۰ است نه قدم ۰

### صداها نمایش داده نمی‌شوند

1. برای فایل‌های WAV نیاز به کتابخانه اضافی نیست

2. برای فرمت‌های دیگر (MP3, FLAC):
   ```bash
   pip install soundfile
   ```

3. یکپارچگی فایل صوتی را بررسی کنید

### تنسوربورد پوشه اشتباه را نشان می‌دهد

آدرس صحیح را به tensorboard بدهید:

```bash
# اگر آدرس پیش‌فرض
tensorboard --logdir=./checkpointsmain/tensorboard

# اگر آدرس سفارشی (مثلا logs)
tensorboard --logdir=./logs
```

## اسکریپت مثال

برای ایجاد فایل‌های نمونه و آزمایش سیستم:

```bash
bash examples/tensorboard_samples_example.sh
```

این اسکریپت:
- پوشه `training_samples` را می‌سازد
- تصاویر و صداهای نمونه ایجاد می‌کند
- دستورات لازم را نشان می‌دهد

## سوالات متداول

**س: چند بار در طول آموزش لاگ می‌شود؟**
ج: به طور پیش‌فرض هر ۱۰۰ قدم یک بار. می‌توانید با `--training-samples-log-interval` تغییر دهید.

**س: آیا فایل‌های اصلی تغییر می‌کنند؟**
ج: خیر، فقط خوانده می‌شوند و به تنسوربورد فرستاده می‌شوند.

**س: آیا بر سرعت آموزش تاثیر می‌گذارد؟**
ج: خیر، چون فقط در فواصل مشخص اجرا می‌شود و غیر مسدودکننده است.

**س: چند فایل می‌توانم اضافه کنم؟**
ج: بدون محدودیت، اما تعداد زیاد فایل ممکن است لاگ کردن را کمی کند کند.

## لینک‌های مفید

- راهنمای کامل انگلیسی: [TENSORBOARD_SAMPLES_GUIDE.md](TENSORBOARD_SAMPLES_GUIDE.md)
- مستندات تنسوربورد: https://www.tensorflow.org/tensorboard
- مخزن پروژه: https://github.com/masoodafar-web/MyXTTSModel

---

**توجه:** این قابلیت در نسخه جدید اضافه شده و با تمام تنظیمات موجود سازگار است.
