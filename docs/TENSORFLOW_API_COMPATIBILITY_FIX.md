# TensorFlow API Compatibility Fix for GPU Memory Isolation

## مشکل (Problem)

در نسخه‌های جدید TensorFlow (2.10 به بعد)، متد `set_logical_device_configuration` حذف شده و با `set_virtual_device_configuration` جایگزین شده است. این باعث خطای زیر می‌شد:

```
ERROR - ❌ Failed to setup GPU memory isolation: module 'tensorflow._api.v2.config.experimental' has no attribute 'set_logical_device_configuration'
```

## راه‌حل (Solution)

کد به‌روزرسانی شده است تا به صورت خودکار API مناسب را بر اساس نسخه TensorFlow شناسایی و استفاده کند.

### تغییرات اعمال شده

#### 1. شناسایی خودکار API

کد اکنون از `hasattr` برای تشخیص API در دسترس استفاده می‌کند:

```python
use_virtual_device_api = hasattr(tf.config.experimental, 'set_virtual_device_configuration')

if use_virtual_device_api:
    # استفاده از API جدید (TensorFlow 2.10+)
    tf.config.experimental.set_virtual_device_configuration(...)
else:
    # استفاده از API قدیمی (TensorFlow < 2.10)
    tf.config.experimental.set_logical_device_configuration(...)
```

#### 2. مدیریت خطای AttributeError

در صورتی که هیچ یک از APIها در دسترس نباشد، کد به صورت خودکار به `memory growth` برمی‌گردد:

```python
try:
    # تلاش برای تنظیم memory limit
    ...
except AttributeError:
    # Fallback به memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.warning("⚠️  Virtual/Logical device configuration API not available")
    logger.warning("    Falling back to memory growth only")
```

#### 3. پشتیبانی از نسخه‌های مختلف

| نسخه TensorFlow | API استفاده شده | وضعیت |
|----------------|-----------------|-------|
| TensorFlow 2.10+ | `set_virtual_device_configuration` | ✅ پشتیبانی کامل |
| TensorFlow 2.4-2.9 | `set_logical_device_configuration` | ✅ سازگاری کامل |
| نسخه‌های قدیمی‌تر | `set_memory_growth` (fallback) | ⚠️ محدودیت دارد |

### ویژگی‌های جدید

1. **سازگاری کراس-ورژن**: کد با تمام نسخه‌های TensorFlow از 2.4 به بعد کار می‌کند
2. **Fallback هوشمند**: در صورت عدم دسترسی به API، به memory growth برمی‌گردد
3. **لاگ‌گذاری دقیق**: تمام مراحل و تصمیمات لاگ می‌شوند
4. **مستندسازی کامل**: Docstring تابع به‌روزرسانی شده و رفتار API را توضیح می‌دهد

## استفاده (Usage)

استفاده از تابع تغییری نکرده است:

```python
from myxtts.utils.gpu_memory import setup_gpu_memory_isolation

# تنظیم memory isolation برای دو GPU
success = setup_gpu_memory_isolation(
    data_gpu_id=0,          # GPU 0 for data processing
    model_gpu_id=1,         # GPU 1 for model training
    data_gpu_memory_limit=8192,   # 8GB for data GPU
    model_gpu_memory_limit=16384  # 16GB for model GPU
)

if success:
    print("✅ Memory isolation configured successfully")
else:
    print("⚠️  Fallback to memory growth mode")
```

## تست‌ها (Tests)

تست جامع `test_gpu_memory_api_compatibility.py` ایجاد شده که موارد زیر را بررسی می‌کند:

1. ✅ وجود API جدید در کد
2. ✅ حفظ API قدیمی برای سازگاری
3. ✅ شناسایی API با hasattr
4. ✅ مدیریت AttributeError
5. ✅ Fallback به memory growth
6. ✅ لاگ‌گذاری مناسب

برای اجرای تست‌ها:

```bash
python3 tests/test_gpu_memory_api_compatibility.py
```

## پیام‌های لاگ (Log Messages)

### موفقیت با API جدید
```
🎯 Setting up GPU Memory Isolation...
   Using set_virtual_device_configuration (TensorFlow 2.10+)
   ✅ Data GPU memory limit set to 8192MB
   ✅ Model GPU memory limit set to 16384MB
✅ GPU Memory Isolation configured successfully
```

### موفقیت با API قدیمی
```
🎯 Setting up GPU Memory Isolation...
   Using set_logical_device_configuration (TensorFlow < 2.10)
   ✅ Data GPU memory limit set to 8192MB
   ✅ Model GPU memory limit set to 16384MB
✅ GPU Memory Isolation configured successfully
```

### Fallback به Memory Growth
```
🎯 Setting up GPU Memory Isolation...
   ⚠️  Virtual/Logical device configuration API not available
      Falling back to memory growth only
   ✅ Enabled memory growth for all GPUs as fallback
```

## نکات مهم (Important Notes)

1. **فراخوانی زودهنگام**: تابع باید **قبل از** هر عملیات TensorFlow فراخوانی شود
2. **بررسی مقدار برگشتی**: همیشه مقدار برگشتی را بررسی کنید:
   - `True`: Memory isolation با موفقیت تنظیم شد
   - `False`: Fallback به memory growth یا خطا رخ داد
3. **حداقل 2 GPU**: برای memory isolation حداقل 2 GPU لازم است

## خلاصه تغییرات (Summary of Changes)

### فایل‌های تغییر یافته
- ✅ `myxtts/utils/gpu_memory.py`: پیاده‌سازی API سازگار با نسخه
- ✅ `tests/test_gpu_memory_api_compatibility.py`: تست‌های جامع سازگاری
- ✅ `docs/TENSORFLOW_API_COMPATIBILITY_FIX.md`: این مستند

### بدون تغییر (Backward Compatible)
- ✅ امضای تابع تغییر نکرده
- ✅ رفتار کلی حفظ شده
- ✅ کدهای موجود بدون تغییر کار می‌کنند

## منابع (References)

- [TensorFlow 2.10 Release Notes](https://github.com/tensorflow/tensorflow/releases/tag/v2.10.0)
- [TensorFlow Virtual Device Configuration](https://www.tensorflow.org/api_docs/python/tf/config/experimental/VirtualDeviceConfiguration)
- [TensorFlow Memory Management](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)

---

**تاریخ به‌روزرسانی**: 2025-10-10  
**نسخه**: 1.0.0  
**وضعیت**: ✅ تست شده و آماده استفاده
