# 🔬 MyXTTS Hyperparameter Benchmarking Guide

## 🎯 Overview

این راهنما به شما کمک می‌کند بهترین hyperparameter ها رو برای مدل MyXTTS خودتون پیدا کنید.

## 🚀 Quick Start

### ⚡ 5-Minute Quick Test
```bash
# روش ساده
bash scripts/benchmark_params.sh

# یا مستقیم
python3 scripts/quick_param_test.py
```

### 🎯 مشکل خاص دارید؟

```bash
# Loss در 2.5 گیر کرده؟
python3 scripts/quick_param_test.py --plateau-fix

# GPU کم استفاده میشه؟  
python3 scripts/quick_param_test.py --gpu-optimize

# Memory کم دارید؟
python3 scripts/quick_param_test.py --memory-safe

# Learning rate بهینه میخواید؟
python3 scripts/quick_param_test.py --learning-rate-sweep
```

## 🔧 Available Tools

### 1. **Quick Parameter Test** (`scripts/quick_param_test.py`)
- **مدت زمان**: 5-20 دقیقه
- **هدف**: تست سریع سناریوهای رایج
- **خروجی**: JSON results + recommendations

### 2. **Full Hyperparameter Benchmark** (`utilities/benchmark_hyperparameters.py`)
- **مدت زمان**: 2-6 ساعت
- **هدف**: جستجوی جامع hyperparameter
- **خروجی**: Comprehensive report + visualizations

### 3. **Interactive Benchmark Script** (`scripts/benchmark_params.sh`)
- **مدت زمان**: متغیر
- **هدف**: راهنمای تعاملی برای انتخاب benchmark
- **خروجی**: منوی انتخاب + اجرای خودکار

## 📋 Benchmark Scenarios

### 🔥 **Basic Test** (5 دقیقه)
```bash
python3 scripts/quick_param_test.py --scenario basic_test
```
**تست میکنه**:
- Model sizes: tiny, small
- Optimization levels: basic, enhanced
- Batch sizes: 4, 8, 16

### 🎯 **Plateau Fix** (10 دقیقه)
```bash
python3 scripts/quick_param_test.py --plateau-fix
```
**برای زمانی که**:
- Loss حدود 2.5 گیر کرده
- بهبودی نمیبینید
- نیاز به breakthrough دارید

### 🎮 **GPU Optimization** (15 دقیقه)
```bash
python3 scripts/quick_param_test.py --gpu-optimize
```
**برای زمانی که**:
- GPU utilization پایینه
- میخواید سرعت training افزایش پیدا کنه
- GPU Stabilizer تست کنید

### 💾 **Memory Safe** (8 دقیقه)
```bash
python3 scripts/quick_param_test.py --memory-safe
```
**برای زمانی که**:
- VRAM محدود دارید
- Out of Memory error میگیرید
- کارت کمتر از 8GB دارید

### 📈 **Learning Rate Sweep** (20 دقیقه)
```bash
python3 scripts/quick_param_test.py --learning-rate-sweep
```
**تست میکنه**:
- 5e-6, 1e-5, 2e-5, 5e-5, 1e-4
- با model size و batch size ثابت
- برای پیدا کردن optimal learning rate

## 🔬 Advanced Benchmarking

### **Full Comprehensive Benchmark**
```bash
# Warning: 2-4 ساعت طول میکشه!
python3 utilities/benchmark_hyperparameters.py --full-sweep
```

### **Quick 15-Minute Advanced Test**
```bash
python3 utilities/benchmark_hyperparameters.py --quick-test
```

### **Custom Configuration**
```bash
python3 utilities/benchmark_hyperparameters.py --config configs/benchmark_config.yaml
```

## 📊 Understanding Results

### **Metrics Explained:**

1. **Final Loss**: پایین‌تر = بهتر
   - `< 2.0`: عالی
   - `2.0-2.5`: خوب  
   - `2.5-3.0`: قابل قبول
   - `> 3.0`: ضعیف

2. **Training Speed**: بالاتر = بهتر
   - `> 30 samples/sec`: عالی
   - `20-30 samples/sec`: خوب
   - `10-20 samples/sec`: متوسط
   - `< 10 samples/sec`: کند

3. **GPU Utilization**: بالاتر = بهتر
   - `> 80%`: عالی
   - `60-80%`: خوب
   - `40-60%`: متوسط
   - `< 40%`: ضعیف

4. **Overall Score**: ترکیب همه metrics
   - `> 0.8`: عالی
   - `0.6-0.8`: خوب
   - `0.4-0.6`: متوسط
   - `< 0.4`: ضعیف

### **Result Files:**
- `param_test_results_*.json`: Raw results
- `benchmark_results_*/`: Full benchmark output
- `benchmark_plots.png`: Visualizations
- `benchmark_report.md`: Comprehensive analysis

## 🎯 Common Use Cases

### **مشکل: Loss stuck at 2.5**
```bash
# تست سریع
python3 scripts/quick_param_test.py --plateau-fix

# نتیجه احتمالی
python3 train_main.py --optimization-level plateau_breaker --batch-size 24 --lr 1.5e-5
```

### **مشکل: GPU Usage پایین**
```bash
# تست GPU optimization
python3 scripts/quick_param_test.py --gpu-optimize

# نتیجه احتمالی  
python3 train_main.py --enable-gpu-stabilizer --optimization-level enhanced --batch-size 32
```

### **مشکل: Out of Memory**
```bash
# تست memory safe
python3 scripts/quick_param_test.py --memory-safe

# نتیجه احتمالی
python3 train_main.py --model-size tiny --batch-size 4 --optimization-level basic
```

### **هدف: Maximum Quality**
```bash
# تست جامع
python3 utilities/benchmark_hyperparameters.py --quick-test

# نتیجه احتمالی
python3 train_main.py --model-size normal --optimization-level enhanced --enable-gpu-stabilizer
```

## 📝 Step-by-Step Workflow

### **1. مشخص کردن هدف**
- Quality بالا میخواید؟
- Speed مهمه؟  
- Memory محدودید؟
- مشکل خاصی دارید؟

### **2. انتخاب benchmark مناسب**
```bash
# برای شروع همیشه این رو اجرا کنید
bash scripts/benchmark_params.sh
```

### **3. تحلیل نتایج**
- بهترین configuration رو پیدا کنید
- Metrics رو بررسی کنید
- Trade-off های مختلف رو در نظر بگیرید

### **4. اعمال نتایج**
```bash
# استفاده از recommended parameters
python3 train_main.py [recommended parameters]
```

### **5. Fine-tuning**
- اگر نتیجه خوب بود، parameters مشابه تست کنید
- اگر نتیجه بد بود، scenarios دیگه امتحان کنید

## 🛠️ Troubleshooting

### **Benchmark fails با Memory Error**
```bash
# کم کردن scope benchmark
python3 scripts/quick_param_test.py --memory-safe
```

### **Results غیر منطقی**
```bash
# چک کردن hardware
nvidia-smi
python3 utilities/quick_memory_test.py
```

### **خیلی کند اجرا میشه**
```bash
# استفاده از quick scenarios
python3 scripts/quick_param_test.py --scenario basic_test
```

## 📈 Pro Tips

1. **همیشه با quick test شروع کنید** قبل full benchmark
2. **نتایج رو save کنید** برای comparison بعدی
3. **hardware constraints رو در نظر بگیرید**
4. **multiple runs** برای confidence بیشتر
5. **specific problems** با targeted scenarios حل کنید

## 🎯 Expected Results

### **RTX 4090 (24GB) - Typical Best Results:**
```yaml
Model Size: normal
Optimization Level: enhanced  
Learning Rate: 2e-5 to 5e-5
Batch Size: 32-48
GPU Stabilizer: enabled
Expected Loss: 2.0-2.3
Expected Speed: 25-35 sps
Expected GPU Usage: 85-95%
```

### **RTX 3080 (10GB) - Typical Best Results:**
```yaml
Model Size: small
Optimization Level: enhanced
Learning Rate: 2e-5 to 3e-5  
Batch Size: 16-24
GPU Stabilizer: enabled
Expected Loss: 2.2-2.5
Expected Speed: 20-30 sps
Expected GPU Usage: 80-90%
```

---

## 🚀 Ready to Benchmark?

```bash
# شروع سریع
bash scripts/benchmark_params.sh

# یا مستقیم
python3 scripts/quick_param_test.py
```

**🎯 پیدا کردن بهترین parameters برای setup خودتون!**