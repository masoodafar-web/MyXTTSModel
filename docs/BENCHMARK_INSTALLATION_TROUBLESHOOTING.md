# 🔧 MyXTTS Benchmark Installation & Troubleshooting

## 🚀 Quick Setup

### **روش 1: اسکریپت خودکار (پیشنهادی)**
```bash
# نصب همه dependencies
bash scripts/install_dependencies.sh
```

### **روش 2: نصب دستی**
```bash
# نصب requirements اصلی
pip3 install -r requirements.txt

# نصب GPU monitoring tools
pip3 install GPUtil seaborn plotly

# یا اگر GPUtil کار نکرد
pip3 install nvidia-ml-py3
```

## 🧪 تست نصب

### **تست سریع**
```bash
# تست benchmark scripts
python3 scripts/quick_param_test.py --help
python3 utilities/benchmark_hyperparameters.py --help

# تست GPU monitoring
python3 -c "import GPUtil; print(f'GPUs: {len(GPUtil.getGPUs())}')"
```

### **تست کامل**
```bash
# اجرای اسکریپت تست
bash scripts/install_dependencies.sh
```

## ❌ مشکلات رایج و راه‌حل

### **🔴 `ModuleNotFoundError: No module named 'GPUtil'`**

**راه‌حل 1:**
```bash
pip3 install GPUtil
```

**راه‌حل 2 (اگر GPUtil نصب نشد):**
```bash
pip3 install nvidia-ml-py3
```

**راه‌حل 3 (fallback موجود در کد):**
- کد خودکار از `nvidia-smi` استفاده می‌کنه
- نیازی به نصب اضافی نیست

### **🔴 `ModuleNotFoundError: No module named 'matplotlib'`**
```bash
pip3 install matplotlib seaborn plotly
```

### **🔴 `ModuleNotFoundError: No module named 'pandas'`**
```bash
pip3 install pandas numpy
```

### **🔴 GPU Detection نمی‌کنه**

**چک کنید:**
```bash
# nvidia-smi کار میکنه؟
nvidia-smi

# TensorFlow GPU میبینه؟
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**راه‌حل:**
- Driver های NVIDIA نصب باشند
- CUDA toolkit نصب باشه
- `nvidia-smi` در PATH باشه

### **🔴 Permission Denied برای اسکریپت‌ها**
```bash
chmod +x scripts/*.sh
chmod +x scripts/*.py
chmod +x utilities/*.py
```

### **🔴 Import Error برای benchmark modules**
```bash
# اضافه کردن path
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/utilities:$(pwd)/scripts"

# یا اجرا از root directory
cd /path/to/MyXTTSModel
python3 scripts/quick_param_test.py
```

### **🔴 Out of Memory در benchmark**
```bash
# استفاده از memory-safe scenario
python3 scripts/quick_param_test.py --memory-safe

# یا کم کردن timeout
python3 scripts/quick_param_test.py --timeout 60
```

## 🔍 تشخیص مشکلات

### **Environment Check**
```bash
# چک Python version
python3 --version  # باید 3.8+ باشه

# چک pip packages
pip3 list | grep -E "(GPUtil|pandas|matplotlib|tensorflow)"

# چک GPU
nvidia-smi
```

### **Dependencies Check**
```bash
python3 -c "
import sys
modules = ['tensorflow', 'numpy', 'pandas', 'matplotlib', 'GPUtil', 'psutil']
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        print(f'❌ {module}')
"
```

### **GPU Check**
```bash
python3 -c "
# Test TensorFlow GPU
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU devices: {len(tf.config.list_physical_devices("GPU"))}')

# Test GPU utilities
try:
    import GPUtil
    gpus = GPUtil.getGPUs()
    print(f'GPUtil found {len(gpus)} GPUs')
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu.load*100:.1f}% util, {gpu.memoryTotal:.1f}GB total')
except Exception as e:
    print(f'GPUtil error: {e}')
    
    # Try fallback
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print(f'nvidia-ml-py found {count} GPUs')
    except Exception as e2:
        print(f'nvidia-ml-py error: {e2}')
"
```

## 🛠️ Manual Fix for Common Issues

### **Fix 1: GPUtil Alternative**
اگر GPUtil نصب نمیشه، این کد رو در `~/.bashrc` اضافه کنید:
```bash
export GPUTIL_FALLBACK=1
```

### **Fix 2: PATH Issues**
```bash
# Add to ~/.bashrc
export PYTHONPATH="${PYTHONPATH}:~/xTTS/MyXTTSModel"
export PATH="${PATH}:~/xTTS/MyXTTSModel/scripts"
```

### **Fix 3: Virtual Environment**
```bash
# ایجاد virtual environment
python3 -m venv myxtts_env
source myxtts_env/bin/activate
pip3 install -r requirements.txt
pip3 install GPUtil seaborn plotly
```

## ✅ تست نهایی

بعد از حل مشکلات، این‌ها رو تست کنید:

### **1. Basic Test**
```bash
python3 scripts/quick_param_test.py --scenario basic_test --timeout 60
```

### **2. GPU Test**  
```bash
python3 scripts/quick_param_test.py --gpu-optimize --timeout 120
```

### **3. Full Menu Test**
```bash
bash scripts/benchmark_params.sh
```

## 🎯 Expected Output

اگر همه چی درست باشه، باید این‌ها رو ببینید:

```
🔬 MyXTTS Quick Parameter Test
Scenario: basic_test
Timeout per test: 300 seconds

🚀 Running scenario: basic_test (3 tests)
============================================================

[1/3] Running: Tiny Basic
🔬 Testing: Tiny Basic
   Command: python3 train_main.py --model-size tiny --optimization-level basic --batch-size 4 --epochs 3
   ✅ COMPLETED in 45.2s
   📊 Final Loss: 2.856
   
...

📋 RESULTS SUMMARY
============================================================
🏆 BEST RESULTS (by final loss):
1. Tiny Enhanced: Loss=2.234, Time=67.3s
2. Small Enhanced: Loss=2.445, Time=89.1s
3. Tiny Basic: Loss=2.856, Time=45.2s
```

## 📞 کمک بیشتر

اگر مشکل حل نشد:

1. **Log کامل رو بفرستید** از اجرای benchmark
2. **System info** بدید:
   ```bash
   nvidia-smi
   python3 --version
   pip3 list | head -20
   ```
3. **Error message کامل** رو کپی کنید

---

**🎯 Goal: همه benchmark tools بدون مشکل کار کنند!**