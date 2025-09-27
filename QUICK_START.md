# 🚀 MyXTTS Quick Start Guide

## ⚡ 30-Second Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start training immediately
python3 train_main.py --model-size tiny --batch-size 4 --epochs 10

# 3. For production training
python3 train_main.py --optimization-level enhanced --enable-gpu-stabilizer
```

## 🎯 Common Use Cases

### 🔸 **Loss is stuck around 2.5?**
```bash
bash scripts/breakthrough_training.sh
# OR
python3 train_main.py --optimization-level plateau_breaker --batch-size 24
```

### 🔸 **Want maximum GPU utilization?**
```bash
python3 train_main.py --enable-gpu-stabilizer --optimization-level enhanced
```

### 🔸 **Need voice cloning?**
```bash
python3 train_main.py --enable-gst --gst-num-style-tokens 12 --model-size normal
```

### 🔸 **Testing with small resources?**
```bash
python3 train_main.py --model-size tiny --batch-size 4 --disable-gpu-stabilizer
```

## 📋 Key Scripts

| Script | Purpose |
|--------|---------|
| `train_main.py` | Main training (start here) |
| `scripts/train_control.sh` | Easy training control |
| `scripts/breakthrough_training.sh` | Fix stuck loss |
| `inference_main.py` | Generate voice |

## ⚙️ Quick Settings

| Setting | Values | Purpose |
|---------|--------|---------|
| `--model-size` | tiny, small, normal, big | Model capacity |
| `--optimization-level` | basic, enhanced, experimental, plateau_breaker | Training strategy |
| `--batch-size` | 4, 8, 16, 32, 48 | Memory vs speed |
| `--enable-gpu-stabilizer` | flag | GPU optimization |

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Use `--model-size tiny --batch-size 4` |
| Low GPU usage | Add `--enable-gpu-stabilizer` |
| Loss not improving | Try `--optimization-level plateau_breaker` |
| Training too slow | Use `--optimization-level enhanced` |

## 📚 More Info

- **Full documentation**: `README.md`
- **All configurations**: `configs/` directory  
- **Examples**: `examples/` directory
- **Troubleshooting guides**: `docs/` directory

---
**🎯 Ready? Just run: `python3 train_main.py` to start!**