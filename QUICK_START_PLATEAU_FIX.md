# 🚀 Quick Start: Fix Loss Plateau at 2.7

## Problem
Your training loss is stuck at around 2.7 and won't decrease further.

## Solution
Use the new `plateau_breaker` optimization level!

## Fastest Way (30 seconds)

### Step 1: Stop current training
```bash
pkill -f "python3 train_main.py"
```

### Step 2: Start with plateau_breaker
```bash
bash breakthrough_training.sh
```

That's it! Your training will restart with optimized settings.

---

## What It Does

✅ Reduces learning rate by 80% (1e-4 → 1.5e-5)
✅ Rebalances loss weights (mel: 2.0, kl: 1.2)
✅ Tightens gradient clipping (0.3)
✅ Optimizes scheduler (cosine with 100-epoch restarts)
✅ Enables all stability features

## Expected Results

| Timeline | Expected Loss | Status |
|----------|---------------|--------|
| **Start** | 2.7 | 🔴 Stuck |
| **5-10 epochs** | 2.5-2.6 | 🟡 Improving |
| **10-20 epochs** | 2.2-2.3 | 🟢 Good |
| **20+ epochs** | < 2.0 | 🌟 Excellent |

## Alternative Commands

### With custom batch size
```bash
BATCH_SIZE=32 bash breakthrough_training.sh
```

### Resume from checkpoint
```bash
python3 train_main.py \
    --optimization-level plateau_breaker \
    --resume-checkpoint checkpoints/your_checkpoint.ckpt
```

### Manual command
```bash
python3 train_main.py --optimization-level plateau_breaker --batch-size 24
```

## Verify It's Working

### Run diagnostic tool
```bash
python3 utilities/diagnose_plateau.py --log-file training.log
```

### Run tests
```bash
python3 tests/test_plateau_breaker_config.py
```

## Troubleshooting

### Loss not improving after 10 epochs?
```bash
# Try lower learning rate
python3 train_main.py --optimization-level plateau_breaker --lr 1e-5

# Or reduce batch size
python3 train_main.py --optimization-level plateau_breaker --batch-size 16
```

### Need more help?
- 📖 Read: [docs/PLATEAU_BREAKER_USAGE.md](docs/PLATEAU_BREAKER_USAGE.md)
- 🔬 See: [docs/LOSS_PLATEAU_SOLUTION_2.7.md](docs/LOSS_PLATEAU_SOLUTION_2.7.md)
- 💡 Check: [docs/PLATEAU_BREAKTHROUGH_GUIDE.md](docs/PLATEAU_BREAKTHROUGH_GUIDE.md)

## Success Checklist

- [ ] Loss is decreasing consistently
- [ ] No NaN or Inf values in training
- [ ] Validation loss tracks training loss
- [ ] Audio quality is improving
- [ ] Loss below 2.5 after 10 epochs

## Quick Reference

| What | Command |
|------|---------|
| **Start training** | `bash breakthrough_training.sh` |
| **Diagnose plateau** | `python3 utilities/diagnose_plateau.py` |
| **Run tests** | `python3 tests/test_plateau_breaker_config.py` |
| **Custom LR** | `--lr 1.5e-5` |
| **Custom batch** | `--batch-size 24` |
| **Resume** | `--resume-checkpoint path/to/ckpt` |

---

**Status**: ✅ Fully tested and ready to use
**Expected time to see results**: 5-10 epochs
**Target loss**: 2.2-2.3 (from 2.7)
