# Plateau Breaker Usage Guide

## Overview
The `plateau_breaker` optimization level is specifically designed to break through loss plateaus that occur during training, particularly when loss gets stuck around 2.5-2.7.

## When to Use

Use `plateau_breaker` when:
- ✅ Loss has plateaued and not decreasing for multiple epochs
- ✅ Loss is stuck around 2.5-2.7 (or any similar plateau)
- ✅ Training appears stable but not improving
- ✅ You've already tried basic or enhanced optimization levels

Do NOT use when:
- ❌ Loss is still decreasing steadily
- ❌ Training just started (use `basic` or `enhanced` first)
- ❌ Loss is NaN or exploding (use `basic` instead)

## Quick Start

### Method 1: Using the Training Script (Recommended)

```bash
bash breakthrough_training.sh
```

This is the simplest way to start training with plateau_breaker settings. The script will:
- Set optimization level to `plateau_breaker`
- Use appropriate batch size (default: 24)
- Configure all necessary parameters automatically

You can customize the script with environment variables:
```bash
BATCH_SIZE=32 EPOCHS=500 bash breakthrough_training.sh
```

### Method 2: Direct Command Line

```bash
python3 train_main.py --optimization-level plateau_breaker --batch-size 24
```

## What It Does

The `plateau_breaker` optimization level applies these specific changes:

### 1. Learning Rate Reduction (80%)
- **Before**: 1e-4 (default) or 8e-5 (enhanced)
- **After**: 1.5e-5
- **Why**: Smaller learning rate allows finer adjustments to weights, helping escape plateaus

### 2. Loss Weight Rebalancing
- **Mel loss weight**: 2.0 (down from 2.5)
- **KL loss weight**: 1.2 (down from 1.8)
- **Why**: Better balance between different loss components prevents one from dominating

### 3. Tighter Gradient Clipping
- **Before**: 0.5-0.8
- **After**: 0.3
- **Why**: More controlled gradient updates prevent overshooting minima

### 4. Optimized Scheduler
- **Type**: Cosine annealing with restarts
- **Restart period**: Every ~100 epochs
- **Why**: Periodic learning rate cycles help explore the loss landscape

### 5. Stability Features Enabled
- ✅ Adaptive loss weights
- ✅ Label smoothing
- ✅ Huber loss
- **Why**: These features improve training stability and convergence

## Expected Results

### Timeline
- **First 5 epochs**: May see small improvements or stability
- **5-10 epochs**: Should see noticeable loss reduction
- **10+ epochs**: Loss should continue decreasing steadily

### Target Loss Values
Starting from **2.7**:
- **After 10 epochs**: Should reach **2.2-2.3**
- **After 20 epochs**: Should reach **2.0-2.1**
- **Long term**: Should continue improving below 2.0

### Quality Indicators
- ✅ Loss decreasing consistently
- ✅ Better balance between mel_loss and stop_loss
- ✅ Reduced training oscillations
- ✅ Improved validation loss

## Monitoring Training

### Key Metrics to Watch

1. **Total Loss**: Should decrease steadily
2. **Mel Loss**: Should decrease but not dominate
3. **Stop Loss**: Should remain stable and balanced
4. **Learning Rate**: Should follow cosine schedule with restarts
5. **Gradient Norm**: Should stay below clipping threshold (0.3)

### Warning Signs

⚠️ **Re-evaluate if you see**:
- Loss increasing consistently
- NaN or Inf values
- Validation loss diverging from training loss
- No improvement after 20 epochs

## Comparison with Other Levels

| Feature | Basic | Enhanced | Plateau Breaker |
|---------|-------|----------|----------------|
| Learning Rate | 1e-5 | 1e-4 | 1.5e-5 |
| Mel Loss Weight | 1.0 | 2.5 | 2.0 |
| Gradient Clip | 0.5 | 0.5 | 0.3 |
| Scheduler | Noam | Cosine+Restarts | Cosine+Restarts |
| Use Case | Initial training | Normal training | Breaking plateaus |
| Stability | Highest | Medium | High |
| Speed | Slowest | Fastest | Medium |

## Troubleshooting

### Loss not improving after 10 epochs

Try:
1. Reduce batch size: `--batch-size 16`
2. Further reduce learning rate: `--lr 1e-5`
3. Check data quality and preprocessing
4. Review validation metrics for overfitting

### Training too slow

This is expected with plateau_breaker due to lower learning rate. If you need faster training:
1. Use `enhanced` level instead if loss is still decreasing
2. Increase batch size if GPU memory allows
3. Reduce epochs if hitting early stopping

### Loss oscillating

This might be normal during restarts. However, if oscillations are too large:
1. Further reduce learning rate: `--lr 1e-5`
2. Increase gradient clipping: modify config.py (e.g., 0.4)
3. Review batch size (may need to reduce)

## Advanced Configuration

### Custom Parameters

You can override specific parameters:

```bash
python3 train_main.py \
    --optimization-level plateau_breaker \
    --lr 1.2e-5 \
    --batch-size 32 \
    --gradient-clip 0.25
```

### Combining with Other Features

```bash
python3 train_main.py \
    --optimization-level plateau_breaker \
    --enable-evaluation \
    --evaluation-interval 10
```

## Best Practices

1. **Start with Enhanced**: Try `enhanced` level first. Only use `plateau_breaker` if you hit a plateau
2. **Monitor Validation**: Watch validation loss more than training loss
3. **Be Patient**: Give it at least 10 epochs to show results
4. **Resume Training**: If you have a checkpoint at plateau, you can resume with plateau_breaker
5. **Document Settings**: Keep track of what worked for your specific dataset

## Example Training Session

```bash
# Initial training with enhanced
python3 train_main.py --optimization-level enhanced --epochs 100

# After hitting plateau at epoch 50, switch to plateau_breaker
# Resume from checkpoint and continue
python3 train_main.py \
    --optimization-level plateau_breaker \
    --resume-checkpoint checkpoints/checkpoint_epoch_50.ckpt \
    --epochs 150
```

## FAQ

**Q: Can I use plateau_breaker from the start?**
A: You can, but it's usually slower than enhanced. Best to start with enhanced and switch if needed.

**Q: How long should I wait before switching to plateau_breaker?**
A: If loss hasn't improved for 5-10 epochs, it's time to try plateau_breaker.

**Q: Will this work for all plateaus?**
A: It's designed for plateaus in the 2-3 range. For other ranges, you may need to adjust parameters.

**Q: Can I switch back to enhanced after the plateau?**
A: Yes, once loss is decreasing again, you can switch back for faster convergence.

**Q: What if plateau_breaker doesn't work?**
A: Review your data quality, model architecture, and consider if you've reached the model's capacity limit.

## Related Documentation

- [PLATEAU_BREAKTHROUGH_GUIDE.md](./PLATEAU_BREAKTHROUGH_GUIDE.md) - Technical analysis
- [FIX_SUMMARY.md](./FIX_SUMMARY.md) - Previous fixes for loss issues
- [TRAINING_PROCESS_FIXES.md](./TRAINING_PROCESS_FIXES.md) - Training improvements
- [FAST_CONVERGENCE_SOLUTION.md](./FAST_CONVERGENCE_SOLUTION.md) - Convergence optimization
