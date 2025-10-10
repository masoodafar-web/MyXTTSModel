# 📚 GPU Utilization Fix - Documentation Index
# راهنمای مستندات راهکار GPU Utilization

**مشکل:** استفاده 1-5% از GPU در سیستم Dual-RTX-4090  
**Problem:** 1-5% GPU utilization on dual RTX 4090 despite all optimizations

**وضعیت:** آماده برای تست  
**Status:** Ready for testing

---

## 🚀 Quick Navigation | دسترسی سریع

### For Users Who Want Quick Fix (5 minutes):
👉 **[START_HERE_GPU_FIX.md](START_HERE_GPU_FIX.md)**
- One-page quick start guide
- Essential information only
- Copy-paste commands

### For Users Who Want Complete Guide:
👉 **[GPU_UTILIZATION_FIX_README.md](GPU_UTILIZATION_FIX_README.md)**
- Complete implementation guide
- Step-by-step instructions
- Troubleshooting section
- Performance tuning tips

### For Developers Who Want Technical Details:
👉 **[CRITICAL_GPU_UTILIZATION_FIX.md](CRITICAL_GPU_UTILIZATION_FIX.md)**
- Deep technical analysis
- Root cause breakdown
- Phase-by-phase implementation
- Code examples

### For Testers Who Want Testing Guide:
👉 **[TESTING_GPU_FIX.md](TESTING_GPU_FIX.md)**
- Complete test plan (7 phases)
- Success criteria
- Verification steps
- Test report template

### For Managers Who Want Overview:
👉 **[GPU_FIX_IMPLEMENTATION_SUMMARY.md](GPU_FIX_IMPLEMENTATION_SUMMARY.md)**
- Implementation overview
- Files created
- Key optimizations
- Expected results

---

## 📁 Document Structure | ساختار مستندات

### 1. Quick Start Documents

#### [START_HERE_GPU_FIX.md](START_HERE_GPU_FIX.md)
**Audience:** All users  
**Time:** 5 minutes  
**Content:**
- Quick fix command (copy-paste)
- What was wrong (brief)
- What the fix does (brief)
- Configuration changes needed
- Verification steps

**When to use:**
- You want immediate solution
- You trust the fix and just want to apply it
- You don't care about technical details

---

### 2. User Guides

#### [GPU_UTILIZATION_FIX_README.md](GPU_UTILIZATION_FIX_README.md)
**Audience:** End users  
**Time:** 30 minutes  
**Content:**
- Complete implementation guide
- Phase-by-phase instructions
- Expected results (with numbers)
- Troubleshooting (5 scenarios)
- Performance tuning (3 hardware tiers)
- Understanding the fix (technical but accessible)
- Verification checklist

**When to use:**
- You want to understand what you're doing
- You want step-by-step guidance
- You might need to troubleshoot
- You want to tune performance

---

### 3. Technical Documentation

#### [CRITICAL_GPU_UTILIZATION_FIX.md](CRITICAL_GPU_UTILIZATION_FIX.md)
**Audience:** Developers, ML engineers  
**Time:** 1 hour  
**Content:**
- Deep root cause analysis (5 causes)
- Comprehensive solution (5 phases)
- Code examples and snippets
- Implementation details
- Advanced troubleshooting
- Performance metrics

**When to use:**
- You're a developer or ML engineer
- You want to understand the internals
- You need to modify or extend the fix
- You're debugging complex issues

---

### 4. Testing & Validation

#### [TESTING_GPU_FIX.md](TESTING_GPU_FIX.md)
**Audience:** QA engineers, testers  
**Time:** 2-3 hours  
**Content:**
- Complete test plan (7 phases)
- Pre-test verification
- Diagnostic baseline
- Apply fix steps
- Test training
- Batch size scaling tests
- Profiling
- Extended training test
- Success criteria
- Test report template

**When to use:**
- You're validating the fix
- You're running production tests
- You need to report results
- You're comparing before/after

---

### 5. Implementation Summary

#### [GPU_FIX_IMPLEMENTATION_SUMMARY.md](GPU_FIX_IMPLEMENTATION_SUMMARY.md)
**Audience:** Project managers, stakeholders  
**Time:** 15 minutes  
**Content:**
- Overview of problem and solution
- Files created (with descriptions)
- Key optimizations (bullet points)
- Performance targets (table)
- Root cause analysis (summary)
- Usage instructions
- Documentation map

**When to use:**
- You need a high-level overview
- You're reporting to stakeholders
- You want to understand what was delivered
- You need to navigate all documentation

---

## 🛠️ Tools & Utilities | ابزارها

### Diagnostic & Configuration Tools

#### [utilities/diagnose_gpu_utilization.py](utilities/diagnose_gpu_utilization.py)
**Purpose:** Diagnose GPU utilization issues

**Usage:**
```bash
python utilities/diagnose_gpu_utilization.py
python utilities/diagnose_gpu_utilization.py --config configs/config.yaml
```

**Features:**
- Check GPU availability
- Verify TensorFlow configuration
- Analyze dataset configuration
- Measure pipeline speed
- Monitor real-time GPU utilization
- Generate diagnostic report

**When to use:**
- Before applying fix (identify issues)
- After applying fix (verify success)
- When troubleshooting problems
- To get recommendations

---

#### [utilities/configure_max_gpu_utilization.py](utilities/configure_max_gpu_utilization.py)
**Purpose:** Configure TensorFlow for maximum GPU utilization

**Usage:**
```bash
python utilities/configure_max_gpu_utilization.py
python utilities/configure_max_gpu_utilization.py --verify
```

**Features:**
- Configure thread pools
- Setup GPU memory
- Enable XLA JIT
- Enable mixed precision
- Apply all optimizations
- Verify configuration

**When to use:**
- Before training (apply optimizations)
- To verify configuration
- When performance is suboptimal
- As part of setup process

---

### Automation Scripts

#### [quick_fix_gpu_utilization.sh](quick_fix_gpu_utilization.sh)
**Purpose:** Automated fix script

**Usage:**
```bash
bash quick_fix_gpu_utilization.sh
bash quick_fix_gpu_utilization.sh --config configs/config.yaml
```

**What it does:**
1. Check prerequisites
2. Run diagnostic
3. Apply TensorFlow optimizations
4. Backup and suggest config updates
5. Provide training command
6. Show monitoring instructions
7. Provide troubleshooting tips

**When to use:**
- When you want automated setup
- First time applying the fix
- To ensure nothing is missed
- To get copy-paste commands

---

### Module & Libraries

#### [myxtts/data/dataset_optimizer.py](myxtts/data/dataset_optimizer.py)
**Purpose:** Dataset optimization functions

**Functions:**
- `configure_tensorflow_for_max_throughput()` - Get TF options
- `apply_aggressive_prefetching()` - Multi-stage prefetch
- `optimize_dataset_pipeline()` - Complete optimization
- `get_optimized_dataset_options()` - Get dataset options

**Usage:**
```python
from myxtts.data.dataset_optimizer import optimize_dataset_pipeline

dataset = optimize_dataset_pipeline(
    dataset,
    batch_size=128,
    num_workers=32,
    prefetch_to_device='/GPU:0'
)
```

**When to use:**
- In your custom training scripts
- When creating datasets programmatically
- For fine-grained control
- In library integrations

---

## 🗺️ Usage Flowchart | نمودار استفاده

```
START
  │
  ├─ Want Quick Fix (5 min)?
  │   YES → START_HERE_GPU_FIX.md → Run quick_fix_gpu_utilization.sh → DONE
  │
  ├─ Want Complete Guide?
  │   YES → GPU_UTILIZATION_FIX_README.md → Follow all steps → DONE
  │
  ├─ Want to Understand Internals?
  │   YES → CRITICAL_GPU_UTILIZATION_FIX.md → Read technical details → DONE
  │
  ├─ Want to Test/Validate?
  │   YES → TESTING_GPU_FIX.md → Follow test plan → Report results → DONE
  │
  └─ Want Overview for Stakeholders?
      YES → GPU_FIX_IMPLEMENTATION_SUMMARY.md → Share with team → DONE
```

---

## 🎯 Common Scenarios | سناریوهای رایج

### Scenario 1: First-Time User
**You:** I just discovered this fix, what should I do?

**Path:**
1. Read: **START_HERE_GPU_FIX.md** (5 min)
2. Run: `bash quick_fix_gpu_utilization.sh`
3. Train: Use provided command
4. Monitor: `watch -n 1 nvidia-smi`

---

### Scenario 2: User with Issues
**You:** I applied the fix but GPU utilization is still low

**Path:**
1. Run: `python utilities/diagnose_gpu_utilization.py --config configs/config.yaml`
2. Read: **GPU_UTILIZATION_FIX_README.md** → Troubleshooting section
3. Try: Recommended solutions
4. If still stuck: Read **CRITICAL_GPU_UTILIZATION_FIX.md** → Advanced troubleshooting

---

### Scenario 3: Developer Integrating Fix
**You:** I want to integrate this into my custom training pipeline

**Path:**
1. Read: **CRITICAL_GPU_UTILIZATION_FIX.md** → Implementation details
2. Study: `myxtts/data/dataset_optimizer.py` code
3. Integrate: Use optimization functions in your code
4. Test: Follow **TESTING_GPU_FIX.md**

---

### Scenario 4: QA Engineer Testing
**You:** I need to validate this fix works as claimed

**Path:**
1. Read: **TESTING_GPU_FIX.md** (complete test plan)
2. Run: All 7 test phases
3. Document: Fill test report template
4. Compare: Expected vs actual results

---

### Scenario 5: Manager Evaluating Solution
**You:** I need to understand what was delivered and expected impact

**Path:**
1. Read: **GPU_FIX_IMPLEMENTATION_SUMMARY.md** (15 min overview)
2. Review: Performance targets table
3. Check: Files created list
4. Decision: Approve for testing/deployment

---

## 📊 Performance Metrics | معیارهای عملکرد

### Expected Improvements:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU:0 Utilization | 1-5% | 60-80% | **15-80x** |
| GPU:1 Utilization | 1-5% | 85-95% | **17-95x** |
| Step Time (batch=128) | 2-5s | <0.5s | **5-10x** |
| Throughput | 20-50 samples/s | 200-300 samples/s | **10-15x** |
| Training Speed | Very slow | Fast | **10-20x** |

---

## ✅ Verification Checklist | چک‌لیست تایید

Before claiming fix is successful:

- [ ] GPU:0 utilization > 60%
- [ ] GPU:1 utilization > 80%
- [ ] Step time < 0.5s (batch=128)
- [ ] No OOM errors
- [ ] Training logs show TF-native loading
- [ ] nvidia-smi shows stable utilization
- [ ] Throughput > 200 samples/s

---

## 🆘 Need Help? | کمک نیاز دارید؟

### Step 1: Run Diagnostic
```bash
python utilities/diagnose_gpu_utilization.py --config configs/config.yaml
```

### Step 2: Check Documentation
- Quick fix not working? → **START_HERE_GPU_FIX.md**
- Need troubleshooting? → **GPU_UTILIZATION_FIX_README.md**
- Technical issue? → **CRITICAL_GPU_UTILIZATION_FIX.md**

### Step 3: Run Profiler
```bash
python utilities/dual_gpu_bottleneck_profiler.py --batch-size 128
```

### Step 4: Report Issue
Include:
- Diagnostic output
- Profiler output
- Training logs
- nvidia-smi output
- Hardware specs

---

## 📅 Document Versions | نسخه‌های مستندات

| Document | Version | Date | Status |
|----------|---------|------|--------|
| START_HERE_GPU_FIX.md | 1.0 | 2025-10-10 | ✅ Final |
| GPU_UTILIZATION_FIX_README.md | 1.0 | 2025-10-10 | ✅ Final |
| CRITICAL_GPU_UTILIZATION_FIX.md | 3.0 | 2025-10-10 | ✅ Final |
| TESTING_GPU_FIX.md | 1.0 | 2025-10-10 | ✅ Final |
| GPU_FIX_IMPLEMENTATION_SUMMARY.md | 1.0 | 2025-10-10 | ✅ Final |
| GPU_FIX_INDEX.md | 1.0 | 2025-10-10 | ✅ Final |

---

## 🎉 Summary | خلاصه

This documentation suite provides **everything you need** to fix the 1-5% GPU utilization issue:

**For Quick Fix:**
- START_HERE_GPU_FIX.md + quick_fix_gpu_utilization.sh

**For Complete Understanding:**
- GPU_UTILIZATION_FIX_README.md (user guide)
- CRITICAL_GPU_UTILIZATION_FIX.md (technical)

**For Testing:**
- TESTING_GPU_FIX.md (test plan)
- diagnose_gpu_utilization.py (diagnostic)

**For Integration:**
- dataset_optimizer.py (library)
- configure_max_gpu_utilization.py (setup)

**Choose your path based on your needs, and follow the appropriate guide!**

---

**Version:** 1.0  
**Date:** 2025-10-10  
**Maintained by:** GitHub Copilot  
**For:** Dual RTX 4090 GPU Utilization Issue
