# Critical Fixes Implementation Guide

This document explains the critical fixes added to the MyXTTS training stack
and how to use or extend them in future work.

## 1. Dataset Profiler Fallback
- **File:** `myxtts/data/ljspeech.py`
- **Change:** The dataset loader now injects a safe no-op profiler when a
  profiler instance is missing or partially implemented. This prevents runtime
  crashes coming from calls such as `self.profiler.record_cache_hit()` during
  dataset iteration.
- **How to customize:** Pass a profiler object through the `DataConfig`
  (`config.data.profiler = YourProfiler()`) that exposes the following
  methods: `record_cache_hit`, `record_cache_miss`, `record_cache_error`, and
  `profile_operation(name)` (returning a context manager). If the implementation
  lacks any of these hooks, the loader automatically falls back to the no-op
  profiler to preserve training stability.

## 2. Config-Aware GPU Setup
- **File:** `myxtts/utils/commons.py`
- **Change:** `configure_gpus` now respects configurable flags for mixed
  precision, XLA, and eager execution debugging. The trainer forwards user
  preferences directly from the loaded configuration, removing the hard-coded
  mixed precision override that previously forced `mixed_float16`.
- **Usage:**
  ```yaml
  data:
    mixed_precision: false
    enable_xla: false
  training:
    enable_eager_debug: true
  ```
  The above settings translate into
  `configure_gpus(..., enable_mixed_precision=False, enable_xla=False,
  enable_eager_debug=True)`, ensuring the runtime honors your training
  requirements. Environment variables `MYXTTS_ENABLE_MIXED_PRECISION` and
  `MYXTTS_ENABLE_XLA` can also be used to force behaviour without editing the
  config file.

## 3. GST Toggle From CLI
- **File:** `train_main.py`
- **Change:** Global Style Tokens can now be disabled explicitly via
  `--disable-gst`. The old pattern `--enable-gst false` no longer worked with
  argparse; the new pair of complementary flags keeps the default enabled while
  allowing `python3 train_main.py --disable-gst ...` to opt out.
- **Documentation:** The usage example in the training script reflects the new
  semantics so future readers see the correct command.

## 4. Verification Steps
- Unit tests are unchanged, but running them now requires `pytest` to be
  installed. If it is missing, install it inside your environment before
  executing:
  ```bash
  pip install pytest
  python3 -m pytest tests/test_basic_functionality.py
  ```
  This quick check ensures the core training entry point still behaves after
  the fixes.

## 5. Impact Summary
- Training scripts no longer crash when the profiler hook is absent.
- GPU initialisation is driven by configuration, preventing unwanted
  mixed-precision activation or eager execution in production runs.
- Voice cloning experiments can toggle GST cleanly via the CLI, simplifying
  benchmarking workflows.

---

# Loss Monitoring & Prefetch Controls (2025-09-27)

## Loss Weight Instrumentation
- **Files:** `myxtts/training/trainer.py`, `myxtts/training/losses.py`
- **What changed:**
  - The trainer now snapshots configured loss weights at start-up and prints
    them once so every run records the active balancing between mel, stop,
    voice, prosody, and auxiliary terms.
  - Helper methods `_log_loss_breakdown` and `_collect_loss_weights` gather the
    per-component losses and weighted contributions returned by `XTTSLoss`.
  - Each epoch summary logs both raw and weighted values (e.g. `voice_similarity_loss`)
    while the progress bar shows a compact `voice=` field beside mel and stop
    losses. Weighted entries are still available for external consumers such as
    W&B.
- **Why it matters:** Adaptive loss scaling makes it hard to see which
  component dominates; these logs let you spot runaway voice/prosody penalties
  early without attaching a debugger.

## Prefetch Configuration From CLI
- **Files:** `train_main.py`, `myxtts/config/config.py`
- **What changed:** Added `--prefetch-buffer-size`, `--prefetch-to-gpu`, and
  `--no-prefetch-to-gpu` flags, plumbing them into `build_config`. Users can
  now trial data-pipeline tweaks directly from the command line instead of
  editing YAML configs when running on memory-constrained GPUs.
- **Usage examples:**
  ```bash
  python3 train_main.py --prefetch-buffer-size 4 --no-prefetch-to-gpu ...
  python3 train_main.py --prefetch-buffer-size 32 --prefetch-to-gpu ...
  ```

## Pretrained Speaker Encoder Controls (Step 3)
- **Files:** `train_main.py`, `myxtts/config/config.py`
- **What changed:**
  - Added CLI switches `--use-pretrained-speaker-encoder` /
    `--disable-pretrained-speaker-encoder`, plus path/type/freezing overrides
    (`--speaker-encoder-path`, `--speaker-encoder-type`,
    `--freeze-speaker-encoder`, `--unfreeze-speaker-encoder`,
    `--contrastive-loss-temperature`, `--contrastive-loss-margin`).
  - `build_config` now propagates these overrides into `ModelConfig`, allowing
    per-run choice between the built-in audio encoder and an external ECAPA,
    Resemblyzer, or Coqui model.
- **Usage examples:**
  ```bash
  python3 train_main.py --use-pretrained-speaker-encoder \
      --speaker-encoder-path ./checkpoints/ecapa_tdnn.pth \
      --speaker-encoder-type ecapa_tdnn --freeze-speaker-encoder

  python3 train_main.py --use-pretrained-speaker-encoder \
      --speaker-encoder-path ./weights/resemblyzer.pt \
      --speaker-encoder-type resemblyzer --unfreeze-speaker-encoder \
      --contrastive-loss-temperature 0.07 --contrastive-loss-margin 0.25
  ```
- **Notes:** When a path is supplied without the enable flag, the script now
  auto-enables the pretrained encoder and warns if the file is missing. Paths
  are not bundled with the repo; provide your own weights.

## Two-Stage Training Workflow (Step 5)
- **Files:** `train_main.py`, `myxtts/training/two_stage_trainer.py`
- **What changed:**
  - CLI flag `--two-stage-training` kicks off the text-to-mel run (Stage 1) and
    an optional HiFi-GAN vocoder fine-tune (Stage 2).
  - Stage-specific knobs (`--stage1-epochs`, `--stage1-lr`, `--stage2-epochs`,
    `--stage2-lr`, `--stage2-batch-size`, `--stage2-max-audio-seconds`) expose
    lightweight defaults for quick experiments.
  - Added `_create_vocoder_dataset` helper that pulls cached mel-specs and
    corresponding waveforms directly from `LJSpeechDataset`, pads batches, and
    caps audio length to keep the vocoder loop memory-friendly.
  - `TwoStageTrainer` now uses the modern `reset_state()` metric API.
- **Usage example (CPU-friendly sanity run):**
  ```bash
  CUDA_VISIBLE_DEVICES='' python3 train_main.py \
      --two-stage-training \
      --stage1-epochs 1 --stage2-epochs 1 --stage2-batch-size 1 \
      --stage2-max-audio-seconds 3.0 \
      --model-size tiny --batch-size 1 --grad-accum 1 \
      --train-data ./small_dataset_test/dataset_train_small \
      --val-data   ./small_dataset_test/dataset_val_small \
      --checkpoint-dir ./outputs/two_stage_demo
  ```
  Stage 1 reuses the existing trainer/checkpoint pipeline; Stage 2 adds a small
  vocoder pass that saves under `<checkpoint-dir>/stage2/`.
- **Notes:** The vocoder loop is still compute-heavy; keep batches small (1–2)
  and trim audio length for CPU runs. Stage 2 runs can be skipped entirely by
  omitting `--two-stage-training`.

## Verification
- `python3 -m pytest tests/test_basic_functionality.py`
- `python3 -m pytest tests/test_enhanced_model.py`
- Sanity training run on the small dataset with new logging enabled:
  ```bash
  python3 train_main.py --model-size tiny --batch-size 2 --epochs 1 \
      --disable-gst --prefetch-buffer-size 4 --no-prefetch-to-gpu \
      --train-data ./small_dataset_test/dataset_train_small \
      --val-data ./small_dataset_test/dataset_val_small
  ```

## Observed Behaviour
- Epoch summaries now include lines such as:
  ```
  Train loss components: mel_loss=4.23, stop_loss=0.64, voice_similarity_loss=0.59
  ```
  and the progress bar shows `voice=0.610` alongside mel/stop values. These
  were reviewed during the latest sanity run.

## Next Steps
- Enable `--enable-evaluation` on a longer training job to pair MOSNet/WER
  metrics with the new loss logs.
- Feed the loss breakdown into W&B (already wired) for trend visualisation
  across experiments.

Keep this guide with your other documentation to track why these fixes were
introduced and how to adapt them when the project evolves.
