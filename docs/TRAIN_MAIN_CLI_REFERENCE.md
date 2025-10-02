# `train_main.py` Command-Line Arguments

This reference describes every CLI switch exposed by `train_main.py`, the
allowed values for each option, and notes on how options interact.  All flags
are optional; defaults appear in parentheses.

## Dataset & Checkpoints
- `--train-data PATH` (default `../dataset/dataset_train`)
  - Root directory for the training subset.
- `--val-data PATH` (default `../dataset/dataset_eval`)
  - Root directory for the validation subset.
- `--checkpoint-dir PATH` (default `./checkpointsmain`)
  - Directory used for checkpoints, logs, and TensorBoard summaries.

## Core Training Schedule
- `--epochs INT` (default `500`)
  - Number of training epochs for Stage 1.
- `--batch-size INT` (default `64`)
  - Global batch size.  Adjust together with `--grad-accum` on small GPUs.
- `--grad-accum INT` (default `2` for enhanced / `16` fallback)
  - Gradient accumulation steps.  Must satisfy `batch_size / grad_accum ≥ 8`
    for best GPU utilisation.
- `--num-workers INT` (default `8`)
  - Number of data-loader workers (tf.data parallel calls).
- `--lr FLOAT` (default `8e-5`)
  - Base learning rate for Stage 1.

## Data Pipeline Controls
- `--prefetch-buffer-size INT`
  - Overrides tf.data prefetch buffer (default taken from optimisation
    presets). Minimum enforced value is `1`.
- `--prefetch-to-gpu`
  - Force prefetch to GPU memory.
- `--no-prefetch-to-gpu`
  - Keep batches on host RAM.  Useful on GPUs with tight memory budgets.
  (Use at most one of the two prefetch flags.)

## Model Capacity & Decoder Options
- `--model-size {tiny, small, normal, big}` (default `normal`)
  - Applies the corresponding entry in `MODEL_SIZE_PRESETS`.
- `--decoder-strategy {autoregressive, non_autoregressive}` (default
  `autoregressive`)
  - Selects between the AR decoder or the parallel FastSpeech-style decoder.
- `--vocoder-type {griffin_lim, hifigan, bigvgan}` (default `griffin_lim`)
  - Backend used during inference/evaluation.

## Run Management
- `--resume`
  - Legacy toggle; no effect (resumption is automatic).
- `--reset-training`
  - Ignore previous checkpoints and start Stage 1 from scratch.
- `--optimization-level {basic, enhanced, experimental, plateau_breaker}`
  (default `enhanced`)
  - Activates bundled hyper-parameter presets.
- `--apply-fast-convergence`
  - Merge in the overrides defined in `fast_convergence_config.py`.

## Speaker Encoder Overrides
- `--use-pretrained-speaker-encoder` / `--disable-pretrained-speaker-encoder`
  - Enable or disable loading an external speaker encoder.  Defaults to
    disabled unless `--speaker-encoder-path` is provided.
- `--speaker-encoder-path PATH`
  - Checkpoint for the speaker encoder.  Requires one of the enable flags.
- `--speaker-encoder-type {ecapa_tdnn, resemblyzer, coqui}` (default
  `ecapa_tdnn`)
  - Architecture hint for the pretrained encoder.
- `--freeze-speaker-encoder` / `--unfreeze-speaker-encoder`
  - Keep encoder weights frozen (default) or finetune them.
- `--contrastive-loss-temperature FLOAT` (default `0.1`)
  - Temperature hyper-parameter for the contrastive speaker loss.
- `--contrastive-loss-margin FLOAT` (default `0.2`)
  - Margin hyper-parameter for the contrastive speaker loss.

## Global Style Tokens (GST)
- `--enable-gst`
  - Ensure GST layers remain active (default behaviour).
- `--gst-num-style-tokens INT` (default `10`)
- `--gst-style-token-dim INT` (default `256`)
- `--gst-style-embedding-dim INT` (default `256`)
- `--gst-num-heads INT` (default `4`)
- `--gst-style-loss-weight FLOAT` (default `1.0`)

## Logging & Diagnostics
- `--simple-loss`
  - Replace the full loss with a minimal stabilised loss (debug only).
- `--enable-gpu-stabilizer` / `--disable-gpu-stabilizer`
  - Force usage of the Advanced GPU stabiliser (default disabled).
- `--multi-gpu`
  - Enable `tf.distribute.MirroredStrategy` and multi-GPU support.
- `--visible-gpus STRING`
  - Comma-separated GPU indices to expose to TensorFlow (e.g. `"0,1"`).
- `--tensorboard-log-dir PATH`
  - Directory to store TensorBoard event files. Defaults to
    `<checkpoint-dir>/tensorboard`.
- `--enable-eager-debug`
  - Run TensorFlow functions eagerly for easier debugging (slower).

## Evaluation & Deployment
- `--enable-evaluation`
  - Run MOSNet / ASR-WER checks after each evaluation interval.
- `--evaluation-interval INT` (default `50`)
  - Epoch interval between automatic evaluations.
- `--create-optimized-model`
  - Launch the post-training compression pipeline.
- `--lightweight-config PATH`
  - Optional lightweight JSON/ YAML config to merge into the default config.
- `--compression-target FLOAT` (default `2.0`)
  - Target compression / speed-up factor when `--create-optimized-model` is set.

## Environment Variables
- `MYXTTS_LOG_DIR`, `MYXTTS_RUN_NAME`
  - Control the rotating log file location and run identifier.
- `MYXTTS_LOG_FILE`
  - Set automatically; points to the active log file.
- `MYXTTS_SKIP_WARMUP`
  - If `1`, skips the distributed warm-up pass.
- `MYXTTS_SIMPLE_LOSS`
  - If `1`, forces simple loss mode (same behaviour as `--simple-loss`).

## Usage Examples
```bash
# Default training (stage 1 only)
python3 train_main.py --model-size tiny --optimization-level enhanced

# Custom GPU visibility and TensorBoard output
python3 train_main.py \
    --visible-gpus 0,1 \
    --tensorboard-log-dir ./runs/tensorboard \
    --batch-size 32 --grad-accum 2

# Speaker encoder finetuning run
python3 train_main.py \
    --use-pretrained-speaker-encoder \
    --speaker-encoder-path ./weights/ecapa_tdnn.pth \
    --speaker-encoder-type ecapa_tdnn \
    --unfreeze-speaker-encoder
```
