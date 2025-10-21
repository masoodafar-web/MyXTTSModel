#!/usr/bin/env python3
"""Lightweight training entrypoint that avoids the CLI interface.

Configure and launch training from Python without passing many CLI flags.
Restored by assistant. Uses stable defaults and aligns token caps with
fixed padding to avoid tf.data padding errors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import train_main
from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
from myxtts.training.trainer import XTTSTrainer
from myxtts.utils.commons import find_latest_checkpoint, setup_logging


@dataclass
class TrainingRunSettings:
    """Programmatic configuration for a training run."""

    # Dataset roots (update to your paths if needed)
    train_data: str = "data/ljspeech/LJSpeech-1.1"
    val_data: Optional[str] = None  # Use train split logic by default
    checkpoint_dir: str = "./checkpointsmain"

    # Core training params
    epochs: int = 500
    batch_size: int = 16
    grad_accum: Optional[int] = 2
    num_workers: int = 16
    lr: float = 3e-5

    # Model/strategy presets
    model_size: str = "tiny"
    decoder_strategy: str = "autoregressive"
    optimization_level: str = "basic"

    # GST
    enable_gst: bool = False
    gst_num_style_tokens: int = 10
    gst_style_token_dim: int = 256
    gst_style_embedding_dim: int = 256
    gst_num_heads: int = 4
    gst_style_loss_weight: float = 1.0

    # Evaluation
    enable_evaluation: bool = False
    evaluation_interval: int = 50
    use_simple_loss: bool = True

    # Static shapes (prevents retracing and keeps GPU steady)
    enable_static_shapes: bool = True
    max_attention_len: Optional[int] = 1024  # cap for attention (tokens)
    max_text_length: int = 200               # fixed padding length for text
    max_mel_frames: Optional[int] = 800      # fixed padding length for mels

    # Pipeline
    prefetch_buffer_size: Optional[int] = 12
    prefetch_to_gpu: Optional[bool] = True

    # Dual‑GPU (optional)
    pipeline_buffer_size: int = 50
    model_start_delay: float = 2.0
    data_gpu: Optional[int] = None
    model_gpu: Optional[int] = None
    enable_memory_isolation: bool = False
    data_gpu_memory: int = 8192
    model_gpu_memory: int = 16384

    # Misc
    apply_fast_convergence: bool = False
    tensorboard_log_dir: Optional[str] = None
    enable_eager_debug: bool = False
    reset_training: bool = True
    enable_full_loss_stack: bool = False
    use_adaptive_loss: Optional[bool] = False
    download_ljspeech: bool = True


def run_training(settings: TrainingRunSettings) -> None:
    """Run training using the provided settings."""

    logger = setup_logging()
    logger.info("🚀 Launching training via programmatic configuration")
    if settings.enable_static_shapes:
        logger.info(
            "ℹ️ Static padding enabled (max_text=%s, max_mels=%s)",
            settings.max_text_length,
            settings.max_mel_frames if settings.max_mel_frames is not None else "auto",
        )
    else:
        logger.info("ℹ️ Static padding disabled; dataset will use dynamic sequence lengths")
    if not settings.enable_full_loss_stack:
        logger.info("ℹ️ Using simplified loss stack (advanced voice/prosody losses disabled)")

    # Build configuration using helpers from train_main
    config_kwargs = dict(
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        epochs=settings.epochs,
        lr=settings.lr,
        checkpoint_dir=settings.checkpoint_dir,
        decoder_strategy=settings.decoder_strategy,
        model_size=settings.model_size,
        enable_gst=settings.enable_gst,
        gst_num_style_tokens=settings.gst_num_style_tokens,
        gst_style_token_dim=settings.gst_style_token_dim,
        gst_style_embedding_dim=settings.gst_style_embedding_dim,
        gst_num_heads=settings.gst_num_heads,
        gst_style_loss_weight=settings.gst_style_loss_weight,
        enable_automatic_evaluation=settings.enable_evaluation,
        evaluation_interval=settings.evaluation_interval,
        use_simple_loss=settings.use_simple_loss,
        enable_static_shapes=settings.enable_static_shapes,
        max_text_length_override=settings.max_text_length,
        max_mel_frames_override=settings.max_mel_frames,
        data_gpu=settings.data_gpu,
        model_gpu=settings.model_gpu,
        pipeline_buffer_size=settings.pipeline_buffer_size,
        model_start_delay=settings.model_start_delay,
        enable_full_loss_stack=settings.enable_full_loss_stack,
        download_ljspeech=settings.download_ljspeech,
    )

    if settings.grad_accum is not None:
        config_kwargs["grad_accum"] = settings.grad_accum
    if settings.prefetch_buffer_size is not None:
        config_kwargs["prefetch_buffer_size"] = settings.prefetch_buffer_size
    if settings.prefetch_to_gpu is not None:
        config_kwargs["prefetch_to_gpu"] = settings.prefetch_to_gpu
    if settings.max_attention_len is not None:
        config_kwargs["max_attention_len_override"] = settings.max_attention_len
    if settings.use_adaptive_loss is not None:
        config_kwargs["use_adaptive_loss_override"] = settings.use_adaptive_loss

    config = train_main.build_config(**config_kwargs)

    # Configure dataset metadata/wavs paths intelligently based on what's present
    try:
        train_root = settings.train_data
        # Training metadata detection
        meta_csv_default = os.path.join(train_root, "metadata.csv")
        meta_csv_train = os.path.join(train_root, "metadata_train.csv")
        wavs_dir = os.path.join(train_root, "wavs")

        if os.path.isfile(meta_csv_default):
            # Use standard LJSpeech layout
            config.data.metadata_train_file = None
            config.data.wavs_train_dir = None
        elif os.path.isfile(meta_csv_train):
            # Use custom train/eval metadata layout (e.g., small_dataset_test)
            config.data.metadata_train_file = "metadata_train.csv"
            config.data.wavs_train_dir = "wavs" if os.path.isdir(wavs_dir) else None
        else:
            # Leave as default and let the loader raise a clear error if missing
            config.data.metadata_train_file = None
            config.data.wavs_train_dir = None

        # Validation metadata detection (use same root when val_data is None)
        val_root = settings.val_data or train_root
        meta_csv_eval = os.path.join(val_root, "metadata_eval.csv")
        if os.path.isfile(meta_csv_eval):
            config.data.metadata_eval_file = "metadata_eval.csv"
            config.data.wavs_eval_dir = "wavs" if os.path.isdir(os.path.join(val_root, "wavs")) else None
        else:
            config.data.metadata_eval_file = None
            config.data.wavs_eval_dir = None

        logger.info(
            "📦 Dataset mapping → train_meta=%s, val_meta=%s, wavs=%s",
            getattr(config.data, "metadata_train_file", None) or "metadata.csv",
            getattr(config.data, "metadata_eval_file", None) or "metadata.csv",
            getattr(config.data, "wavs_train_dir", None) or "wavs",
        )
    except Exception as e:
        logger.warning("Dataset auto-detection failed: %s (falling back to defaults)", e)
        config.data.metadata_train_file = None
        config.data.metadata_eval_file = None
        config.data.wavs_train_dir = None
        config.data.wavs_eval_dir = None

    # Enable periodic text2audio evaluation
    config.training.enable_text2audio_eval = True
    config.training.text2audio_interval_steps = 100

    # Conservative runtime defaults for stability
    try:
        config.data.enable_xla = False
        config.data.mixed_precision = False
    except Exception:
        pass

    # Apply higher-level presets
    args_ns = SimpleNamespace(model_size=settings.model_size, batch_size=settings.batch_size)
    config = train_main.apply_optimization_level(config, settings.optimization_level, args_ns)
    if settings.apply_fast_convergence:
        config = train_main.apply_fast_convergence_config(config)

    # Align token cap with fixed padding to avoid padded_batch shape errors
    try:
        if getattr(config.data, "pad_to_fixed_length", False) and getattr(config.data, "max_text_length", None):
            config.data.max_text_tokens = config.data.max_text_length
    except Exception:
        pass

    # Re-apply user LR preference after optimization profile adjustment
    try:
        if settings.lr is not None:
            config.training.learning_rate = settings.lr
    except Exception:
        pass

    if settings.tensorboard_log_dir:
        setattr(config.training, "tensorboard_log_dir", settings.tensorboard_log_dir)
    if settings.enable_eager_debug:
        setattr(config.training, "enable_eager_debug", True)
    if settings.grad_accum is not None:
        config.training.gradient_accumulation_steps = max(1, settings.grad_accum)
        logger.info("ℹ️ Forcing gradient accumulation steps to %d", config.training.gradient_accumulation_steps)

    # Locate latest checkpoint unless a reset is requested
    resume_ckpt = None
    if not settings.reset_training:
        resume_ckpt = find_latest_checkpoint(settings.checkpoint_dir)
        if resume_ckpt:
            logger.info("🔁 Resuming from checkpoint: %s", resume_ckpt)

    # Choose trainer
    is_multi_gpu = settings.data_gpu is not None and settings.model_gpu is not None
    if settings.enable_memory_isolation and not is_multi_gpu:
        logger.warning("⚠️ Memory isolation requested but data/model GPU IDs are missing; disabling it.")

    model_device = "/GPU:1" if is_multi_gpu else None

    if settings.enable_memory_isolation and is_multi_gpu:
        trainer = MemoryIsolatedDualGPUTrainer(
            config=config,
            data_gpu_id=settings.data_gpu,
            model_gpu_id=settings.model_gpu,
            data_gpu_memory_limit=settings.data_gpu_memory,
            model_gpu_memory_limit=settings.model_gpu_memory,
            resume_checkpoint=resume_ckpt,
            enable_monitoring=True,
        )
    else:
        trainer = XTTSTrainer(
            config=config,
            resume_checkpoint=resume_ckpt,
            model_device=model_device,
        )

    train_ds, val_ds = trainer.prepare_datasets(
        train_data_path=settings.train_data,
        val_data_path=settings.val_data,
    )

    trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=config.training.epochs,
    )

    final_checkpoint = os.path.join(config.training.checkpoint_dir, "final_model")
    trainer.save_checkpoint(final_checkpoint)
    logger.info("✅ Training finished. Final checkpoint saved to %s", final_checkpoint)


if __name__ == "__main__":
    CUSTOM_SETTINGS = TrainingRunSettings()
    run_training(CUSTOM_SETTINGS)
