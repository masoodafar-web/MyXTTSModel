#!/usr/bin/env python3
"""Lightweight training entrypoint that avoids the CLI interface.

Edit the `TrainingRunSettings` dataclass (or pass a customised instance to
`run_training`) to configure the run directly from Python code.
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

    # train_data: str = "../dataset/dataset_train"
    # val_data: str = "../dataset/dataset_eval"
    train_data: str = None
    val_data: str = None
    checkpoint_dir: str = "./checkpointsmain"

    epochs: int = 500
    batch_size: int = 32
    grad_accum: Optional[int] = None
    num_workers: int = 16
    lr: float = 8e-5

    model_size: str = "tiny"
    decoder_strategy: str = "autoregressive"
    optimization_level: str = "enhanced"

    enable_gst: bool = True
    gst_num_style_tokens: int = 10
    gst_style_token_dim: int = 256
    gst_style_embedding_dim: int = 256
    gst_num_heads: int = 4
    gst_style_loss_weight: float = 1.0

    enable_evaluation: bool = False
    evaluation_interval: int = 50
    use_simple_loss: bool = False

    enable_static_shapes: bool = False
    max_attention_len: Optional[int] = 1024
    max_text_length: int = 1024
    max_mel_frames: Optional[int] = None

    prefetch_buffer_size: Optional[int] = None
    prefetch_to_gpu: Optional[bool] = None

    pipeline_buffer_size: int = 50
    model_start_delay: float = 2.0

    data_gpu: Optional[int] = None
    model_gpu: Optional[int] = None
    enable_memory_isolation: bool = False
    data_gpu_memory: int = 8192
    model_gpu_memory: int = 16384

    apply_fast_convergence: bool = False
    tensorboard_log_dir: Optional[str] = None
    enable_eager_debug: bool = False
    reset_training: bool = False
    enable_full_loss_stack: bool = False
    use_adaptive_loss: Optional[bool] = None
    download_ljspeech: bool = True


def run_training(settings: TrainingRunSettings) -> None:
    """Run training using the provided settings."""

    logger = setup_logging()
    logger.info("🚀 Launching training via programmatic configuration")
    if not settings.enable_static_shapes:
        logger.info("ℹ️ Static padding disabled; dataset will use dynamic sequence lengths")
    else:
        logger.info(
            "ℹ️ Static padding enabled (max_text=%s, max_mels=%s)",
            settings.max_text_length,
            settings.max_mel_frames if settings.max_mel_frames is not None else "auto",
        )
    if not settings.enable_full_loss_stack:
        logger.info("ℹ️ Using simplified loss stack (advanced voice/prosody losses disabled)")

    # Build configuration using the helper from train_main
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
    if settings.max_mel_frames is not None:
        config_kwargs["max_mel_frames_override"] = settings.max_mel_frames
    if settings.use_adaptive_loss is not None:
        config_kwargs["use_adaptive_loss_override"] = settings.use_adaptive_loss

    config = train_main.build_config(**config_kwargs)
    config.training.enable_text2audio_eval = True
    config.training.text2audio_interval_steps = 100

    # Apply higher level optimisation presets, reusing the CLI helper
    args_ns = SimpleNamespace(model_size=settings.model_size, batch_size=settings.batch_size)
    config = train_main.apply_optimization_level(config, settings.optimization_level, args_ns)

    if settings.apply_fast_convergence:
        config = train_main.apply_fast_convergence_config(config)

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
    # Update the fields below to match your dataset and hardware setup.
    CUSTOM_SETTINGS = TrainingRunSettings()
    run_training(CUSTOM_SETTINGS)
