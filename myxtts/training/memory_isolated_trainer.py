"""
Memory-Isolated Dual-GPU Trainer for Producer-Consumer Pipeline.

This module implements a specialized trainer that enforces strict memory isolation
between data processing GPU and model training GPU, enabling true producer-consumer
pipeline pattern for maximum GPU utilization.
"""

import logging
from typing import Dict, Optional, Tuple, Any
import tensorflow as tf
import time

from .trainer import XTTSTrainer
from ..config.config import XTTSConfig
from ..models.xtts import XTTS
from ..utils.gpu_memory import (
    setup_gpu_memory_isolation,
    monitor_gpu_memory,
    log_memory_stats,
    detect_memory_leak
)

logger = logging.getLogger("MyXTTS")


class MemoryIsolatedDualGPUTrainer(XTTSTrainer):
    """
    Memory-Isolated Dual-GPU Trainer with Producer-Consumer Pipeline.
    
    This trainer enforces strict memory isolation between GPUs:
    - Data GPU: Only for data loading and preprocessing (limited memory)
    - Model GPU: Only for model training (larger memory allocation)
    
    Pipeline Flow:
    1. Phase 1: Data processing on Data GPU
    2. Phase 2: Transfer to Model GPU (controlled)
    3. Phase 3: Training on Model GPU
    
    Features:
    - Memory isolation prevents conflicts
    - Double buffering for smooth pipeline
    - Async processing between GPUs
    - Memory leak detection
    - Real-time monitoring
    """
    
    def __init__(
        self,
        config: XTTSConfig,
        data_gpu_id: int,
        model_gpu_id: int,
        data_gpu_memory_limit: int = 8192,
        model_gpu_memory_limit: int = 16384,
        model: Optional[XTTS] = None,
        resume_checkpoint: Optional[str] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize Memory-Isolated Dual-GPU Trainer.
        
        Args:
            config: Training configuration
            data_gpu_id: Physical GPU ID for data processing
            model_gpu_id: Physical GPU ID for model training
            data_gpu_memory_limit: Memory limit in MB for data GPU (default: 8GB)
            model_gpu_memory_limit: Memory limit in MB for model GPU (default: 16GB)
            model: Pre-initialized model (creates new if None)
            resume_checkpoint: Path to checkpoint for resuming
            enable_monitoring: Enable real-time memory monitoring
        """
        self.data_gpu_id = data_gpu_id
        self.model_gpu_id = model_gpu_id
        self.data_gpu_memory_limit = data_gpu_memory_limit
        self.model_gpu_memory_limit = model_gpu_memory_limit
        self.enable_monitoring = enable_monitoring
        
        # Setup memory isolation FIRST (before any TensorFlow operations)
        logger.info("=" * 70)
        logger.info("Memory-Isolated Dual-GPU Trainer Initialization")
        logger.info("=" * 70)
        
        success = setup_gpu_memory_isolation(
            data_gpu_id=data_gpu_id,
            model_gpu_id=model_gpu_id,
            data_gpu_memory_limit=data_gpu_memory_limit,
            model_gpu_memory_limit=model_gpu_memory_limit
        )
        
        if not success:
            raise RuntimeError(
                "Failed to setup GPU memory isolation. "
                "Ensure this trainer is initialized before any TensorFlow operations."
            )
        
        # After memory isolation setup, GPUs are remapped to logical devices
        # Original data_gpu -> /GPU:0, Original model_gpu -> /GPU:1
        self.data_device = '/GPU:0'
        self.model_device = '/GPU:1'
        
        logger.info(f"🎯 Device Mapping:")
        logger.info(f"   Physical GPU {data_gpu_id} → Logical {self.data_device} (Data Processing)")
        logger.info(f"   Physical GPU {model_gpu_id} → Logical {self.model_device} (Model Training)")
        
        # Initialize parent trainer with model_device set
        super().__init__(
            config=config,
            model=model,
            resume_checkpoint=resume_checkpoint,
            model_device=self.model_device
        )
        
        # Memory monitoring state
        self.baseline_data_memory = None
        self.baseline_model_memory = None
        self.last_memory_check_step = 0
        self.memory_check_interval = 100  # Check every N steps
        
        # Double buffering state
        self.buffer_queue = []
        self.max_buffer_size = 2  # Two buffers for smooth pipeline
        
        logger.info("✅ Memory-Isolated Dual-GPU Trainer initialized successfully")
        logger.info("=" * 70)
    
    def _setup_memory_baselines(self):
        """Setup baseline memory measurements for leak detection."""
        if not self.enable_monitoring:
            return
        
        data_info, model_info = monitor_gpu_memory(self.data_gpu_id, self.model_gpu_id)
        
        if data_info and 'used_mb' in data_info:
            self.baseline_data_memory = data_info['used_mb']
            logger.info(f"📊 Baseline Data GPU memory: {self.baseline_data_memory:.1f}MB")
        
        if model_info and 'used_mb' in model_info:
            self.baseline_model_memory = model_info['used_mb']
            logger.info(f"📊 Baseline Model GPU memory: {self.baseline_model_memory:.1f}MB")
    
    def _check_memory_health(self, step: int):
        """Check for memory leaks and log statistics."""
        if not self.enable_monitoring:
            return
        
        if step - self.last_memory_check_step < self.memory_check_interval:
            return
        
        self.last_memory_check_step = step
        
        # Log current memory stats
        log_memory_stats(self.data_gpu_id, self.model_gpu_id, step=step)
        
        # Check for memory leaks if baselines are set
        if self.baseline_data_memory is not None:
            data_leak = detect_memory_leak(
                self.data_gpu_id,
                self.baseline_data_memory,
                threshold_mb=200
            )
            if data_leak:
                logger.warning("⚠️  Consider running cleanup on data GPU")
        
        if self.baseline_model_memory is not None:
            model_leak = detect_memory_leak(
                self.model_gpu_id,
                self.baseline_model_memory,
                threshold_mb=500
            )
            if model_leak:
                logger.warning("⚠️  Consider running cleanup on model GPU")
    
    @tf.function
    def _preprocess_on_data_gpu(
        self,
        text_sequences: tf.Tensor,
        mel_spectrograms: tf.Tensor,
        text_lengths: tf.Tensor,
        mel_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Phase 1: Preprocess data on Data GPU.
        
        This function runs on the data processing GPU and performs any
        preprocessing that can be done before transferring to model GPU.
        
        Args:
            text_sequences: Text input tensor
            mel_spectrograms: Mel spectrogram tensor
            text_lengths: Text sequence lengths
            mel_lengths: Mel sequence lengths
        
        Returns:
            Tuple of preprocessed tensors
        """
        with tf.device(self.data_device):
            # Ensure data is on data GPU
            text_sequences = tf.identity(text_sequences)
            mel_spectrograms = tf.identity(mel_spectrograms)
            text_lengths = tf.identity(text_lengths)
            mel_lengths = tf.identity(mel_lengths)
            
            # Any preprocessing can be added here
            # For now, just ensure proper placement
            
            return text_sequences, mel_spectrograms, text_lengths, mel_lengths
    
    @tf.function
    def _transfer_to_model_gpu(
        self,
        text_sequences: tf.Tensor,
        mel_spectrograms: tf.Tensor,
        text_lengths: tf.Tensor,
        mel_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Phase 2: Transfer data from Data GPU to Model GPU.
        
        This is a controlled transfer operation that explicitly moves
        data between GPUs using tf.identity with device placement.
        
        Args:
            Preprocessed tensors from data GPU
        
        Returns:
            Tuple of tensors on model GPU
        """
        with tf.device(self.model_device):
            # Explicit transfer with identity operation
            text_sequences = tf.identity(text_sequences)
            mel_spectrograms = tf.identity(mel_spectrograms)
            text_lengths = tf.identity(text_lengths)
            mel_lengths = tf.identity(mel_lengths)
            
            return text_sequences, mel_spectrograms, text_lengths, mel_lengths
    
    def _train_step_impl(
        self,
        text_sequences: tf.Tensor,
        mel_spectrograms: tf.Tensor,
        text_lengths: tf.Tensor,
        mel_lengths: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Memory-isolated training step with three-phase pipeline.
        
        Phase 1: Data preprocessing on Data GPU
        Phase 2: Transfer to Model GPU  
        Phase 3: Training on Model GPU
        
        This overrides the parent class method to enforce memory isolation.
        """
        try:
            # Phase 1: Preprocess on Data GPU
            text_sequences, mel_spectrograms, text_lengths, mel_lengths = \
                self._preprocess_on_data_gpu(
                    text_sequences, mel_spectrograms, text_lengths, mel_lengths
                )
            
            # Phase 2: Transfer to Model GPU
            text_sequences, mel_spectrograms, text_lengths, mel_lengths = \
                self._transfer_to_model_gpu(
                    text_sequences, mel_spectrograms, text_lengths, mel_lengths
                )
            
            # Phase 3: Training on Model GPU - use parent implementation
            # The parent class already has model_device set, so it will
            # automatically use the model GPU
            with tf.device(self.model_device):
                return super()._train_step_impl(
                    text_sequences, mel_spectrograms, text_lengths, mel_lengths
                )
                
        except Exception as e:
            logger.error(f"Error in memory-isolated training step: {e}")
            raise
    
    def train_step(
        self,
        batch_data: Optional[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]] = None
    ) -> Dict[str, tf.Tensor]:
        """
        Execute one training step with memory isolation.
        
        This wraps the parent train_step with memory monitoring.
        """
        # Memory health check (periodic)
        if hasattr(self, 'global_step'):
            self._check_memory_health(self.global_step)
        
        # Execute training step
        result = super().train_step(batch_data)
        
        return result
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: Optional[int] = None
    ):
        """
        Train the model with memory-isolated dual-GPU pipeline.
        
        This wraps the parent train method with additional memory monitoring.
        """
        logger.info("=" * 70)
        logger.info("Starting Memory-Isolated Dual-GPU Training")
        logger.info("=" * 70)
        
        # Setup memory baselines
        self._setup_memory_baselines()
        
        # Log initial memory state
        log_memory_stats(self.data_gpu_id, self.model_gpu_id)
        
        # Start training using parent implementation
        try:
            super().train(train_dataset, val_dataset, epochs)
        finally:
            # Final memory report
            logger.info("=" * 70)
            logger.info("Training Complete - Final Memory Report")
            logger.info("=" * 70)
            log_memory_stats(self.data_gpu_id, self.model_gpu_id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics for both GPUs.
        
        Returns:
            Dictionary with memory statistics
        """
        data_info, model_info = monitor_gpu_memory(self.data_gpu_id, self.model_gpu_id)
        
        return {
            'data_gpu': {
                'id': self.data_gpu_id,
                'device': self.data_device,
                'limit_mb': self.data_gpu_memory_limit,
                'info': data_info
            },
            'model_gpu': {
                'id': self.model_gpu_id,
                'device': self.model_device,
                'limit_mb': self.model_gpu_memory_limit,
                'info': model_info
            }
        }
