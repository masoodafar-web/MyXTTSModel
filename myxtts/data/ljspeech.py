"""
LJSpeech Dataset Loader for MyXTTS.

This module provides utilities to load and process the LJSpeech dataset
in a format compatible with the MyXTTS training pipeline.
"""

import os
import csv
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from ..utils.audio import AudioProcessor
from ..utils.text import TextProcessor
from ..config.config import DataConfig


class LJSpeechDataset:
    """
    LJSpeech dataset loader and processor.
    
    The LJSpeech dataset is a public domain speech dataset consisting of
    13,100 short audio clips of a single speaker reading passages from
    7 non-fiction books. The dataset is commonly used for TTS training.
    """
    
    URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    
    def __init__(
        self,
        data_path: str,
        config: DataConfig,
        subset: str = "train",
        download: bool = True,
        preprocess: bool = True
    ):
        """
        Initialize LJSpeech dataset.
        
        Args:
            data_path: Path to dataset directory
            config: Data configuration
            subset: Dataset subset ("train", "val", "test")
            download: Whether to download dataset if not present
            preprocess: Whether to preprocess audio files
        """
        self.data_path = Path(data_path)
        self.config = config
        self.subset = subset
        self.download_flag = download
        self.preprocess_flag = preprocess
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sample_rate=config.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            normalize=config.normalize_audio,
            trim_silence=config.trim_silence
        )
        
        self.text_processor = TextProcessor(
            language=config.language,
            cleaner_names=config.text_cleaners,
            add_blank=config.add_blank
        )
        
        # Dataset paths
        self.dataset_dir = self.data_path / "LJSpeech-1.1"
        self.wavs_dir = self.dataset_dir / "wavs"
        self.metadata_file = self.dataset_dir / "metadata.csv"
        
        # Processed data paths
        self.processed_dir = self.data_path / "processed"
        self.splits_file = self.processed_dir / "splits.json"
        
        # Load dataset
        self._prepare_dataset()
        self.metadata = self._load_metadata()
        
        # Create splits if they don't exist
        if not self.splits_file.exists():
            self._create_splits()
        
        self.splits = self._load_splits()
        self.items = self.splits[self.subset]
        
        print(f"Loaded {len(self.items)} items for {subset} subset")
    
    def _prepare_dataset(self):
        """Download and extract dataset if necessary."""
        if not self.dataset_dir.exists() and self.download_flag:
            print("Downloading LJSpeech dataset...")
            self._download_dataset()
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"LJSpeech dataset not found at {self.dataset_dir}. "
                "Set download=True to download automatically."
            )
        
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_dataset(self):
        """Download and extract LJSpeech dataset."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Download
        archive_path = self.data_path / "LJSpeech-1.1.tar.bz2"
        if not archive_path.exists():
            print(f"Downloading from {self.URL}...")
            urllib.request.urlretrieve(self.URL, archive_path)
        
        # Extract
        print("Extracting dataset...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(self.data_path)
        
        print("Dataset downloaded and extracted successfully")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata from CSV file."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        # LJSpeech metadata format: ID|transcription|normalized_transcription
        df = pd.read_csv(
            self.metadata_file,
            sep='|',
            header=None,
            names=['id', 'transcription', 'normalized_transcription'],
            quoting=csv.QUOTE_NONE
        )
        
        # Add audio file paths
        df['audio_path'] = df['id'].apply(
            lambda x: str(self.wavs_dir / f"{x}.wav")
        )
        
        # Verify audio files exist
        missing_files = []
        for idx, row in df.iterrows():
            if not Path(row['audio_path']).exists():
                missing_files.append(row['audio_path'])
        
        if missing_files:
            print(f"Warning: {len(missing_files)} audio files are missing")
            # Remove missing files from dataframe
            df = df[df['audio_path'].apply(lambda x: Path(x).exists())]
        
        return df
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        print("Creating dataset splits...")
        
        # Shuffle items
        indices = np.arange(len(self.metadata))
        np.random.shuffle(indices)
        
        # Calculate split sizes
        total_size = len(indices)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.val_split)
        
        # Create splits
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        splits = {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(), 
            "test": test_indices.tolist()
        }
        
        # Save splits
        with open(self.splits_file, "w") as f:
            json.dump(splits, f, indent=2)
        
        print(f"Created splits: train={len(train_indices)}, "
              f"val={len(val_indices)}, test={len(test_indices)}")
    
    def _load_splits(self) -> Dict[str, List[int]]:
        """Load dataset splits."""
        with open(self.splits_file, "r") as f:
            splits = json.load(f)
        return splits
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing:
                - text: Original text
                - text_normalized: Normalized text
                - text_sequence: Tokenized text sequence
                - audio_path: Path to audio file
                - mel_spectrogram: Mel spectrogram (if preprocessed)
                - audio_length: Audio length in samples
                - text_length: Text sequence length
        """
        item_idx = self.items[idx]
        row = self.metadata.iloc[item_idx]
        
        # Get text data
        text = row['transcription']
        text_normalized = row['normalized_transcription']
        
        # Process text
        text_sequence = self.text_processor.text_to_sequence(text_normalized)
        
        # Get audio path
        audio_path = row['audio_path']
        
        # Load and process audio
        audio = self.audio_processor.load_audio(audio_path)
        if self.config.normalize_audio:
            audio = self.audio_processor.preprocess_audio(audio)
        
        # Extract mel spectrogram
        mel_spec = self.audio_processor.wav_to_mel(audio)
        
        return {
            "id": row['id'],
            "text": text,
            "text_normalized": text_normalized,
            "text_sequence": np.array(text_sequence, dtype=np.int32),
            "audio_path": audio_path,
            "audio": audio.astype(np.float32),
            "mel_spectrogram": mel_spec.astype(np.float32),
            "audio_length": len(audio),
            "text_length": len(text_sequence),
            "mel_length": mel_spec.shape[1]
        }
    
    def get_sample_rate(self) -> int:
        """Get audio sample rate."""
        return self.config.sample_rate
    
    def get_vocab_size(self) -> int:
        """Get text vocabulary size."""
        return self.text_processor.get_vocab_size()
    
    def create_tf_dataset(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        repeat: bool = True,
        prefetch: bool = True
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.
        
        Args:
            batch_size: Batch size (uses config if None)
            shuffle: Whether to shuffle dataset
            repeat: Whether to repeat dataset
            prefetch: Whether to prefetch data
            
        Returns:
            TensorFlow dataset
        """
        batch_size = batch_size or self.config.batch_size
        
        def generator():
            """Data generator function."""
            indices = list(range(len(self)))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                item = self[idx]
                yield (
                    item["text_sequence"],
                    item["mel_spectrogram"],
                    item["text_length"],
                    item["mel_length"]
                )
        
        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.int32),  # text_sequence
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # mel_spectrogram
            tf.TensorSpec(shape=(), dtype=tf.int32),  # text_length
            tf.TensorSpec(shape=(), dtype=tf.int32)   # mel_length
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Pad and batch
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [None],      # text_sequence
                [None, None], # mel_spectrogram  
                [],          # text_length
                []           # mel_length
            ),
            padding_values=(
                0,    # text_sequence (pad token)
                0.0,  # mel_spectrogram
                0,    # text_length
                0     # mel_length
            )
        )
        
        if repeat:
            dataset = dataset.repeat()
        
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        print("Computing dataset statistics...")
        
        text_lengths = []
        audio_lengths = []
        mel_lengths = []
        
        for idx in tqdm(range(len(self)), desc="Computing statistics"):
            item = self[idx]
            text_lengths.append(item["text_length"])
            audio_lengths.append(item["audio_length"])
            mel_lengths.append(item["mel_length"])
        
        stats = {
            "total_samples": len(self),
            "text_length": {
                "min": np.min(text_lengths),
                "max": np.max(text_lengths),
                "mean": np.mean(text_lengths),
                "std": np.std(text_lengths)
            },
            "audio_length": {
                "min": np.min(audio_lengths),
                "max": np.max(audio_lengths), 
                "mean": np.mean(audio_lengths),
                "std": np.std(audio_lengths)
            },
            "mel_length": {
                "min": np.min(mel_lengths),
                "max": np.max(mel_lengths),
                "mean": np.mean(mel_lengths),
                "std": np.std(mel_lengths)
            },
            "audio_duration_hours": np.sum(audio_lengths) / self.config.sample_rate / 3600,
            "vocab_size": self.get_vocab_size(),
            "sample_rate": self.get_sample_rate()
        }
        
        return stats


def create_ljspeech_dataset(
    data_path: str,
    config: DataConfig,
    subset: str = "train",
    download: bool = True
) -> LJSpeechDataset:
    """
    Create LJSpeech dataset instance.
    
    Args:
        data_path: Path to dataset directory
        config: Data configuration
        subset: Dataset subset
        download: Whether to download if not present
        
    Returns:
        LJSpeech dataset instance
    """
    return LJSpeechDataset(
        data_path=data_path,
        config=config,
        subset=subset,
        download=download
    )