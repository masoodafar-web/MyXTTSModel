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
        
        # Determine if we're using custom metadata files with potentially different wav directories
        self.use_custom_metadata = (
            (subset == "train" and config.metadata_train_file) or
            (subset in ["val", "test"] and config.metadata_eval_file)
        )
        
        # First determine dataset directory and metadata paths - this is done together now
        self.dataset_dir, self.metadata_file, self.wavs_dir = self._determine_dataset_paths(subset)
        
        # Processed data paths
        self.processed_dir = self.data_path / "processed"
        self.splits_file = self.processed_dir / "splits.json"
        
        # Load dataset
        self._prepare_dataset()
        self.metadata = self._load_metadata()
        
        # Handle splits differently based on whether custom metadata files are used
        if self.use_custom_metadata:
            # When using custom metadata files, each subset uses its own metadata file directly
            # No need for splits.json as train/eval data come from different files
            self.items = list(range(len(self.metadata)))
        else:
            # Default behavior: use single metadata file with percentage-based splits
            # Create splits if they don't exist
            if not self.splits_file.exists():
                self._create_splits()
            
            self.splits = self._load_splits()
            self.items = self.splits[self.subset]
        
        print(f"Loaded {len(self.items)} items for {subset} subset")
    
    def _determine_dataset_paths(self, subset: str) -> Tuple[Path, Path, Path]:
        """
        Determine dataset directory, metadata file, and wavs directory paths.
        
        This supports both:
        1. Custom datasets where files are directly in the provided path
        2. Downloaded LJSpeech datasets in the LJSpeech-1.1 subdirectory
        3. Custom metadata files with potentially different wav directories
        
        Args:
            subset: Dataset subset ("train", "val", "test")
            
        Returns:
            Tuple of (dataset_directory, metadata_file_path, wavs_directory_path)
        """
        # Handle custom metadata files first
        if subset == "train" and self.config.metadata_train_file:
            # Use custom train metadata file
            metadata_path = Path(self.config.metadata_train_file)
            if not metadata_path.is_absolute():
                # Try relative to data_path first
                if (self.data_path / metadata_path).exists():
                    metadata_path = self.data_path / metadata_path
                else:
                    # Fall back to LJSpeech-1.1 subdirectory for backward compatibility
                    metadata_path = self.data_path / "LJSpeech-1.1" / metadata_path
            
            # Use custom train wav directory if provided, otherwise use metadata file's directory
            if self.config.wavs_train_dir:
                wavs_path = Path(self.config.wavs_train_dir)
                if not wavs_path.is_absolute():
                    if (self.data_path / wavs_path).exists():
                        wavs_path = self.data_path / wavs_path
                    else:
                        wavs_path = self.data_path / "LJSpeech-1.1" / wavs_path
            else:
                # Default to wavs directory in same location as metadata file
                wavs_path = metadata_path.parent / "wavs"
            
            dataset_dir = metadata_path.parent
            return dataset_dir, metadata_path, wavs_path
            
        elif subset in ["val", "test"] and self.config.metadata_eval_file:
            # Use custom eval metadata file for validation and test
            metadata_path = Path(self.config.metadata_eval_file)
            if not metadata_path.is_absolute():
                # Try relative to data_path first
                if (self.data_path / metadata_path).exists():
                    metadata_path = self.data_path / metadata_path
                else:
                    # Fall back to LJSpeech-1.1 subdirectory for backward compatibility
                    metadata_path = self.data_path / "LJSpeech-1.1" / metadata_path
            
            # Use custom eval wav directory if provided, otherwise use metadata file's directory
            if self.config.wavs_eval_dir:
                wavs_path = Path(self.config.wavs_eval_dir)
                if not wavs_path.is_absolute():
                    if (self.data_path / wavs_path).exists():
                        wavs_path = self.data_path / wavs_path
                    else:
                        wavs_path = self.data_path / "LJSpeech-1.1" / wavs_path
            else:
                # Default to wavs directory in same location as metadata file
                wavs_path = metadata_path.parent / "wavs"
            
            dataset_dir = metadata_path.parent
            return dataset_dir, metadata_path, wavs_path
            
        else:
            # Handle default behavior - check if dataset files exist directly in provided path
            direct_metadata = self.data_path / "metadata.csv"
            direct_wavs = self.data_path / "wavs"
            
            if direct_metadata.exists() and direct_wavs.exists():
                # Dataset files found directly in provided path
                return self.data_path, direct_metadata, direct_wavs
            
            # Fall back to the traditional LJSpeech-1.1 subdirectory structure
            ljspeech_dir = self.data_path / "LJSpeech-1.1"
            return ljspeech_dir, ljspeech_dir / "metadata.csv", ljspeech_dir / "wavs"
    
    def _prepare_dataset(self):
        """Download and extract dataset if necessary."""
        # Check if we have the essential files rather than just the directory
        essential_files_exist = self._check_essential_files_exist()
        
        if not essential_files_exist and self.download_flag:
            print("Dataset files not found. Downloading LJSpeech dataset...")
            self._download_dataset()
            # Re-determine paths after download
            self.dataset_dir, self.metadata_file, self.wavs_dir = self._determine_dataset_paths(self.subset)
        
        # Final check that essential files exist
        if not self._check_essential_files_exist():
            missing_files = []
            if not self.metadata_file.exists():
                missing_files.append(str(self.metadata_file))
            if not self.wavs_dir.exists():
                missing_files.append(str(self.wavs_dir))
            
            raise FileNotFoundError(
                f"Dataset files not found: {missing_files}. "
                f"Expected metadata at {self.metadata_file} and wavs at {self.wavs_dir}. "
                "Set download=True to download LJSpeech automatically, or provide correct dataset paths."
            )
        
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_essential_files_exist(self) -> bool:
        """
        Check if the essential dataset files exist.
        
        Returns:
            True if both metadata file and wavs directory exist, False otherwise
        """
        return self.metadata_file.exists() and self.wavs_dir.exists()
    
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
        
        # Add audio file paths using the correct wavs directory
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
        if self.use_custom_metadata:
            # When using custom metadata, use direct indexing
            item_idx = idx
        else:
            # When using single metadata with splits, use split indices
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