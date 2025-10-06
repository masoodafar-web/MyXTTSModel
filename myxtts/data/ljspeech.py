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
import mmap
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..utils.audio import AudioProcessor
from ..utils.text import TextProcessor
from ..config.config import DataConfig


class _NullProfiler:
    """No-op profiler used when no external profiler is supplied."""

    def record_cache_hit(self) -> None:
        pass

    def record_cache_miss(self) -> None:
        pass

    def record_cache_error(self) -> None:
        pass

    def profile_operation(self, _name: str):
        return nullcontext()


class LJSpeechDataset:
    """
    Multi-speaker dataset loader and processor.
    
    Originally designed for LJSpeech (single speaker), now extended to support
    multi-speaker datasets with speaker identification. The dataset can handle:
    - Single speaker datasets (original LJSpeech format)
    - Multi-speaker datasets with speaker ID extraction from filename or metadata
    - Audio normalization including loudness matching and VAD processing
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
        
        # Simple in-memory cache for tokenized text to avoid recomputation
        self._text_cache: Dict[str, np.ndarray] = {}
        # Memory-mapped cache for mel spectrograms (faster than np.load)
        self._mel_mmap_cache: Dict[str, np.memmap] = {}
        # Optional index map when using custom metadata and cache filtering
        self._custom_index_map: Optional[List[int]] = None
        # Thread lock for cache access
        self._cache_lock = threading.RLock()
        
        # Multi-speaker support
        self.speaker_mapping: Dict[str, int] = {}  # speaker_id -> speaker_index
        self.reverse_speaker_mapping: Dict[int, str] = {}  # speaker_index -> speaker_id
        self.num_speakers = 0
        self.is_multispeaker = getattr(config, 'enable_multispeaker', False)
        self.speaker_id_pattern = getattr(config, 'speaker_id_pattern', None)  # regex pattern for speaker extraction
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sample_rate=config.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            normalize=config.normalize_audio,
            trim_silence=config.trim_silence,
            enable_loudness_normalization=getattr(config, 'enable_loudness_normalization', True),
            target_loudness_lufs=getattr(config, 'target_loudness_lufs', -23.0),
            enable_vad=getattr(config, 'enable_vad', True)
        )

        profiler = getattr(config, 'profiler', None)
        if profiler is None or not hasattr(profiler, 'profile_operation'):
            profiler = _NullProfiler()
        # Fail-safe to ensure required profiler hooks exist even if a partial implementation is passed in.
        for hook in ("record_cache_hit", "record_cache_miss", "record_cache_error"):
            if not hasattr(profiler, hook):
                profiler = _NullProfiler()
                break
        self.profiler = profiler

        # Dataset paths
        self.dataset_dir = self.data_path
        #/ "LJSpeech-1.1"
        self.wavs_dir = self.dataset_dir 
        #/ "wavs"
        
        # Determine if we're using custom metadata files with potentially different wav directories
        self.use_custom_metadata = (
            (subset == "train" and config.metadata_train_file) or
            (subset in ["val", "test"] and config.metadata_eval_file)
        )
        
        # Set metadata file path and wav directory based on subset and configuration
        self.metadata_file, self.wavs_dir = self._get_metadata_and_wavs_path(subset)
        
        # Processed data paths (only for splits and symbol map)
        self.processed_dir = self.data_path / "processed"
        self.splits_file = self.processed_dir / "splits.json"
        # Symbol map path for phoneme consistency
        tokenizer_tag = 'custom'
        phon_tag = 'ph' if getattr(config, 'use_phonemes', False) else 'noph'
        blank_tag = 'blank' if getattr(config, 'add_blank', True) else 'noblank'
        lang_tag = getattr(config, 'language', 'xx')
        symbol_dir = self.processed_dir / f"tokens_{tokenizer_tag}_{phon_tag}_{blank_tag}_{lang_tag}"
        self.symbol_map_path = symbol_dir / "symbols.json"

        # Load dataset
        self._prepare_dataset()
        self.metadata = self._load_metadata()

        # Determine symbol map before instantiating text processor
        custom_symbols = None
        allow_symbol_growth = True
        if self.symbol_map_path.exists():
            try:
                with open(self.symbol_map_path, "r", encoding="utf-8") as f:
                    custom_symbols = json.load(f)
                allow_symbol_growth = False
            except Exception as exc:
                print(f"Warning: Failed to load symbol map {self.symbol_map_path}: {exc}. Regenerating.")
                custom_symbols = None

        if custom_symbols is None and getattr(config, 'use_phonemes', False):
            if subset.lower() == 'train':
                custom_symbols = self._build_symbol_map()
                allow_symbol_growth = False
            else:
                print(
                    f"Warning: Symbol map not found at {self.symbol_map_path} for subset '{subset}'. "
                    "Using dynamic symbol growth; regenerate symbol map from training data for consistency."
                )

        self.text_processor = TextProcessor(
            language=config.language,
            cleaner_names=config.text_cleaners,
            add_blank=config.add_blank,
            use_phonemes=getattr(config, 'use_phonemes', False),
            phoneme_language=getattr(config, 'phoneme_language', None),
            custom_symbols=custom_symbols,
            allow_symbol_growth=allow_symbol_growth
        )

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

    def _save_npy_atomic(self, path: Path, array: np.ndarray) -> None:
        """Safely save numpy array atomically to avoid partial files."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = Path(str(path) + ".tmp")
            np.save(tmp_path, array)
            os.replace(tmp_path, path)
        except Exception:
            # Fall back to direct save if atomic replace fails
            np.save(path, array)



    def _load_tokens_optimized(self, text_normalized: str, cache_key: str) -> np.ndarray:
        """
        Load tokens with in-memory caching only (on-the-fly processing).
        
        Args:
            text_normalized: Normalized text for tokenization
            cache_key: Cache key for in-memory storage
            
        Returns:
            Token sequence array
        """
        with self._cache_lock:
            # Check in-memory cache first (fastest)
            if cache_key in self._text_cache:
                self.profiler.record_cache_hit()
                return self._text_cache[cache_key]
            
            # Tokenize on-the-fly
            self.profiler.record_cache_miss()
            with self.profiler.profile_operation("tokenization"):
                text_sequence = self.text_processor.text_to_sequence(text_normalized)
                text_sequence = np.array(text_sequence, dtype=np.int32)
            
            # Cache the result in memory only
            self._text_cache[cache_key] = text_sequence
            
            return text_sequence

    def _build_symbol_map(self) -> List[str]:
        """Generate a deterministic phoneme symbol map and persist it."""
        temp_processor = TextProcessor(
            language=self.config.language,
            cleaner_names=self.config.text_cleaners,
            add_blank=self.config.add_blank,
            use_phonemes=getattr(self.config, 'use_phonemes', False),
            phoneme_language=getattr(self.config, 'phoneme_language', None)
        )

        normalized_texts = list(self.metadata['normalized_transcription'])
        for text in normalized_texts:
            temp_processor.text_to_sequence(text)

        symbols = list(temp_processor.symbols)

        try:
            self.symbol_map_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.symbol_map_path, 'w', encoding='utf-8') as f:
                json.dump(symbols, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Warning: Failed to save symbol map to {self.symbol_map_path}: {exc}")

        return symbols

    def _get_metadata_and_wavs_path(self, subset: str) -> Tuple[Path, Path]:
        """
        Get metadata file path and wav directory based on subset and configuration.
        
        Args:
            subset: Dataset subset ("train", "val", "test")
            
        Returns:
            Tuple of (metadata_file_path, wavs_directory_path)
        """
        if subset == "train" and self.config.metadata_train_file:
            # Use custom train metadata file
            metadata_path = Path(self.config.metadata_train_file)
            if not metadata_path.is_absolute():
                # Relative path - make it relative to dataset directory
                metadata_path = self.dataset_dir / metadata_path
            
            # Use custom train wav directory if provided, otherwise use metadata file's directory
            if self.config.wavs_train_dir:
                wavs_path = Path(self.config.wavs_train_dir)
                if not wavs_path.is_absolute():
                    wavs_path = self.dataset_dir / wavs_path
            else:
                # Default to wavs directory in same location as metadata file
                wavs_path = metadata_path.parent / "wavs"
            
            return metadata_path, wavs_path
            
        elif subset in ["val", "test"] and self.config.metadata_eval_file:
            # Use custom eval metadata file for validation and test
            metadata_path = Path(self.config.metadata_eval_file)
            if not metadata_path.is_absolute():
                # Relative path - make it relative to dataset directory
                metadata_path = self.dataset_dir / metadata_path
            
            # Use custom eval wav directory if provided, otherwise use metadata file's directory
            if self.config.wavs_eval_dir:
                wavs_path = Path(self.config.wavs_eval_dir)
                if not wavs_path.is_absolute():
                    wavs_path = self.dataset_dir / wavs_path
            else:
                # Default to wavs directory in same location as metadata file
                wavs_path = metadata_path.parent / "wavs"
            
            return metadata_path, wavs_path
            
        else:
            # Use default metadata.csv file and wavs directory
            return self.dataset_dir / "metadata.csv", self.dataset_dir / "wavs"
    
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
    
    def _extract_speaker_id(self, audio_id: str) -> str:
        """
        Extract speaker ID from audio file ID.
        
        Args:
            audio_id: Audio file identifier
            
        Returns:
            Speaker ID string
        """
        if not self.is_multispeaker:
            return "single_speaker"
        
        if self.speaker_id_pattern:
            import re
            match = re.search(self.speaker_id_pattern, audio_id)
            if match:
                return match.group(1)
        
        # Default patterns for common multi-speaker dataset formats
        # Pattern 1: speaker_id_utterance_id (e.g., "p225_001", "VCTK_p225_001")
        if '_' in audio_id:
            parts = audio_id.split('_')
            # Look for speaker patterns (e.g., p225, spk001, speaker01)
            for part in parts:
                if part.startswith(('p', 'spk', 'speaker')) or part.isdigit():
                    return part
        
        # Pattern 2: directory-based extraction (if available in path)
        # This would require additional context, so fallback to full ID
        return audio_id.split('_')[0] if '_' in audio_id else "unknown_speaker"
    
    def _build_speaker_mapping(self, df: pd.DataFrame):
        """
        Build speaker ID to index mapping.
        
        Args:
            df: Metadata dataframe with speaker_id column
        """
        unique_speakers = sorted(df['speaker_id'].unique())
        self.speaker_mapping = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        self.reverse_speaker_mapping = {idx: speaker for speaker, idx in self.speaker_mapping.items()}
        self.num_speakers = len(unique_speakers)
        
        print(f"Found {self.num_speakers} unique speakers: {list(unique_speakers)}")
    
    def get_speaker_index(self, speaker_id: str) -> int:
        """Get speaker index from speaker ID."""
        return self.speaker_mapping.get(speaker_id, 0)
    
    def get_speaker_id(self, speaker_index: int) -> str:
        """Get speaker ID from speaker index."""
        return self.reverse_speaker_mapping.get(speaker_index, "unknown_speaker")
    
    def _compute_duration_target(self, text_sequence: np.ndarray, mel_spec: np.ndarray) -> np.ndarray:
        """
        Compute simple duration targets for text-to-mel alignment.
        
        Args:
            text_sequence: Text token sequence
            mel_spec: Mel spectrogram [time, n_mels]
            
        Returns:
            Duration targets [text_len] - number of mel frames per text token
        """
        text_len = len(text_sequence)
        mel_len = mel_spec.shape[0]
        
        if text_len == 0:
            return np.array([], dtype=np.float32)
        
        # Simple uniform distribution as baseline
        # In practice, this could be learned from forced alignment
        base_duration = mel_len / text_len
        duration_target = np.full(text_len, base_duration, dtype=np.float32)
        
        # Add some variation for more realistic targets
        # Make stop tokens shorter, content tokens longer
        for i, token in enumerate(text_sequence):
            # Simple heuristic: punctuation gets shorter duration
            if token in [0, 1, 2]:  # Assuming these are special tokens (PAD, UNK, EOS)
                duration_target[i] *= 0.5
            # Content tokens get slightly longer
            else:
                duration_target[i] *= 1.1
        
        # Normalize to ensure sum equals mel_len
        duration_target = duration_target * (mel_len / duration_target.sum())
        
        return duration_target
    
    def _apply_pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pitch shifting augmentation to audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Pitch-shifted audio
        """
        if not self.config.enable_pitch_shift:
            return audio
            
        try:
            import librosa
            # Random pitch shift within configured range
            shift_semitones = np.random.uniform(
                self.config.pitch_shift_range[0], 
                self.config.pitch_shift_range[1]
            )
            # Apply pitch shift
            audio_shifted = librosa.effects.pitch_shift(
                audio, 
                sr=self.config.sample_rate, 
                n_steps=shift_semitones
            )
            return audio_shifted
        except Exception as e:
            print(f"Warning: Pitch shift failed: {e}")
            return audio
    
    def _apply_noise_mixing(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise mixing augmentation to audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Audio with added noise
        """
        if not self.config.enable_noise_mixing:
            return audio
        
        # Apply noise mixing with configured probability
        if np.random.random() > self.config.noise_mixing_probability:
            return audio
            
        try:
            # Generate white noise
            noise = np.random.randn(len(audio))
            
            # Random SNR within configured range  
            target_snr_db = np.random.uniform(
                self.config.noise_mixing_snr_range[0],
                self.config.noise_mixing_snr_range[1]
            )
            
            # Calculate noise scaling factor
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(noise ** 2)
            snr_ratio = 10 ** (target_snr_db / 10)
            noise_scale = np.sqrt(signal_power / (noise_power * snr_ratio))
            
            # Mix signal with scaled noise
            mixed_audio = audio + noise_scale * noise
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val
                
            return mixed_audio
        except Exception as e:
            print(f"Warning: Noise mixing failed: {e}")
            return audio
    
    def _detect_language(self, text: str, audio_id: str) -> str:
        """
        Detect language from text or metadata.
        
        Args:
            text: Input text
            audio_id: Audio file identifier
            
        Returns:
            Detected language code
        """
        if not self.config.enable_multilingual:
            return self.config.language
            
        if self.config.language_detection_method == "metadata":
            # Try to extract language from metadata or filename
            # Common patterns: "en_001", "speaker_en_text", etc.
            for lang in self.config.supported_languages or ['en', 'es', 'fr', 'de', 'it']:
                if lang in audio_id.lower():
                    return lang
                    
        elif self.config.language_detection_method == "filename":
            # Extract language from filename pattern
            import re
            lang_match = re.search(r'[_-]([a-z]{2})[_-]', audio_id.lower())
            if lang_match:
                detected_lang = lang_match.group(1)
                if self.config.supported_languages is None or detected_lang in self.config.supported_languages:
                    return detected_lang
                    
        elif self.config.language_detection_method == "auto":
            # Simple heuristic-based language detection
            # This is a basic implementation - could be enhanced with proper language detection
            try:
                # Check for common language-specific characters
                if any(ord(c) > 127 for c in text):  # Non-ASCII characters
                    # Arabic script
                    if any('\u0600' <= c <= '\u06FF' for c in text):
                        return 'ar'
                    # Chinese characters
                    elif any('\u4e00' <= c <= '\u9fff' for c in text):
                        return 'zh'
                    # Japanese characters  
                    elif any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
                        return 'ja'
                    # Korean characters
                    elif any('\uac00' <= c <= '\ud7af' for c in text):
                        return 'ko'
                    # Cyrillic (Russian, etc.)
                    elif any('\u0400' <= c <= '\u04ff' for c in text):
                        return 'ru'
                
                # Basic European language detection by common words
                text_lower = text.lower()
                if any(word in text_lower for word in ['the', 'and', 'is', 'to', 'of']):
                    return 'en'
                elif any(word in text_lower for word in ['el', 'la', 'de', 'que', 'y']):
                    return 'es'
                elif any(word in text_lower for word in ['le', 'de', 'et', 'à', 'un']):
                    return 'fr'
                elif any(word in text_lower for word in ['der', 'die', 'und', 'ist', 'zu']):
                    return 'de'
                elif any(word in text_lower for word in ['il', 'di', 'e', 'che', 'la']):
                    return 'it'
                    
            except Exception:
                pass
                
        # Fallback to default language
        return self.config.language
    
    def _apply_phone_level_normalization(self, text: str, language: str) -> str:
        """
        Apply phone-level normalization to text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Phone-level normalized text
        """
        if not self.config.enable_phone_normalization:
            return text
            
        try:
            # Use the existing text processor with phoneme support
            if hasattr(self.text_processor, 'text_to_phonemes'):
                # Temporarily set language for phonemization
                original_lang = self.text_processor.language
                self.text_processor.language = language
                
                normalized_text = self.text_processor.text_to_phonemes(text)
                
                # Restore original language
                self.text_processor.language = original_lang
                
                return normalized_text
            else:
                # Fallback to regular text cleaning
                return self.text_processor.clean_text(text)
                
        except Exception as e:
            print(f"Warning: Phone normalization failed for language {language}: {e}")
            return text
    
    def _get_language_index(self, language: str) -> int:
        """
        Get language index for multi-language training.
        
        Args:
            language: Language code
            
        Returns:
            Language index
        """
        if not self.config.enable_multilingual:
            return 0
            
        # Initialize language mapping if not done
        if not hasattr(self, 'language_mapping'):
            supported_langs = self.config.supported_languages or ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'hu', 'ko']
            self.language_mapping = {lang: idx for idx, lang in enumerate(supported_langs)}
            self.num_languages = len(supported_langs)
        
        return self.language_mapping.get(language, 0)  # Default to 0 (usually English)
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata from CSV file robustly.

        Handles lines with extra '|' characters inside text by splitting on the
        first and last separator, skips obvious header lines, and ignores empty
        or malformed rows. Expects logical format: id|transcription|normalized.
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        rows = []
        skipped = 0
        header_skipped = 0

        with open(self.metadata_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue

                # Remove potential BOM and check for header-like first token
                candidate = line.lstrip('\ufeff')

                # Skip common header variants (e.g., "id|transcription|normalized_transcription")
                lower = candidate.lower()
                if lower.startswith('id|') or lower.startswith('fileid|') or lower.startswith('wav_id|'):
                    header_skipped += 1
                    continue

                # Use first and last '|' as field boundaries to be robust to extra '|' in middle
                first = candidate.find('|')
                last = candidate.rfind('|')

                id_, trans, norm = None, None, None
                if first == -1:
                    # Not enough separators; try splitting normally or skip
                    parts = candidate.split('|')
                    if len(parts) == 3:
                        id_, trans, norm = parts
                    elif len(parts) == 2:
                        id_, trans = parts
                        norm = trans
                    else:
                        skipped += 1
                        continue
                elif first == last:
                    # Only one separator; duplicate transcription as normalized
                    id_ = candidate[:first]
                    trans = candidate[first + 1:]
                    norm = trans
                else:
                    # Normal or with extra separators inside transcription
                    id_ = candidate[:first]
                    trans = candidate[first + 1:last]
                    norm = candidate[last + 1:]

                id_ = (id_ or '').strip()
                trans = (trans or '').strip()
                norm = (norm or '').strip()

                # Basic validation
                if not id_:
                    skipped += 1
                    continue

                # Extract speaker ID if multispeaker mode is enabled
                speaker_id = self._extract_speaker_id(id_)

                rows.append({
                    'id': id_,
                    'transcription': trans,
                    'normalized_transcription': norm,
                    'speaker_id': speaker_id
                })

        if header_skipped:
            print(f"Info: skipped {header_skipped} header line(s) in {self.metadata_file}")
        if skipped:
            print(f"Warning: skipped {skipped} malformed/empty metadata line(s)")

        df = pd.DataFrame(rows, columns=['id', 'transcription', 'normalized_transcription', 'speaker_id'])

        # Add audio file paths using the correct wavs directory
        df['audio_path'] = df['id'].apply(lambda x: str(self.wavs_dir / f"{x}.wav"))

        # Build speaker mapping if multispeaker mode
        if self.is_multispeaker:
            self._build_speaker_mapping(df)

        # Verify audio files exist
        missing_files = []
        for _, row in df.iterrows():
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
        if self.use_custom_metadata and self._custom_index_map is not None:
            return len(self._custom_index_map)
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get dataset item with optimized caching and minimal CPU overhead.
        
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
        with self.profiler.profile_operation("getitem_total"):
            if self.use_custom_metadata:
                # When using custom metadata, optionally use filtered index map
                if self._custom_index_map is not None:
                    item_idx = self._custom_index_map[idx]
                else:
                    item_idx = idx
            else:
                # When using single metadata with splits, use split indices
                item_idx = self.items[idx]
                
            row = self.metadata.iloc[item_idx]
            
            # Get text data
            text = row['transcription']
            text_normalized = row['normalized_transcription']
            
            # Multi-language support: detect language and apply phone-level normalization
            audio_id = str(row['id'])
            detected_language = self._detect_language(text, audio_id)
            
            # Apply phone-level normalization if enabled
            text_normalized = self._apply_phone_level_normalization(text_normalized, detected_language)
            
            # Process text with in-memory caching only (on-the-fly processing)
            cache_key = str(row['id'])
            
            with self.profiler.profile_operation("text_processing"):
                text_sequence = self._load_tokens_optimized(text_normalized, cache_key)
            
            # Get audio path
            audio_path = row['audio_path']
            
            # Process audio on-the-fly (no disk caching)
            with self.profiler.profile_operation("mel_computation"):
                audio = self.audio_processor.load_audio(audio_path)
                if self.config.normalize_audio:
                    audio = self.audio_processor.preprocess_audio(audio)
                
                # Apply audio augmentations during training
                if self.subset == "train":
                    audio = self._apply_pitch_shift(audio)
                    audio = self._apply_noise_mixing(audio)
                
                mel_spec = self.audio_processor.wav_to_mel(audio).T  # [time, n_mels]
        
        return {
            "id": row['id'],
            "text": text,
            "text_normalized": text_normalized,
            "text_sequence": text_sequence.astype(np.int32),
            "audio_path": audio_path,
            "audio": (audio.astype(np.float32) if audio is not None else np.array([], dtype=np.float32)),
            "mel_spectrogram": mel_spec.astype(np.float32),
            "audio_length": (len(audio) if audio is not None else int(mel_spec.shape[0] * self.audio_processor.hop_length)),
            "text_length": len(text_sequence),
            # Time dimension after transpose is axis 0
            "mel_length": int(mel_spec.shape[0]),
            # Multi-speaker support
            "speaker_id": row['speaker_id'],
            "speaker_index": self.get_speaker_index(row['speaker_id']) if self.is_multispeaker else 0,
            # Multi-language support (NEW)
            "language": detected_language,
            "language_index": self._get_language_index(detected_language),
            # Duration target (simple heuristic for now)
            "duration_target": self._compute_duration_target(text_sequence, mel_spec)
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
        prefetch: bool = True,
        use_cache_files: bool = True,
        memory_cache: bool = False,
        num_parallel_calls: Optional[int] = None,
        buffer_size_multiplier: int = 10,
        drop_remainder: bool = False,
    ) -> tf.data.Dataset:
        """
        Create optimized TensorFlow dataset with minimal CPU overhead.
        
        Args:
            batch_size: Batch size (uses config if None)
            shuffle: Whether to shuffle dataset
            repeat: Whether to repeat dataset
            prefetch: Whether to prefetch data
            use_cache_files: Whether to use cached files for faster loading
            memory_cache: Whether to cache data in memory
            num_parallel_calls: Number of parallel calls for map operations
            buffer_size_multiplier: Multiplier for shuffle buffer size
            
        Returns:
            Optimized TensorFlow dataset
        """
        batch_size = batch_size or self.config.batch_size
        
        # Use optimal number of parallel calls
        if num_parallel_calls is None:
            num_parallel_calls = min(self.config.num_workers * 2, tf.data.AUTOTUNE)
        
        # CRITICAL GPU OPTIMIZATION: Always use cache files for maximum GPU utilization
        # The else branch with py_function was causing the 4% GPU utilization issue
        if use_cache_files or getattr(self.config, 'use_tf_native_loading', True):
            # Force cache files usage when TF-native loading is enabled for GPU optimization
            # Build lists of cache file paths for current subset
            if self.use_custom_metadata and self._custom_index_map is not None:
                selected_indices = list(self._custom_index_map)
            elif self.use_custom_metadata:
                selected_indices = list(range(len(self.metadata)))
            else:
                selected_indices = list(self.items)

            token_paths = []
            mel_paths = []
            audio_paths = []
            norm_texts = []
            for i in selected_indices:
                row = self.metadata.iloc[i]
                sid = str(row['id'])
                token_paths.append(str(self.tokens_cache_dir / f"{sid}.npy"))
                mel_paths.append(str(self.mel_cache_dir / f"{sid}.npy"))
                audio_paths.append(str(row['audio_path']))
                norm_texts.append(str(row['normalized_transcription']))

            # Create dataset from file paths
            ds = tf.data.Dataset.from_tensor_slices((token_paths, mel_paths, audio_paths, norm_texts))

            # Shuffle early for better randomization
            if shuffle:
                shuffle_buffer_size = min(len(token_paths), max(1000, batch_size * buffer_size_multiplier))
                ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

            # Optimized loading using numpy to respect real .npy headers while staying inside tf.data
            max_tokens = int(getattr(self.config, 'max_text_tokens', 0) or 0)

            def _load_from_cache_numpy(tok_path, mel_path):
                tok_path = tok_path.decode('utf-8')
                mel_path = mel_path.decode('utf-8')
                try:
                    if os.path.exists(tok_path):
                        tokens = np.load(tok_path)
                    else:
                        tokens = np.zeros([1], dtype=np.int32)
                    if os.path.exists(mel_path):
                        mel = np.load(mel_path)
                    else:
                        mel = np.zeros([1, self.audio_processor.n_mels], dtype=np.float32)
                except Exception:
                    tokens = np.zeros([1], dtype=np.int32)
                    mel = np.zeros([1, self.audio_processor.n_mels], dtype=np.float32)
                if max_tokens > 0 and tokens.shape[0] > max_tokens:
                    tokens = tokens[:max_tokens]
                # Return scalar lengths to simplify batching and distribution
                text_len = np.int32(tokens.shape[0])

                mel_frames_cap = getattr(self.config, 'max_mel_frames', None)
                if mel_frames_cap and mel.shape[0] > mel_frames_cap:
                    mel = mel[:mel_frames_cap]
                mel_len = np.int32(mel.shape[0])
                return tokens.astype(np.int32), mel.astype(np.float32), text_len, mel_len

            def _load_from_cache_optimized(tok_path_t: tf.Tensor, mel_path_t: tf.Tensor,
                                          audio_path_t: tf.Tensor, norm_text_t: tf.Tensor):
                tokens, mel, text_len, mel_len = tf.numpy_function(
                    func=_load_from_cache_numpy,
                    inp=[tok_path_t, mel_path_t],
                    Tout=(tf.int32, tf.float32, tf.int32, tf.int32)
                )
                tokens.set_shape([None])
                mel.set_shape([None, self.audio_processor.n_mels])
                text_len.set_shape([])
                mel_len.set_shape([])
                # Already scalar lengths; return as-is
                return tokens, mel, text_len, mel_len

            # Map with parallel processing
            dataset = ds.map(
                _load_from_cache_optimized, 
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False  # Allow non-deterministic for better performance
            )

            # Optionally cap mel frames length to avoid OOM
            max_frames = getattr(self.config, 'max_mel_frames', None)
            if max_frames and max_frames > 0:
                def _cap_lengths(text_seq, mel_spec, text_len, mel_len):
                    new_mel = mel_spec[:max_frames]
                    new_mel_len = tf.minimum(mel_len, tf.constant(max_frames, dtype=mel_len.dtype))
                    return text_seq, new_mel, text_len, new_mel_len
                dataset = dataset.map(_cap_lengths, num_parallel_calls=tf.data.AUTOTUNE)
            
        else:
            # CRITICAL GPU FIX: Instead of falling back to slow py_function, warn and use TF-native
            import logging
            logger = logging.getLogger("MyXTTS")
            logger.warning("Cache files not found but using TF-native loading for GPU optimization. "
                          "Run precompute_mels() and precompute_tokens() first for best performance.")
            
            # Use same TF-native approach even without cache files (will create dummy data for missing files)
            if self.use_custom_metadata and self._custom_index_map is not None:
                selected_indices = list(self._custom_index_map)
            elif self.use_custom_metadata:
                selected_indices = list(range(len(self.metadata)))
            else:
                selected_indices = list(self.items)

            token_paths = []
            mel_paths = []
            audio_paths = []
            norm_texts = []
            
            for i in selected_indices:
                row = self.metadata.iloc[i]
                sid = str(row['id'])
                tok_path = str(self.tokens_cache_dir / f"{sid}.npy")
                mel_path = str(self.mel_cache_dir / f"{sid}.npy")
                audio_path = str(row['audio_path'])
                norm_text = str(row['normalized_text'])
                
                token_paths.append(tok_path)
                mel_paths.append(mel_path)
                audio_paths.append(audio_path) 
                norm_texts.append(norm_text)

            # Create dataset from paths
            ds = tf.data.Dataset.from_tensor_slices((token_paths, mel_paths, audio_paths, norm_texts))

            # Shuffle early for better randomization
            if shuffle:
                shuffle_buffer_size = min(len(token_paths), max(1000, batch_size * buffer_size_multiplier))
                ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

            # Use the same TF-native loading function
            dataset = ds.map(
                _load_from_cache_optimized, 
                num_parallel_calls=num_parallel_calls,
                deterministic=False  # Allow non-deterministic for better performance
            )

            # Optionally cap mel frames length to avoid OOM
            max_frames = getattr(self.config, 'max_mel_frames', None)
            if max_frames and max_frames > 0:
                def _cap_lengths(text_seq, mel_spec, text_len, mel_len):
                    new_mel = mel_spec[:max_frames]
                    new_mel_len = tf.minimum(mel_len, tf.constant(max_frames, dtype=mel_len.dtype))
                    return text_seq, new_mel, text_len, new_mel_len
                dataset = dataset.map(_cap_lengths, num_parallel_calls=tf.data.AUTOTUNE)

        # Apply memory caching if requested (use sparingly as it consumes RAM)
        if memory_cache:
            dataset = dataset.cache()

        # Filter out any invalid entries (empty sequences, etc.)
        dataset = dataset.filter(
            lambda text_seq, mel_spec, text_len, mel_len: 
            tf.logical_and(
                tf.greater(text_len, 0),
                tf.greater(mel_len, 0)
            )
        )

        # Pad and batch with improved padding strategy
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [None],       # text_sequence
                [None, self.audio_processor.n_mels], # mel_spectrogram
                [],           # text_length
                []            # mel_length
            ),
            padding_values=(
                tf.constant(0, dtype=tf.int32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32)
            ),
            drop_remainder=drop_remainder
        )

        # Repeat dataset for training
        if repeat:
            dataset = dataset.repeat()

        # Optimized prefetching strategy
        if prefetch:
            # Use config-driven prefetch buffer (reduces peak memory vs. batch_size//4)
            prefetch_buffer = max(1, int(getattr(self.config, 'prefetch_buffer_size', 2)))
            dataset = dataset.prefetch(prefetch_buffer)

        # CRITICAL GPU OPTIMIZATION: Advanced prefetching and CPU-GPU overlap
        try:
            # Check if enhanced GPU prefetching is enabled
            use_enhanced_prefetch = getattr(self.config, 'enhanced_gpu_prefetch', True)
            
            if use_enhanced_prefetch and bool(getattr(self.config, 'prefetch_to_gpu', True)):
                gpus = tf.config.list_logical_devices('GPU')
                # Single GPU training only - pin prefetch to GPU:0
                if gpus and len(gpus) > 0:
                    gpu_device = '/GPU:0'
                    
                    # Intelligent buffer sizing based on hardware and auto-tuning
                    base_buffer = getattr(self.config, 'prefetch_buffer_size', 8)
                    if getattr(self.config, 'auto_tune_performance', True):
                        # Auto-scale buffer based on number of workers and GPU count
                        worker_factor = max(1, self.config.num_workers // 8)
                        gpu_factor = len(gpus)
                        gpu_buf = max(6, min(20, base_buffer * worker_factor * gpu_factor))
                    else:
                        gpu_buf = max(4, int(base_buffer))
                    
                    # Apply GPU prefetching with automatic memory management
                    dataset = dataset.apply(
                        tf.data.experimental.prefetch_to_device(
                            gpu_device,
                            buffer_size=gpu_buf
                        )
                    )
                # If multi-GPU, keep host prefetching only (strategy will shard inputs)
                    
                    # Additional optimization: Enable GPU memory growth to avoid fragmentation
                    if getattr(self.config, 'optimize_cpu_gpu_overlap', True):
                        try:
                            physical_devices = tf.config.list_physical_devices('GPU')
                            if physical_devices:
                                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                        except Exception:
                            pass  # Ignore if already configured
                        
                    # Allow tf.data to further tune buffering on host side
                    dataset = dataset.prefetch(tf.data.AUTOTUNE)

        except Exception as e:
            print(f"Could not prefetch to GPU: {e}")

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # CRITICAL PERFORMANCE OPTIMIZATION: Enhanced TensorFlow data pipeline options
        options = tf.data.Options()
        
        # Core optimizations for GPU utilization
        options.experimental_deterministic = False  # Allow non-deterministic for better performance
        options.threading.private_threadpool_size = max(4, self.config.num_workers)
        options.threading.max_intra_op_parallelism = 1  # Avoid CPU oversubscription
        
        # Advanced pipeline optimizations
        if getattr(self.config, 'optimize_cpu_gpu_overlap', True):
            options.experimental_optimization.parallel_batch = True
            options.experimental_optimization.map_fusion = True
            options.experimental_optimization.map_parallelization = True
            options.experimental_optimization.filter_fusion = True
            options.experimental_optimization.filter_parallelization = True
            
            # GPU-specific optimizations
            try:
                options.experimental_optimization.map_vectorization.enabled = True
                options.experimental_optimization.map_vectorization.use_choose_fastest = True
            except AttributeError:
                pass  # Skip if not available in this TF version
                
            # Memory optimization
            options.experimental_optimization.apply_default_optimizations = True
            
            # Threading optimization for GPU workloads - more aggressive for better CPU utilization
            options.threading.private_threadpool_size = min(16, max(6, self.config.num_workers * 2))
            
            # Enhanced buffer management for sustained GPU feeding
            if hasattr(options.experimental_optimization, 'use_global_threadpool'):
                options.experimental_optimization.use_global_threadpool = True
        
        dataset = dataset.with_options(options)

        return dataset
    
    def cleanup_cache(self):
        """Clean up memory-mapped caches to free memory."""
        with self._cache_lock:
            # Clear memory-mapped cache
            for mmap_array in self._mel_mmap_cache.values():
                try:
                    if hasattr(mmap_array, '_mmap'):
                        mmap_array._mmap.close()
                except Exception:
                    pass
            self._mel_mmap_cache.clear()
            
            # Intelligent cache management - clear oldest entries if cache gets too large
            if len(self._text_cache) > 10000:  # Threshold for memory management
                # Keep the most recently used 5000 items (LRU-like behavior)
                items = list(self._text_cache.items())
                self._text_cache.clear()
                # Keep the last 5000 items (most recently added)
                for key, value in items[-5000:]:
                    self._text_cache[key] = value
    
    def __del__(self):
        """Cleanup resources when dataset is destroyed."""
        try:
            self.cleanup_cache()
        except Exception:
            pass

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
