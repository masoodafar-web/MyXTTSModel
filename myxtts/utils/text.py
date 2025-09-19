"""
Text processing utilities for MyXTTS.

This module provides text preprocessing, phonemization, and tokenization
functions compatible with multilingual text-to-speech synthesis.
"""

import re
import string
import unicodedata
from typing import List, Dict, Optional, Tuple
import numpy as np
from unidecode import unidecode

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    
try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Character sets for different languages
ENGLISH_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBERS = "0123456789"
PUNCTUATION = "!\"#%&'()*+,-./:;=?@[\\]_`{|}~"
SYMBOLS = " " + ENGLISH_CHARS + NUMBERS + PUNCTUATION

# Special tokens
PAD = "_"
EOS = "~"
BOS = "^"
SPECIAL_TOKENS = [PAD, BOS, EOS]

# Complete symbol set
ALL_SYMBOLS = SPECIAL_TOKENS + list(SYMBOLS)


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations in text."""
    abbreviations = {
        "mr.": "mister",
        "mrs.": "missus", 
        "dr.": "doctor",
        "prof.": "professor",
        "sr.": "senior",
        "jr.": "junior",
        "ltd.": "limited",
        "inc.": "incorporated",
        "co.": "company",
        "corp.": "corporation",
        "etc.": "etcetera",
        "vs.": "versus",
        "e.g.": "for example",
        "i.e.": "that is",
    }
    
    text_lower = text.lower()
    for abbrev, expansion in abbreviations.items():
        text_lower = text_lower.replace(abbrev, expansion)
    
    return text_lower


def expand_numbers(text: str) -> str:
    """Expand numbers to their spoken form (basic implementation)."""
    number_map = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
        "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
        "18": "eighteen", "19": "nineteen", "20": "twenty"
    }
    
    # Replace simple numbers
    words = text.split()
    for i, word in enumerate(words):
        # Remove punctuation for number checking
        clean_word = word.strip(string.punctuation)
        if clean_word.isdigit() and clean_word in number_map:
            words[i] = word.replace(clean_word, number_map[clean_word])
    
    return " ".join(words)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters."""
    # Normalize to NFD (decomposed) form
    text = unicodedata.normalize('NFD', text)
    # Remove combining characters (accents, etc.)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace characters."""
    # Replace multiple whitespaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def basic_cleaners(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower()
    text = remove_extra_whitespace(text)
    return text


def english_cleaners(text: str) -> str:
    """English text cleaning pipeline."""
    text = text.lower()
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = normalize_unicode(text)
    text = unidecode(text)  # Convert to ASCII
    text = remove_extra_whitespace(text)
    return text


def multilingual_cleaners(text: str) -> str:
    """Multilingual text cleaning pipeline."""
    text = text.lower()
    text = expand_abbreviations(text)
    text = normalize_unicode(text)
    text = remove_extra_whitespace(text)
    return text


# Text cleaners registry
text_cleaners = {
    "basic_cleaners": basic_cleaners,
    "english_cleaners": english_cleaners,
    "multilingual_cleaners": multilingual_cleaners,
}


class TextProcessor:
    """
    Text processing class for MyXTTS.
    
    Handles text cleaning, phonemization, and tokenization for multilingual TTS.
    Supports both custom symbol-based tokenization and NLLB-200 tokenizer.
    """
    
    def __init__(
        self,
        language: str = "en",
        cleaner_names: List[str] = None,
        add_blank: bool = True,
        use_phonemes: bool = True,
        custom_symbols: Optional[List[str]] = None,
        tokenizer_type: str = "custom",
        tokenizer_model: str = "facebook/nllb-200-distilled-600M"
    ):
        """
        Initialize TextProcessor.
        
        Args:
            language: Language code (e.g., "en", "es", "fr")
            cleaner_names: List of text cleaner function names
            add_blank: Whether to add blank tokens between characters (custom tokenizer only)
            use_phonemes: Whether to use phonemes instead of characters (custom tokenizer only)
            custom_symbols: Custom symbol set (None = use default, custom tokenizer only)
            tokenizer_type: Type of tokenizer ("custom" or "nllb")
            tokenizer_model: HuggingFace model name for NLLB tokenizer
        """
        self.language = language
        self.cleaner_names = cleaner_names or ["english_cleaners"]
        self.add_blank = add_blank
        self.use_phonemes = use_phonemes
        self.tokenizer_type = tokenizer_type
        self.tokenizer_model = tokenizer_model
        
        if self.tokenizer_type == "nllb":
            # Initialize NLLB tokenizer
            self.nllb_tokenizer = NLLBTokenizer(tokenizer_model)
            self.symbols = None  # Not used with NLLB
            self.symbol_to_id = None
            self.id_to_symbol = None
        else:
            # Initialize custom tokenizer
            self.nllb_tokenizer = None
            
            # Set up symbol set
            if custom_symbols is not None:
                self.symbols = custom_symbols
            else:
                self.symbols = ALL_SYMBOLS
            
            # Create symbol to ID mapping
            self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
            self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}
        
        # Initialize phonemizer if available and requested (custom tokenizer only)
        self.phonemizer = None
        if self.tokenizer_type == "custom" and self.use_phonemes and PHONEMIZER_AVAILABLE:
            try:
                self.phonemizer = EspeakBackend(
                    language=self.language,
                    preserve_punctuation=True,
                    with_stress=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize phonemizer for {language}: {e}")
                self.use_phonemes = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean text using specified cleaners.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        for cleaner_name in self.cleaner_names:
            if cleaner_name in text_cleaners:
                text = text_cleaners[cleaner_name](text)
            else:
                print(f"Warning: Unknown cleaner '{cleaner_name}'")
        
        return text
    
    def text_to_phonemes(self, text: str) -> str:
        """
        Convert text to phonemes (custom tokenizer only).
        
        Args:
            text: Input text
            
        Returns:
            Phonemized text
        """
        if not self.use_phonemes or self.phonemizer is None:
            return text
        
        try:
            phonemes = self.phonemizer.phonemize([text], strip=True)[0]
            return phonemes
        except Exception as e:
            print(f"Warning: Phonemization failed: {e}")
            return text
    
    def text_to_sequence(
        self, 
        text: str, 
        cleanup: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Convert text to sequence of token IDs.
        
        Args:
            text: Input text
            cleanup: Whether to clean text
            max_length: Maximum sequence length (NLLB tokenizer only)
            
        Returns:
            List of token IDs
        """
        if cleanup:
            text = self.clean_text(text)
        
        if self.tokenizer_type == "nllb":
            # Use NLLB tokenizer with language context
            self.nllb_tokenizer.set_language(self.language)
            return self.nllb_tokenizer.text_to_sequence(text, max_length=max_length)
        else:
            # Use custom tokenizer
            if self.use_phonemes:
                text = self.text_to_phonemes(text)
            
            sequence = []
            
            # Add BOS token
            sequence.append(self.symbol_to_id[BOS])
            
            for char in text:
                if char in self.symbol_to_id:
                    sequence.append(self.symbol_to_id[char])
                    # Add blank token between characters
                    if self.add_blank and char != ' ':
                        sequence.append(self.symbol_to_id[PAD])
                else:
                    # Skip unknown characters
                    print(f"Warning: Unknown character '{char}' (ord: {ord(char)})")
            
            # Add EOS token
            sequence.append(self.symbol_to_id[EOS])
            
            return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert sequence of token IDs back to text.
        
        Args:
            sequence: List of symbol IDs
            
        Returns:
            Decoded text
        """
        if self.tokenizer_type == "nllb":
            return self.nllb_tokenizer.sequence_to_text(sequence)
        else:
            symbols = []
            for symbol_id in sequence:
                if symbol_id in self.id_to_symbol:
                    symbol = self.id_to_symbol[symbol_id]
                    if symbol not in [PAD, BOS, EOS]:
                        symbols.append(symbol)
            
            return "".join(symbols)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.tokenizer_type == "nllb":
            return self.nllb_tokenizer.get_vocab_size()
        else:
            return len(self.symbols)
    
    def pad_sequences(
        self, 
        sequences: List[List[int]], 
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad sequences to same length.
        
        Args:
            sequences: List of sequences
            max_length: Maximum length (None = use longest sequence)
            
        Returns:
            Tuple of (padded_sequences, lengths)
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = np.zeros((len(sequences), max_length), dtype=np.int32)
        lengths = np.zeros(len(sequences), dtype=np.int32)
        
        # Determine pad token ID based on tokenizer type
        if self.tokenizer_type == "nllb":
            pad_token_id = self.nllb_tokenizer.pad_token_id or 0
        else:
            pad_token_id = self.symbol_to_id[PAD]
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]
            # Fill the rest with pad tokens
            padded[i, length:] = pad_token_id
            lengths[i] = length
        
        return padded, lengths
    
    def batch_text_to_sequence(
        self, 
        texts: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert batch of texts to padded sequences.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (padded_sequences, lengths)
        """
        if self.tokenizer_type == "nllb":
            return self.nllb_tokenizer.batch_text_to_sequence(texts, max_length=max_length)
        else:
            sequences = [self.text_to_sequence(text) for text in texts]
            return self.pad_sequences(sequences, max_length)


# NLLB-200 language codes mapping
# Maps ISO 639-1 language codes to NLLB-200 standardized codes
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",  # English
    "es": "spa_Latn",  # Spanish
    "fr": "fra_Latn",  # French
    "de": "deu_Latn",  # German
    "it": "ita_Latn",  # Italian
    "pt": "por_Latn",  # Portuguese
    "ru": "rus_Cyrl",  # Russian
    "zh": "zho_Hans",  # Chinese (Simplified)
    "ja": "jpn_Jpan",  # Japanese
    "ko": "kor_Hang",  # Korean
    "ar": "arb_Arab",  # Arabic
    "hi": "hin_Deva",  # Hindi
    "tr": "tur_Latn",  # Turkish
    "pl": "pol_Latn",  # Polish
    "nl": "nld_Latn",  # Dutch
    "sv": "swe_Latn",  # Swedish
    "da": "dan_Latn",  # Danish
    "no": "nob_Latn",  # Norwegian
    "fi": "fin_Latn",  # Finnish
    "cs": "ces_Latn",  # Czech
    "hu": "hun_Latn",  # Hungarian
    "he": "heb_Hebr",  # Hebrew
    "th": "tha_Thai",  # Thai
    "vi": "vie_Latn",  # Vietnamese
    "uk": "ukr_Cyrl",  # Ukrainian
    "bg": "bul_Cyrl",  # Bulgarian
    "hr": "hrv_Latn",  # Croatian
    "sk": "slk_Latn",  # Slovak
    "sl": "slv_Latn",  # Slovenian
    "et": "est_Latn",  # Estonian
    "lv": "lvs_Latn",  # Latvian
    "lt": "lit_Latn",  # Lithuanian
    "ro": "ron_Latn",  # Romanian
    "mt": "mlt_Latn",  # Maltese
    "is": "isl_Latn",  # Icelandic
    "ga": "gle_Latn",  # Irish
    "cy": "cym_Latn",  # Welsh
    "eu": "eus_Latn",  # Basque
    "ca": "cat_Latn",  # Catalan
    "gl": "glg_Latn",  # Galician
    "fa": "pes_Arab",  # Persian/Farsi
    "ur": "urd_Arab",  # Urdu
    "bn": "ben_Beng",  # Bengali
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "ml": "mal_Mlym",  # Malayalam
    "kn": "kan_Knda",  # Kannada
    "gu": "guj_Gujr",  # Gujarati
    "pa": "pan_Guru",  # Punjabi
    "mr": "mar_Deva",  # Marathi
    "ne": "npi_Deva",  # Nepali
    "si": "sin_Sinh",  # Sinhala
    "my": "mya_Mymr",  # Burmese
    "km": "khm_Khmr",  # Khmer
    "lo": "lao_Laoo",  # Lao
    "ka": "kat_Geor",  # Georgian
    "hy": "hye_Armn",  # Armenian
    "az": "azj_Latn",  # Azerbaijani
    "kk": "kaz_Cyrl",  # Kazakh
    "ky": "kir_Cyrl",  # Kyrgyz
    "uz": "uzn_Latn",  # Uzbek
    "tg": "tgk_Cyrl",  # Tajik
    "mn": "khk_Cyrl",  # Mongolian
    "af": "afr_Latn",  # Afrikaans
    "sq": "als_Latn",  # Albanian
    "am": "amh_Ethi",  # Amharic
    "as": "asm_Beng",  # Assamese
    "be": "bel_Cyrl",  # Belarusian
    "bs": "bos_Latn",  # Bosnian
    "br": "bre_Latn",  # Breton
    "dv": "div_Thaa",  # Divehi
    "fo": "fao_Latn",  # Faroese
    "fy": "fry_Latn",  # Frisian
    "gd": "gla_Latn",  # Scottish Gaelic
    "ha": "hau_Latn",  # Hausa
    "ig": "ibo_Latn",  # Igbo
    "id": "ind_Latn",  # Indonesian
    "jv": "jav_Latn",  # Javanese
    "ku": "ckb_Arab",  # Kurdish
    "lb": "ltz_Latn",  # Luxembourgish
    "mk": "mkd_Cyrl",  # Macedonian
    "mg": "plt_Latn",  # Malagasy
    "ms": "zsm_Latn",  # Malay
    "or": "ory_Orya",  # Odia
    "ps": "pbt_Arab",  # Pashto
    "qu": "quy_Latn",  # Quechua
    "rw": "kin_Latn",  # Kinyarwanda
    "sa": "san_Deva",  # Sanskrit
    "sd": "snd_Arab",  # Sindhi
    "so": "som_Latn",  # Somali
    "su": "sun_Latn",  # Sundanese
    "sw": "swh_Latn",  # Swahili
    "tl": "tgl_Latn",  # Tagalog
    "tt": "tat_Cyrl",  # Tatar
    "ti": "tir_Ethi",  # Tigrinya
    "tk": "tuk_Latn",  # Turkmen
    "tw": "twi_Latn",  # Twi
    "ug": "uig_Arab",  # Uyghur
    "wo": "wol_Latn",  # Wolof
    "xh": "xho_Latn",  # Xhosa
    "yi": "yid_Hebr",  # Yiddish
    "yo": "yor_Latn",  # Yoruba
    "zu": "zul_Latn",  # Zulu
}


def get_nllb_language_code(language: str) -> str:
    """
    Get NLLB-200 standardized language code from ISO language code.
    
    Args:
        language: ISO 639-1 language code (e.g., "en", "fr", "de")
        
    Returns:
        NLLB-200 language code (e.g., "eng_Latn", "fra_Latn", "deu_Latn")
        Falls back to "eng_Latn" if language not found.
    """
    return NLLB_LANGUAGE_CODES.get(language, "eng_Latn")


def get_supported_nllb_languages() -> List[str]:
    """
    Get list of ISO language codes supported by NLLB mapping.
    
    Returns:
        List of ISO 639-1 language codes supported by NLLB-200
    """
    return list(NLLB_LANGUAGE_CODES.keys())


def is_nllb_language_supported(language: str) -> bool:
    """
    Check if a language is supported by NLLB mapping.
    
    Args:
        language: ISO 639-1 language code
        
    Returns:
        True if language is supported, False otherwise
    """
    return language in NLLB_LANGUAGE_CODES


def get_language_phonemizer(language: str) -> Optional[str]:
    """
    Get appropriate phonemizer backend for language.
    
    Args:
        language: Language code
        
    Returns:
        Phonemizer backend name or None
    """
    language_map = {
        "en": "en-us",
        "es": "es",
        "fr": "fr-fr", 
        "de": "de",
        "it": "it",
        "pt": "pt",
        "pl": "pl",
        "tr": "tr",
        "ru": "ru",
        "nl": "nl",
        "cs": "cs",
        "ar": "ar",
        "zh": "cmn",
        "ja": "ja",
        "hu": "hu",
        "ko": "ko",
    }
    
    return language_map.get(language)


def setup_nltk():
    """Download required NLTK data."""
    if NLTK_AVAILABLE:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")


class NLLBTokenizer:
    """
    NLLB-200 tokenizer wrapper for MyXTTS.
    
    Provides a compatible interface for using the facebook/nllb-200-distilled-600M
    tokenizer with the MyXTTS model architecture.
    """
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """
        Initialize NLLB tokenizer.
        
        Args:
            model_name: HuggingFace model name for NLLB tokenizer
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for NLLB tokenizer. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Cache vocab size
        self._vocab_size = len(self.tokenizer)
        
        # Special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        
        # Current language for NLLB context
        self.current_language = None
    
    def set_language(self, language: str):
        """
        Set the language context for NLLB tokenization.
        
        Args:
            language: ISO language code (e.g., "en", "fr", "de")
        """
        nllb_code = get_nllb_language_code(language)
        self.current_language = nllb_code
        
        # Set the source language for NLLB tokenizer
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = nllb_code
    
    def encode(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        language: Optional[str] = None
    ) -> List[int]:
        """
        Encode text to token IDs with optional language context.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            return_tensors: Format for returned tensors ("tf", "pt", or None)
            language: Optional ISO language code for this specific encoding
            
        Returns:
            List of token IDs or tensor if return_tensors specified
        """
        # Set language context if provided
        if language is not None:
            self.set_language(language)
        
        encoded = self.tokenizer.encode(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        if return_tensors is None:
            return encoded
        else:
            return encoded
    
    def decode(
        self, 
        token_ids: List[int], 
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict:
        """
        Encode batch of texts with optional language context.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            return_tensors: Format for returned tensors
            language: Optional ISO language code for this batch
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        # Set language context if provided
        if language is not None:
            self.set_language(language)
            
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def text_to_sequence(
        self, 
        text: str, 
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Convert text to sequence compatible with MyXTTS.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        return self.encode(text, max_length=max_length, truncation=True)
    
    def batch_text_to_sequence(
        self,
        texts: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert batch of texts to padded sequences.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (padded_sequences, lengths)
        """
        encoded = self.batch_encode(
            texts, 
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        sequences = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # Calculate actual lengths (excluding padding)
        lengths = np.sum(attention_mask, axis=1)
        
        return sequences, lengths
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert sequence of token IDs back to text.
        
        Args:
            sequence: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.decode(sequence)
    
    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/nllb-200-distilled-600M") -> 'NLLBTokenizer':
        """
        Create tokenizer from pretrained model.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            NLLBTokenizer instance
        """
        return cls(model_name)
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of ISO language codes supported by NLLB mapping.
        
        Returns:
            List of ISO 639-1 language codes
        """
        return list(NLLB_LANGUAGE_CODES.keys())
    
    def get_nllb_code(self, language: str) -> str:
        """
        Get NLLB code for ISO language code.
        
        Args:
            language: ISO language code
            
        Returns:
            NLLB language code
        """
        return get_nllb_language_code(language)