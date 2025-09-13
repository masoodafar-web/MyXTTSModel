#!/usr/bin/env python3
"""
Example script demonstrating NLLB-200 tokenizer integration with MyXTTS.

This example shows how to use the facebook/nllb-200-distilled-600M tokenizer
with the MyXTTS model for multilingual text-to-speech synthesis.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import ModelConfig, XTTSConfig
from myxtts.utils.text import (
    TextProcessor, 
    NLLBTokenizer, 
    TRANSFORMERS_AVAILABLE,
    get_supported_nllb_languages,
    get_nllb_language_code,
    is_nllb_language_supported
)


def demo_nllb_tokenizer():
    """Demonstrate NLLB tokenizer usage."""
    
    print("MyXTTS NLLB-200 Tokenizer Integration Demo")
    print("=" * 50)
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers library is not available.")
        print("Please install with: pip install transformers")
        return
    
    # Create configuration with NLLB tokenizer
    config = XTTSConfig()
    config.model.tokenizer_type = "nllb"
    config.model.tokenizer_model = "facebook/nllb-200-distilled-600M"
    config.model.text_vocab_size = 256_256  # NLLB-200 vocab size
    
    print(f"Model Configuration:")
    print(f"  Tokenizer Type: {config.model.tokenizer_type}")
    print(f"  Tokenizer Model: {config.model.tokenizer_model}")
    print(f"  Vocab Size: {config.model.text_vocab_size:,}")
    print()
    
    # Create text processors for comparison
    print("Creating Text Processors...")
    
    # Custom tokenizer
    custom_processor = TextProcessor(
        language="en",
        tokenizer_type="custom"
    )
    print(f"Custom tokenizer vocab size: {custom_processor.get_vocab_size()}")
    
    # NLLB tokenizer (will attempt to download model)
    try:
        nllb_processor = TextProcessor(
            language="en",
            tokenizer_type="nllb",
            tokenizer_model="facebook/nllb-200-distilled-600M"
        )
        print(f"NLLB tokenizer vocab size: {nllb_processor.get_vocab_size():,}")
        
        # Test text processing with language-specific tokenization
        sample_texts = [
            ("Hello, world!", "en"),
            ("This is a multilingual text-to-speech system.", "en"),
            ("¡Hola, mundo!", "es"),  # Spanish
            ("Bonjour le monde!", "fr"),  # French
            ("Hallo Welt!", "de"),  # German
            ("سلام دنیا!", "fa"),  # Persian/Farsi
        ]
        
        print("\nTokenization Comparison with Language Context:")
        print("-" * 50)
        
        for text, lang in sample_texts:
            print(f"\nText: '{text}' (Language: {lang})")
            print(f"NLLB code: {nllb_processor.nllb_tokenizer.get_nllb_code(lang)}")
            
            # Custom tokenizer
            custom_seq = custom_processor.text_to_sequence(text)
            print(f"Custom tokenizer:  {len(custom_seq):3d} tokens: {custom_seq[:10]}...")
            
            # NLLB tokenizer with language context
            nllb_processor.language = lang  # Set language context
            nllb_seq = nllb_processor.text_to_sequence(text)
            print(f"NLLB tokenizer:    {len(nllb_seq):3d} tokens: {nllb_seq[:10]}...")
        
        # Test batch processing
        print("\nBatch Processing:")
        print("-" * 20)
        
        batch_texts = [text for text, _ in sample_texts[:3]]
        nllb_sequences, nllb_lengths = nllb_processor.batch_text_to_sequence(batch_texts)
        print(f"Batch sequences shape: {nllb_sequences.shape}")
        print(f"Sequence lengths: {nllb_lengths}")
        
        # Display supported languages
        print("\nSupported NLLB Languages (sample):")
        print("-" * 35)
        supported_langs = nllb_processor.nllb_tokenizer.get_supported_languages()
        print(f"Total supported languages: {len(supported_langs)}")
        for i, lang in enumerate(supported_langs[:10]):
            nllb_code = nllb_processor.nllb_tokenizer.get_nllb_code(lang)
            print(f"  {lang} -> {nllb_code}")
        print("  ... and more")
        
        print("\nNLLB tokenizer integration with language standardization successful!")
        
    except Exception as e:
        print(f"Could not initialize NLLB tokenizer: {e}")
        print("This is expected if there's no internet connection to download the model.")
        print("\nHowever, the integration code is ready and will work when:")
        print("1. Internet connection is available")
        print("2. The model is downloaded")
        print("3. The tokenizer is properly initialized")


def demo_config_usage():
    """Demonstrate configuration with NLLB tokenizer."""
    
    print("\nConfiguration Usage Example:")
    print("=" * 30)
    
    # Create config with NLLB settings
    config = ModelConfig()
    config.tokenizer_type = "nllb"
    config.tokenizer_model = "facebook/nllb-200-distilled-600M"
    config.text_vocab_size = 256_256
    
    print("Configuration created with NLLB tokenizer:")
    print(f"  - tokenizer_type: {config.tokenizer_type}")
    print(f"  - tokenizer_model: {config.tokenizer_model}")
    print(f"  - text_vocab_size: {config.text_vocab_size:,}")
    
    # Show how to use in model creation (conceptual)
    print("\nTo use with XTTS model:")
    print("```python")
    print("from myxtts.models.xtts import XTTS")
    print("from myxtts.config.config import ModelConfig")
    print("")
    print("config = ModelConfig()")
    print("config.tokenizer_type = 'nllb'")
    print("config.text_vocab_size = 256_256")
    print("")
    print("model = XTTS(config)")
    print("```")


if __name__ == "__main__":
    demo_nllb_tokenizer()
    demo_config_usage()