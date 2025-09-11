"""Tests for NLLB tokenizer integration."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from myxtts.utils.text import TextProcessor, NLLBTokenizer, TRANSFORMERS_AVAILABLE
from myxtts.config.config import ModelConfig


class MockTokenizer:
    """Mock tokenizer that simulates NLLB behavior without requiring downloads."""
    
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self._vocab_size = 256_256
    
    def __len__(self):
        return self._vocab_size
    
    def encode(self, text, max_length=None, padding=False, truncation=True, return_tensors=None):
        # Simple mock encoding: convert text to list of fake token IDs
        tokens = [self.bos_token_id] + [ord(c) % 1000 + 10 for c in text] + [self.eos_token_id]
        
        if max_length and truncation:
            tokens = tokens[:max_length]
        
        if return_tensors == "np":
            return np.array(tokens)
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Simple mock decoding
        filtered_ids = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.pad_token_id, self.eos_token_id, self.bos_token_id]:
                continue
            filtered_ids.append(token_id)
        
        # Convert back to characters (mock)
        return "".join(chr((token_id - 10) % 128 + 32) for token_id in filtered_ids if token_id > 10)
    
    def __call__(self, texts, max_length=None, padding=True, truncation=True, return_tensors=None):
        """Batch encoding."""
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = [self.encode(text, max_length, padding=False, truncation=truncation) for text in texts]
        
        if padding and max_length:
            max_len = max_length
        else:
            max_len = max(len(seq) for seq in encoded) if encoded else 0
        
        # Pad sequences
        padded = []
        attention_mask = []
        for seq in encoded:
            padded_seq = seq + [self.pad_token_id] * (max_len - len(seq))
            mask = [1] * len(seq) + [0] * (max_len - len(seq))
            padded.append(padded_seq)
            attention_mask.append(mask)
        
        result = {
            "input_ids": np.array(padded) if return_tensors == "np" else padded,
            "attention_mask": np.array(attention_mask) if return_tensors == "np" else attention_mask
        }
        
        return result


class TestNLLBIntegration(unittest.TestCase):
    """Test NLLB tokenizer integration with mock tokenizer."""
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers library not available")
    @patch('myxtts.utils.text.AutoTokenizer')
    def test_nllb_tokenizer_mock(self, mock_auto_tokenizer):
        """Test NLLBTokenizer with mock to avoid network calls."""
        
        # Setup mock
        mock_tokenizer = MockTokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Create NLLB tokenizer
        nllb_tokenizer = NLLBTokenizer("facebook/nllb-200-distilled-600M")
        
        # Test basic properties
        self.assertEqual(nllb_tokenizer.get_vocab_size(), 256_256)
        self.assertIsNotNone(nllb_tokenizer.pad_token_id)
        self.assertIsNotNone(nllb_tokenizer.eos_token_id)
        
        # Test encoding
        text = "Hello, world!"
        tokens = nllb_tokenizer.encode(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Test decoding
        decoded = nllb_tokenizer.decode(tokens)
        self.assertIsInstance(decoded, str)
        
        # Test batch encoding
        texts = ["Hello", "World"]
        batch_result = nllb_tokenizer.batch_encode(texts, max_length=20, return_tensors="np")
        
        self.assertIn("input_ids", batch_result)
        self.assertIn("attention_mask", batch_result)
        self.assertEqual(batch_result["input_ids"].shape[0], 2)  # batch size
        
        # Verify mock was called
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("facebook/nllb-200-distilled-600M")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers library not available")
    @patch('myxtts.utils.text.AutoTokenizer')
    def test_text_processor_with_mock_nllb(self, mock_auto_tokenizer):
        """Test TextProcessor with mocked NLLB tokenizer."""
        
        # Setup mock
        mock_tokenizer = MockTokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Create text processor with NLLB tokenizer
        processor = TextProcessor(
            tokenizer_type="nllb",
            tokenizer_model="facebook/nllb-200-distilled-600M"
        )
        
        # Test basic properties
        self.assertEqual(processor.tokenizer_type, "nllb")
        self.assertEqual(processor.get_vocab_size(), 256_256)
        self.assertIsNone(processor.symbols)  # Not used with NLLB
        
        # Test text to sequence
        text = "Hello, multilingual world!"
        sequence = processor.text_to_sequence(text)
        self.assertIsInstance(sequence, list)
        self.assertGreater(len(sequence), 0)
        
        # Test sequence to text
        decoded = processor.sequence_to_text(sequence)
        self.assertIsInstance(decoded, str)
        
        # Test batch processing
        texts = ["Hello", "World", "Test"]
        sequences, lengths = processor.batch_text_to_sequence(texts, max_length=15)
        
        self.assertEqual(sequences.shape[0], 3)  # batch size
        self.assertEqual(len(lengths), 3)
        self.assertEqual(sequences.shape[1], 15)  # max length
    
    def test_config_integration(self):
        """Test model configuration with NLLB settings."""
        
        config = ModelConfig()
        config.tokenizer_type = "nllb"
        config.tokenizer_model = "facebook/nllb-200-distilled-600M"
        config.text_vocab_size = 256_256
        
        # Verify configuration
        self.assertEqual(config.tokenizer_type, "nllb")
        self.assertEqual(config.tokenizer_model, "facebook/nllb-200-distilled-600M")
        self.assertEqual(config.text_vocab_size, 256_256)
        
        # Test that config can be serialized/deserialized
        from myxtts.config.config import XTTSConfig
        
        xtts_config = XTTSConfig()
        xtts_config.model = config
        
        config_dict = xtts_config.to_dict()
        self.assertEqual(config_dict["model"]["tokenizer_type"], "nllb")
        self.assertEqual(config_dict["model"]["text_vocab_size"], 256_256)
    
    def test_custom_vs_nllb_tokenizer_types(self):
        """Test both custom and NLLB tokenizer types work."""
        
        # Test custom tokenizer
        custom_processor = TextProcessor(tokenizer_type="custom")
        self.assertEqual(custom_processor.tokenizer_type, "custom")
        self.assertIsNotNone(custom_processor.symbols)
        self.assertIsNone(custom_processor.nllb_tokenizer)
        
        custom_vocab_size = custom_processor.get_vocab_size()
        self.assertLess(custom_vocab_size, 1000)  # Custom tokenizer has small vocab
        
        # Test text processing with custom tokenizer
        text = "Hello!"
        custom_seq = custom_processor.text_to_sequence(text)
        self.assertIsInstance(custom_seq, list)
        self.assertGreater(len(custom_seq), 0)
        
        # Verify different behavior between tokenizer types
        # (We can't test NLLB without mock, but the structure is different)
        self.assertIsNotNone(custom_processor.symbol_to_id)
        
        # Custom tokenizer should handle simple texts
        decoded = custom_processor.sequence_to_text(custom_seq)
        self.assertIsInstance(decoded, str)


if __name__ == '__main__':
    unittest.main()