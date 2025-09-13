"""
Test NLLB language code standardization functionality.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNLLBLanguageCodeMapping(unittest.TestCase):
    """Test NLLB language code mapping functionality without external dependencies."""
    
    def test_basic_language_mappings(self):
        """Test that basic language mappings work correctly."""
        # Import the mapping directly to avoid dependency issues
        from myxtts.utils.text import NLLB_LANGUAGE_CODES, get_nllb_language_code
        
        # Test key language mappings mentioned in the issue
        test_cases = [
            ("en", "eng_Latn"),  # English - main example from the issue
            ("es", "spa_Latn"),  # Spanish
            ("fr", "fra_Latn"),  # French
            ("de", "deu_Latn"),  # German
            ("fa", "pes_Arab"),  # Persian/Farsi - important for the issue context
            ("ar", "arb_Arab"),  # Arabic
            ("zh", "zho_Hans"),  # Chinese
            ("ja", "jpn_Jpan"),  # Japanese
            ("ru", "rus_Cyrl"),  # Russian
        ]
        
        for iso_code, expected_nllb in test_cases:
            with self.subTest(language=iso_code):
                actual_nllb = get_nllb_language_code(iso_code)
                self.assertEqual(actual_nllb, expected_nllb, 
                    f"Expected {iso_code} -> {expected_nllb}, got {actual_nllb}")
    
    def test_unsupported_language_fallback(self):
        """Test that unsupported languages fall back to English."""
        from myxtts.utils.text import get_nllb_language_code
        
        # Test various unsupported language codes
        unsupported_codes = ["xyz", "invalid", "", "999"]
        
        for unsupported in unsupported_codes:
            with self.subTest(language=unsupported):
                result = get_nllb_language_code(unsupported)
                self.assertEqual(result, "eng_Latn", 
                    f"Unsupported language '{unsupported}' should fallback to 'eng_Latn', got '{result}'")
    
    def test_helper_functions(self):
        """Test utility functions for language code handling."""
        from myxtts.utils.text import (
            get_supported_nllb_languages, 
            is_nllb_language_supported,
            NLLB_LANGUAGE_CODES
        )
        
        # Test get_supported_nllb_languages
        supported = get_supported_nllb_languages()
        self.assertIsInstance(supported, list)
        self.assertGreater(len(supported), 0)
        self.assertIn("en", supported)
        self.assertIn("fa", supported)  # Persian should be supported
        
        # Test consistency with NLLB_LANGUAGE_CODES
        self.assertEqual(set(supported), set(NLLB_LANGUAGE_CODES.keys()))
        
        # Test is_nllb_language_supported
        self.assertTrue(is_nllb_language_supported("en"))
        self.assertTrue(is_nllb_language_supported("fa"))
        self.assertFalse(is_nllb_language_supported("xyz"))
        self.assertFalse(is_nllb_language_supported(""))
    
    def test_nllb_mapping_completeness(self):
        """Test that the NLLB mapping contains expected languages."""
        from myxtts.utils.text import NLLB_LANGUAGE_CODES
        
        # Check that key languages are included
        essential_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ar", "fa"]
        
        for lang in essential_languages:
            with self.subTest(language=lang):
                self.assertIn(lang, NLLB_LANGUAGE_CODES, 
                    f"Essential language '{lang}' should be in NLLB_LANGUAGE_CODES")
        
        # Check that the mapping is substantial (should have many languages)
        self.assertGreaterEqual(len(NLLB_LANGUAGE_CODES), 50, 
            "NLLB language mapping should contain at least 50 languages")


if __name__ == '__main__':
    # Run without importing other modules that might have dependencies
    try:
        unittest.main()
    except ImportError as e:
        print(f"Import error during testing: {e}")
        print("This might be due to missing dependencies like librosa, tensorflow, etc.")
        print("The core functionality can still be tested directly.")
        
        # Fallback manual testing
        print("\nRunning manual tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNLLBLanguageCodeMapping)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)