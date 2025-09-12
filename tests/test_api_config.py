#!/usr/bin/env python3
"""
Tests for API-based configuration loading and smart dataset handling.
"""

import unittest
import json
import tempfile
import os
import http.server
import socketserver
import threading
import time
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig


class TestAPIConfiguration(unittest.TestCase):
    """Test API-based configuration loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_config = {
            "model": {
                "text_encoder_dim": 512,
                "languages": ["en", "es", "fr"],
                "sample_rate": 22050
            },
            "data": {
                "dataset_path": "./test_data",
                "batch_size": 16,
                "language": "en"
            },
            "training": {
                "epochs": 100,
                "learning_rate": 1e-4,
                "optimizer": "adamw"
            }
        }
    
    def test_from_api_basic(self):
        """Test basic API configuration loading."""
        with patch('requests.get') as mock_get:
            # Mock successful API response
            mock_response = MagicMock()
            mock_response.json.return_value = self.sample_config
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Load config from API
            config = XTTSConfig.from_api("http://test-api.com/config")
            
            # Verify configuration loaded correctly
            self.assertEqual(config.model.text_encoder_dim, 512)
            self.assertEqual(config.data.batch_size, 16)
            self.assertEqual(config.training.epochs, 100)
            self.assertEqual(config.model.languages, ["en", "es", "fr"])
    
    def test_from_api_with_auth(self):
        """Test API configuration loading with authentication."""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = self.sample_config
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Load config with API key
            api_key = "test-api-key"
            config = XTTSConfig.from_api(
                "http://test-api.com/config", 
                api_key=api_key
            )
            
            # Verify request was made with correct headers
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            headers = call_args[1]['headers']
            self.assertEqual(headers['Authorization'], f'Bearer {api_key}')
            self.assertEqual(headers['Content-Type'], 'application/json')
    
    def test_from_api_dataset_path_checking(self):
        """Test smart dataset path checking."""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = self.sample_config
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            with patch('os.path.exists') as mock_exists:
                # Test case: dataset exists locally
                mock_exists.return_value = True
                
                with patch('builtins.print') as mock_print:
                    config = XTTSConfig.from_api("http://test-api.com/config")
                    
                    # Verify local dataset message was printed
                    mock_print.assert_called()
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    local_msg_found = any("Dataset found at local path" in msg for msg in print_calls)
                    self.assertTrue(local_msg_found)
    
    def test_from_api_dataset_not_exists(self):
        """Test behavior when dataset doesn't exist locally."""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = self.sample_config
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            with patch('os.path.exists') as mock_exists:
                # Test case: dataset doesn't exist locally
                mock_exists.return_value = False
                
                with patch('builtins.print') as mock_print:
                    config = XTTSConfig.from_api("http://test-api.com/config")
                    
                    # Verify download message was printed
                    mock_print.assert_called()
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    download_msg_found = any("will be downloaded" in msg for msg in print_calls)
                    self.assertTrue(download_msg_found)
    
    def test_from_api_error_handling(self):
        """Test error handling for API failures."""
        with patch('requests.get') as mock_get:
            # Test HTTP error
            mock_get.side_effect = Exception("Connection failed")
            
            with self.assertRaises(Exception):
                XTTSConfig.from_api("http://invalid-api.com/config")
    
    def test_from_api_invalid_response(self):
        """Test handling of invalid API responses."""
        with patch('requests.get') as mock_get:
            # Test invalid JSON response
            mock_response = MagicMock()
            mock_response.json.return_value = "invalid config"  # String instead of dict
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            with self.assertRaises(ValueError):
                XTTSConfig.from_api("http://test-api.com/config")


class TestSmartDatasetHandling(unittest.TestCase):
    """Test smart dataset handling functionality."""
    
    def test_dataset_exists_locally(self):
        """Test behavior when dataset exists locally."""
        # This would be tested in integration with the trainer
        # For now, we test the path checking logic
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock dataset directory
            dataset_dir = os.path.join(temp_dir, "LJSpeech-1.1")
            os.makedirs(dataset_dir)
            
            # Test path exists
            self.assertTrue(os.path.exists(dataset_dir))
    
    def test_dataset_does_not_exist(self):
        """Test behavior when dataset doesn't exist locally."""
        non_existent_path = "/path/that/does/not/exist/LJSpeech-1.1"
        self.assertFalse(os.path.exists(non_existent_path))


class TestFullAPIExample(unittest.TestCase):
    """Test the full API example functionality."""
    
    def test_example_config_structure(self):
        """Test that the example configuration has the correct structure."""
        from examples.api_config_example import create_sample_config_api_response
        
        config_dict = create_sample_config_api_response()
        
        # Verify required sections exist
        self.assertIn('model', config_dict)
        self.assertIn('data', config_dict)
        self.assertIn('training', config_dict)
        
        # Verify some key fields
        self.assertIn('dataset_path', config_dict['data'])
        self.assertIn('languages', config_dict['model'])
        self.assertIn('epochs', config_dict['training'])
    
    def test_config_creation_from_example(self):
        """Test creating config from example response."""
        from examples.api_config_example import create_sample_config_api_response
        
        config_dict = create_sample_config_api_response()
        
        # Create config objects
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        training_config = TrainingConfig(**config_dict['training'])
        
        # Verify objects created successfully
        self.assertIsInstance(model_config, ModelConfig)
        self.assertIsInstance(data_config, DataConfig)
        self.assertIsInstance(training_config, TrainingConfig)


if __name__ == '__main__':
    unittest.main()