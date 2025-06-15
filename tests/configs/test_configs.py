import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic_settings import SettingsConfigDict

# Import the functions to test
from knowlang.configs.base import get_resource_path 


class TestGetResourcePath(unittest.TestCase):
    """Test cases for get_resource_path function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_relative_path = "settings/.env.app"
        self.test_cwd = Path("/home/user/project")
        self.test_meipass = Path("/tmp/_MEI123456/")
    
    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_development_mode(self, mock_cwd):
        """Test get_resource_path in development mode (not PyInstaller)"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        
        # Act
        result = get_resource_path(self.test_relative_path)
        
        # Assert
        expected = self.test_cwd / self.test_relative_path
        self.assertEqual(result, expected)
        mock_cwd.assert_called_once()

    @patch('knowlang.configs.base.sys._MEIPASS', "", create=True)
    @patch('knowlang.configs.base.sys.frozen', True, create=True)
    def test_get_resource_path_pyinstaller_mode(self):
        """Test get_resource_path in PyInstaller bundle mode"""
        # Arrange
        sys._MEIPASS = str(self.test_meipass)
        
        # Act
        result = get_resource_path(self.test_relative_path)
        
        # Assert
        expected = self.test_meipass / self.test_relative_path
        self.assertEqual(result, expected)
    
    @patch('knowlang.configs.base.sys.frozen', True, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_frozen_without_meipass(self, mock_cwd):
        """Test get_resource_path when frozen but no _MEIPASS (edge case)"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        # Ensure _MEIPASS doesn't exist
        if hasattr(sys, '_MEIPASS'):
            delattr(sys, '_MEIPASS')
        
        # Act
        result = get_resource_path(self.test_relative_path)
        
        # Assert
        expected = self.test_cwd / self.test_relative_path
        self.assertEqual(result, expected)
        mock_cwd.assert_called_once()
    
    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_empty_string(self, mock_cwd):
        """Test get_resource_path with empty relative path"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        
        # Act
        result = get_resource_path("")
        
        # Assert
        expected = self.test_cwd
        self.assertEqual(result, expected)
    
    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_nested_path(self, mock_cwd):
        """Test get_resource_path with deeply nested relative path"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        nested_path = "config/settings/dev/.env.local"
        
        # Act
        result = get_resource_path(nested_path)
        
        # Assert
        expected = self.test_cwd / nested_path
        self.assertEqual(result, expected)

    @patch('knowlang.configs.base.sys._MEIPASS', "", create=True)
    @patch('knowlang.configs.base.sys.frozen', True, create=True)
    def test_get_resource_path_pyinstaller_nested_path(self):
        """Test get_resource_path in PyInstaller mode with nested path"""
        # Arrange
        sys._MEIPASS = str(self.test_meipass)
        nested_path = "config/settings/prod/.env.app"
        
        # Act
        result = get_resource_path(nested_path)
        
        # Assert
        expected = self.test_meipass / nested_path
        self.assertEqual(result, expected)
