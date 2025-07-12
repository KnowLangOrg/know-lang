import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        # Called once at the beginning of the function
        self.assertEqual(mock_cwd.call_count, 1)

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
        # Called once at the beginning of the function
        self.assertEqual(mock_cwd.call_count, 1)
    
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

    @patch('knowlang.configs.base.Path.exists')
    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_primary_exists(self, mock_cwd, mock_exists):
        """Test get_resource_path when primary path exists"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        mock_exists.return_value = True  # Primary path exists
        
        # Act
        result = get_resource_path(self.test_relative_path)
        
        # Assert
        expected = self.test_cwd / self.test_relative_path
        self.assertEqual(result, expected)
        # Called once at the beginning of the function
        self.assertEqual(mock_cwd.call_count, 1)
        mock_exists.assert_called_once()

    @patch('knowlang.configs.base.Path.exists')
    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_primary_not_exists_default_exists(self, mock_cwd, mock_exists):
        """Test get_resource_path when primary doesn't exist but default path exists"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        # First call returns False (primary doesn't exist), second call returns True (default exists)
        mock_exists.side_effect = [False, True]
        default_path = "settings/chat_example.yaml"
        
        # Act
        result = get_resource_path(self.test_relative_path, default_path=default_path)
        
        # Assert
        expected = self.test_cwd / default_path
        self.assertEqual(result, expected)
        # Called twice: once for primary, once for default
        self.assertEqual(mock_exists.call_count, 2)

    @patch('knowlang.configs.base.Path.exists')
    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_both_not_exist(self, mock_cwd, mock_exists):
        """Test get_resource_path when neither primary nor default path exists"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        mock_exists.return_value = False  # Neither path exists
        default_path = "settings/chat_example.yaml"
        
        # Act
        result = get_resource_path(self.test_relative_path, default_path=default_path)
        
        # Assert
        expected = self.test_cwd / self.test_relative_path
        self.assertEqual(result, expected)
        # Called once at the beginning, then twice for exists() checks
        self.assertEqual(mock_cwd.call_count, 1)
        self.assertEqual(mock_exists.call_count, 2)

    @patch('knowlang.configs.base.Path.exists')
    @patch('knowlang.configs.base.sys._MEIPASS', "", create=True)
    @patch('knowlang.configs.base.sys.frozen', True, create=True)
    def test_get_resource_path_pyinstaller_primary_not_exists_default_exists(self, mock_exists):
        """Test get_resource_path in PyInstaller mode with fallback"""
        # Arrange
        sys._MEIPASS = str(self.test_meipass)
        mock_exists.side_effect = [False, True]  # Primary fails, default succeeds
        default_path = "settings/chat_example.yaml"
        
        # Act
        result = get_resource_path(self.test_relative_path, default_path=default_path)
        
        # Assert
        expected = self.test_meipass / default_path
        self.assertEqual(result, expected)
        self.assertEqual(mock_exists.call_count, 2)

    @patch('knowlang.configs.base.Path.exists')
    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_no_default_path(self, mock_cwd, mock_exists):
        """Test get_resource_path without default path when primary doesn't exist"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        mock_exists.return_value = False  # Primary path doesn't exist
        
        # Act
        result = get_resource_path(self.test_relative_path)
        
        # Assert
        expected = self.test_cwd / self.test_relative_path
        self.assertEqual(result, expected)
        # Called once at the beginning
        self.assertEqual(mock_cwd.call_count, 1)
        mock_exists.assert_called_once()

    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_none_default_path(self, mock_cwd):
        """Test get_resource_path with None as default_path"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        
        # Act
        result = get_resource_path(self.test_relative_path, default_path=None)
        
        # Assert - should return primary path
        expected = self.test_cwd / self.test_relative_path
        self.assertEqual(result, expected)

    @patch('knowlang.configs.base.sys.frozen', False, create=True)
    @patch('knowlang.configs.base.Path.cwd')
    def test_get_resource_path_empty_default_path(self, mock_cwd):
        """Test get_resource_path with empty string as default_path"""
        # Arrange
        mock_cwd.return_value = self.test_cwd
        default_path = ""
        
        # Act
        result = get_resource_path(self.test_relative_path, default_path=default_path)
        
        # Assert - should return primary path
        expected = self.test_cwd / self.test_relative_path
        self.assertEqual(result, expected)
