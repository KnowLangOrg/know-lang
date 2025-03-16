import json
from knowlang.evaluations.base import SearchConfiguration
from knowlang.evaluations.config_manager import SearchConfigurationManager

class TestSearchConfigurationManager:
    """Tests for the SearchConfigurationManager."""
    
    def test_init(self, temp_dir):
        """Test initialization."""
        config_dir = temp_dir / "configs"
        manager = SearchConfigurationManager(config_dir)
        
        assert manager.config_dir == config_dir
        assert config_dir.exists()
    
    def test_save_configuration(self, temp_dir, sample_search_configuration):
        """Test saving a configuration."""
        config_dir = temp_dir / "configs"
        manager = SearchConfigurationManager(config_dir)
        
        manager.save_configuration(sample_search_configuration)
        
        file_path = config_dir / f"{sample_search_configuration.name}.json"
        assert file_path.exists()
        
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data["name"] == sample_search_configuration.name
        assert loaded_data["description"] == sample_search_configuration.description
    
    def test_load_configuration(self, temp_dir, sample_search_configuration):
        """Test loading a configuration."""
        config_dir = temp_dir / "configs"
        manager = SearchConfigurationManager(config_dir)
        
        # First save the configuration
        manager.save_configuration(sample_search_configuration)
        
        # Then load it
        loaded_config = manager.load_configuration(sample_search_configuration.name)
        
        assert loaded_config is not None
        assert loaded_config.name == sample_search_configuration.name
        assert loaded_config.description == sample_search_configuration.description
        assert loaded_config.keyword_search_enabled == sample_search_configuration.keyword_search_enabled
    
    def test_load_nonexistent_configuration(self, temp_dir):
        """Test loading a configuration that doesn't exist."""
        config_dir = temp_dir / "configs"
        manager = SearchConfigurationManager(config_dir)
        
        loaded_config = manager.load_configuration("nonexistent")
        
        assert loaded_config is None
    
    def test_list_configurations(self, temp_dir, sample_search_configuration):
        """Test listing configurations."""
        config_dir = temp_dir / "configs"
        manager = SearchConfigurationManager(config_dir)
        
        # Save multiple configurations
        manager.save_configuration(sample_search_configuration)
        
        second_config = SearchConfiguration(
            name="second_config",
            description="Second test configuration",
            keyword_search_enabled=False,
            vector_search_enabled=True,
            reranking_enabled=False
        )
        manager.save_configuration(second_config)
        
        # List configurations
        configs = manager.list_configurations()
        
        assert len(configs) == 2
        assert sample_search_configuration.name in configs
        assert second_config.name in configs
    
    def test_create_default_configurations(self, temp_dir):
        """Test creating default configurations."""
        config_dir = temp_dir / "configs"
        manager = SearchConfigurationManager(config_dir)
        
        default_configs = manager.create_default_configurations()
        
        assert len(default_configs) == 6  # There are 6 default configurations
        
        # Check that all configurations were saved to files
        config_files = list(config_dir.glob("*.json"))
        assert len(config_files) == 6
        
        # Verify configurations have expected names
        config_names = [config.name for config in default_configs]
        assert "baseline" in config_names
        assert "keyword_only" in config_names
        assert "vector_only" in config_names