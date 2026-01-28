"""
Tests for Gardener enrichment graph.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


class TestGardenerConfig:
    """Test GardenerConfig validation."""
    
    def test_gardener_config_no_business_defaults(self):
        """Verify no business-specific defaults in config."""
        from contextrouter.core.config import GardenerConfig
        
        config = GardenerConfig()
        
        # tenant_id should be empty, not "traverse" or similar
        assert config.tenant_id == ""
        
        # Should NOT have prompts_path or ontology_path
        assert not hasattr(config, "prompts_path")
        assert not hasattr(config, "ontology_path")
    
    def test_gardener_config_has_processing_defaults(self):
        """Verify processing defaults are set."""
        from contextrouter.core.config import GardenerConfig
        
        config = GardenerConfig()
        
        assert config.batch_size == 50
        assert config.poll_interval_sec == 900  # 15 min


class TestLoadPrompt:
    """Test prompt loading from files."""
    
    def test_load_prompt_from_directory(self, tmp_path):
        """Test loading prompt from custom directory."""
        from contextrouter.cortex.graphs.commerce.gardener.nodes import _load_prompt
        
        # Create test prompt
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test.txt").write_text("Test prompt content")
        
        result = _load_prompt(str(prompts_dir), "test.txt")
        
        assert result == "Test prompt content"
    
    def test_load_prompt_not_found_raises(self, tmp_path):
        """Test error when prompt not found."""
        from contextrouter.cortex.graphs.commerce.gardener.nodes import _load_prompt
        
        with pytest.raises(FileNotFoundError, match="Prompt not found"):
            _load_prompt(str(tmp_path), "nonexistent.txt")


class TestSlugify:
    """Test slugify helper."""
    
    def test_slugify_basic(self):
        """Test basic slugification."""
        from contextrouter.cortex.graphs.commerce.gardener.nodes import _slugify
        
        assert _slugify("Gore-Tex Pro") == "gore-tex-pro"
        assert _slugify("Arc'teryx") == "arcteryx"
        assert _slugify("  Vibram Megagrip  ") == "vibram-megagrip"


class TestParseJsonResponse:
    """Test JSON parsing from LLM responses."""
    
    def test_parse_json_array(self):
        """Test parsing JSON array from response."""
        from contextrouter.cortex.graphs.commerce.gardener.nodes import _parse_json_response
        
        content = '[{"id": 1, "category": "outdoor.jackets"}]'
        result = _parse_json_response(content)
        
        assert len(result) == 1
        assert result[0]["id"] == 1
    
    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown."""
        from contextrouter.cortex.graphs.commerce.gardener.nodes import _parse_json_response
        
        content = '''Here is the result:
```json
[{"id": 1, "category": "test"}]
```
'''
        result = _parse_json_response(content)
        
        assert len(result) == 1
        assert result[0]["category"] == "test"
    
    def test_parse_invalid_json_returns_empty(self):
        """Test invalid JSON returns empty list."""
        from contextrouter.cortex.graphs.commerce.gardener.nodes import _parse_json_response
        
        result = _parse_json_response("not json at all")
        
        assert result == []
