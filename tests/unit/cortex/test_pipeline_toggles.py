"""Tests for Phase 6: Unified retrieval_augmented pipeline features.

- DataSourceDefinition extended schema
- PipelineToggles feature flags
- Template config merging with pipeline settings
"""

from contextunity.router.cortex.compiler.template_loader import (
    DataSourceDefinition,
    PipelineToggles,
    TemplateConfig,
    load_template,
)


class TestTemplateConfigWithPipeline:
    """TemplateConfig integrates pipeline toggles."""

    def test_pipeline_override(self):
        cfg = TemplateConfig(pipeline=PipelineToggles(memory=True, visualization=True))
        assert cfg.pipeline.memory is True
        assert cfg.pipeline.visualization is True

    def test_data_sources_with_pipeline(self):
        cfg = TemplateConfig(
            data_sources=[
                DataSourceDefinition(type="vector", binding="brain_search"),
                DataSourceDefinition(type="sql", binding="analytics_db"),
            ],
            pipeline=PipelineToggles(verification=True),
        )
        assert len(cfg.data_sources) == 2
        assert cfg.pipeline.verification is True


class TestRetrievalAugmentedPipelineConfig:
    """Template loads with pipeline toggles from YAML."""

    def test_template_has_pipeline_config(self):
        tpl = load_template("retrieval_augmented")
        assert tpl.config.pipeline.reflection is True
        assert tpl.config.pipeline.suggestions is True
        assert tpl.config.pipeline.memory is False
