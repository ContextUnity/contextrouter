"""Masking configuration — business-agnostic entity rules.

The consumer project (e.g. medical_analyst, contextunity.commerce) provides
the column_rules and text_patterns configuration. contextunity.router.cortex.privacy itself
does NOT know what a "doctor" or "patient" is — it only knows entity types
and their token prefixes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Literal

from contextunity.core.parsing import yaml_load
from contextunity.core.types import is_object_dict, is_object_list
from pydantic import BaseModel, ConfigDict, Field


class EntityRule(BaseModel):
    """Rule for a single entity type.

    Attributes:
        entity_type: Semantic type name (e.g. "doctor", "patient", "episode").
        prefix: Token prefix (e.g. "DOC", "PAT", "EPI"). Max 5 uppercase chars.
        pattern: Optional regex to detect this entity in free text.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    entity_type: str
    prefix: str = Field(max_length=5, pattern=r"^[A-Z]{2,5}$")
    pattern: str | None = None  # regex for text detection

    def compiled_pattern(self) -> re.Pattern[str] | None:
        """Return compiled regex if pattern is set."""
        if self.pattern:
            return re.compile(self.pattern, re.UNICODE)
        return None


class MaskingConfig(BaseModel):
    """Configuration for PII masking.

    Attributes:
        column_rules: Mapping of DataFrame column name → EntityRule.
                      Used by PIIMasker.mask_dataframe().
        text_entity_rules: List of EntityRule for free-text masking.
                           Used by PIIMasker.mask_text().
        drop_columns: Column names to remove entirely from DataFrame.
        transform_columns: Column name -> transform function name.
                           E.g. {"age": "age_bucket"} -> 23 -> "20-29".
        token_style: How to generate token suffixes.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    column_rules: dict[str, EntityRule] = Field(default_factory=dict)
    text_entity_rules: list[EntityRule] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    transform_columns: dict[str, str] = Field(default_factory=dict)
    token_style: Literal["random_hex", "uuid", "sequential"] = "random_hex"

    @classmethod
    def from_yaml(cls, path: str | Path) -> MaskingConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml_load(f)

        data = loaded if is_object_dict(loaded) else {}
        rules: list[EntityRule] = []
        rules_raw = data.get("rules", [])
        if is_object_list(rules_raw):
            for rule_raw in rules_raw:
                if is_object_dict(rule_raw):
                    rules.append(EntityRule.model_validate(rule_raw))

        return cls(text_entity_rules=rules)
