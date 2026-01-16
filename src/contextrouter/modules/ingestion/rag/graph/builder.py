"""Knowledge graph builder from raw data."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx

from contextrouter.core import Config

from ..core.types import RawData
from ..core.utils import (
    llm_generate_tsv,
    parse_tsv_line,
)
from .prompts import format_extraction_prompt
from .serialization import (
    load_graph_secure,
    save_graph_secure,
)
from .utils import (
    GENERIC_NODES,
    clean_entity,
    clean_label,
    extract_name_key,
    find_best_canonical,
    is_name_like,
    normalize_article,
)

logger = logging.getLogger(__name__)

# Default allowed relation labels (can be overridden by ontology)
DEFAULT_ALLOWED_LABELS = {
    "CAUSES",
    "LEADS_TO",
    "ENABLES",
    "REQUIRES",
    "SUPPORTS",
    "OPPOSES",
    "PREVENTS",
    "RESULTS_IN",
    "IS_A",
    "IS_PART_OF",
    "INCLUDES",
    "EXAMPLE_OF",
    "APPLIES_TO",
    "IS_ABOUT",
    "RELATED_TO",
}

# Exported for ontology stage.
ALLOWED_RELATION_LABELS = DEFAULT_ALLOWED_LABELS


class GraphBuilder:
    """Builds a knowledge graph from raw ingestion data."""

    def __init__(
        self,
        max_workers: int = 4,
        taxonomy_path: Path | None = None,
        ontology_path: Path | None = None,
        *,
        model: str | None = None,
        mode: str = "llm",  # "llm", "local", "hybrid"
        core_cfg: Config,
    ) -> None:
        """Initialize graph builder.

        Args:
            mode: Graph building mode
                - "llm": Use LLM for graph extraction (default, high quality)
                - "local": Use cognee for local extraction (fast, free, lower quality)
                - "hybrid": Try local first, fallback to LLM for complex content
        """
        self.graph = nx.Graph()
        self.max_workers = max_workers
        self.model = model or core_cfg.models.ingestion.graph.model
        self.mode = mode
        self.core_cfg = core_cfg
        self.taxonomy: dict[str, Any] | None = None
        self.canonical_map: dict[str, str] = {}
        self.allowed_relation_labels = set(DEFAULT_ALLOWED_LABELS)
        self._debug_bundles_written = 0

        # Initialize cognee builder for local/hybrid modes
        self._cognee_builder = None
        if self.mode in ("local", "hybrid"):
            try:
                from contextrouter.modules.tools.cognee import CogneeGraphBuilder

                self._cognee_builder = CogneeGraphBuilder()
            except ImportError:
                logger.warning("Cognee not available, falling back to LLM mode")
                if self.mode == "local":
                    self.mode = "llm"

        self._load_taxonomy(taxonomy_path)
        self._load_ontology(ontology_path)

    def _load_taxonomy(self, path: Path | None) -> None:
        """Load taxonomy for entity normalization."""
        if not path or not path.exists():
            return

        try:
            with open(path, encoding="utf-8") as f:
                self.taxonomy = json.load(f)
            self.canonical_map = self.taxonomy.get("canonical_map", {})
            logger.info("Loaded taxonomy with %d canonical mappings", len(self.canonical_map))
        except Exception as e:
            logger.warning("Failed to load taxonomy: %s", e)

    def _load_ontology(self, path: Path | None) -> None:
        """Load ontology for relation label validation."""
        if not path:
            logger.info(
                "No ontology specified, using default allowed_labels (%d)",
                len(self.allowed_relation_labels),
            )
            return

        if not path.exists():
            logger.warning("Ontology path specified but file not found: %s", path)
            return

        try:
            with open(path, encoding="utf-8") as f:
                onto = json.load(f)
            rel = onto.get("relations", {}) if isinstance(onto, dict) else {}
            labels = rel.get("allowed_labels") if isinstance(rel, dict) else None

            if isinstance(labels, list):
                parsed = {str(x).strip() for x in labels if isinstance(x, str) and str(x).strip()}
                if parsed:
                    self.allowed_relation_labels = parsed
                    sample = ", ".join(sorted(parsed)[:5])
                    logger.info(
                        "Loaded ontology from %s: %d allowed_labels (%s%s)",
                        path,
                        len(parsed),
                        sample,
                        "..." if len(parsed) > 5 else "",
                    )
                else:
                    logger.warning("Ontology file has empty allowed_labels, using defaults")
            else:
                logger.warning("Ontology file missing 'relations.allowed_labels', using defaults")
        except Exception as e:
            logger.warning("Failed to load ontology from %s: %s", path, e)

    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity using taxonomy canonical map + fuzzy matching."""
        cleaned = clean_entity(entity)
        if not cleaned:
            return ""

        entity_lower = cleaned.lower()

        # 1. Exact match
        if canonical := self.canonical_map.get(entity_lower):
            return canonical

        # 2. Article variations
        no_article = normalize_article(entity_lower)
        if no_article != entity_lower:
            if canonical := self.canonical_map.get(no_article):
                return canonical

        if canonical := self.canonical_map.get(f"the {entity_lower}"):
            return canonical

        # 3. Partial name matching
        if len(entity_lower) >= 5:
            candidates = [
                (len(canon_lower), canon)
                for canon_lower, canon in self.canonical_map.items()
                if entity_lower in canon_lower or canon_lower in entity_lower
            ]

            # Check existing graph nodes
            if self.graph:
                for node in self.graph.nodes():
                    node_lower = str(node).lower()
                    if entity_lower.startswith(
                        node_lower[: len(entity_lower)]
                    ) or node_lower.startswith(entity_lower):
                        if is_name_like(str(node)) or is_name_like(entity):
                            candidates.append((len(node_lower), str(node)))

            if best := find_best_canonical(
                entity_lower,
                candidates,
                max_diff=20 if is_name_like(entity) else 10,
            ):
                return best

        return cleaned

    def _choose_relation_columns(
        self, parts: list[str]
    ) -> tuple[str | None, str | None, str | None]:
        """Extract (source, target, label) from TSV parts, handling column swaps."""
        if len(parts) < 3:
            return (None, None, None)

        source = clean_entity(parts[0].strip())
        b_raw, c_raw = parts[1].strip(), parts[2].strip()

        b_label = clean_label(b_raw, self.allowed_relation_labels)
        c_label = clean_label(c_raw, self.allowed_relation_labels)

        b_allowed = bool(b_label and b_label in self.allowed_relation_labels)
        c_allowed = bool(c_label and c_label in self.allowed_relation_labels)

        # Prefer column with allowed label
        if c_allowed and not b_allowed:
            return (source, clean_entity(b_raw), c_label)
        if b_allowed and not c_allowed:
            return (source, clean_entity(c_raw), b_label)

        # Fallback: prefer any valid label
        if c_label and not b_label:
            return (source, clean_entity(b_raw), c_label)
        if b_label and not c_label:
            return (source, clean_entity(c_raw), b_label)

        # Default: assume prompt order
        return (source, clean_entity(b_raw), c_label)

    def _parse_tsv_output(self, text: str) -> tuple[list[str], list[dict]]:
        """Parse TSV output from LLM into entities and relations."""
        entities: list[str] = []
        relations: list[dict] = []
        section = None

        for line in (text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.upper().startswith("ENTITIES"):
                section = "entities"
                continue
            if line.upper().startswith("RELATIONS"):
                section = "relations"
                continue

            if section == "entities":
                if ent := clean_entity(line.lstrip("- ")):
                    entities.append(ent)
            elif section == "relations":
                parts = parse_tsv_line(line)
                if len(parts) >= 3:
                    result = self._choose_relation_columns(parts)
                    if result[0] and result[1] and result[2]:
                        src, dst, lbl = result
                        relations.append({"source": src, "target": dst, "label": lbl})

        return (entities, relations)

    def _extract_from_item(
        self, raw_data: RawData, *, output_path: Path
    ) -> tuple[list[str], list[dict]]:
        """Extract entities and relations from a single RawData item."""
        content = raw_data.content

        if len(content) < 200:
            logger.debug("Skipping graph extraction for short content (%d chars)", len(content))
            return ([], [])

        # Truncate large content
        if len(content) > 10000:
            summary = raw_data.metadata.get("summary", "")
            content = content[:10000] + (f"\n\nSummary: {summary}" if summary else "")

        try:
            # Try local extraction first for hybrid/local modes
            if (
                self.mode in ("local", "hybrid")
                and self._cognee_builder
                and self._cognee_builder.is_available()
            ):
                try:
                    logger.debug("Attempting local graph extraction with cognee")
                    entities, relations = self._cognee_builder.build_graph(content)
                    logger.info(
                        f"Local extraction successful: {len(entities)} entities, {len(relations)} relations"
                    )
                except Exception as e:
                    logger.warning(
                        f"Local extraction failed ({e}), falling back to LLM"
                        if self.mode == "hybrid"
                        else f"Local extraction failed: {e}"
                    )
                    if self.mode == "local":
                        raise  # For local-only mode, don't fallback
                    # Fall through to LLM extraction for hybrid mode
                    entities, relations = self._extract_with_llm(content)
            else:
                # Use LLM extraction (default or when cognee unavailable)
                entities, relations = self._extract_with_llm(content)
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            raise

    def _extract_with_llm(self, content: str) -> tuple[list[str], list[dict]]:
        """Extract entities and relations using LLM."""
        try:
            prompt = format_extraction_prompt(content, self.taxonomy)
            # temperature=0 for deterministic output
            text = llm_generate_tsv(
                core_cfg=self.core_cfg,
                prompt=prompt,
                model=self.model,
                max_tokens=8192,
                temperature=0.0,
                retries=5,
            )
            entities, relations = self._parse_tsv_output(text)

            # Normalize entities
            normalized_entities = [
                e for e in (self._normalize_entity(e) for e in entities if e) if e
            ]

            # Normalize relations
            normalized_relations = []
            for rel in relations:
                source = self._normalize_entity(rel.get("source", ""))
                target = self._normalize_entity(rel.get("target", ""))
                label = (
                    clean_label(rel.get("label", ""), self.allowed_relation_labels) or "RELATED_TO"
                )

                if source and target and label and source != target:
                    normalized_relations.append(
                        {"source": source, "target": target, "label": label}
                    )

            return (normalized_entities, normalized_relations)
        except Exception as e:
            logger.warning("Failed to extract graph data from content: %s", e)
            # Don't raise here - allow pipeline to continue with empty results
            return ([], [])

    def _write_debug_bundle(
        self,
        output_path: Path,
        kind: str,
        prompt: str,
        raw_output: str,
        raw_data: RawData,
    ) -> None:
        """Write debug bundle for failed extractions."""
        if self._debug_bundles_written >= 5:
            return
        self._debug_bundles_written += 1

        try:
            import datetime

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = output_path.parent / "output" / "_processing" / "graph"
            debug_dir.mkdir(parents=True, exist_ok=True)
            path = debug_dir / f"graph_debug_{kind}_{ts}.json"

            path.write_text(
                json.dumps(
                    {
                        "kind": kind,
                        "meta": {
                            "content_len": len(raw_data.content or ""),
                            "raw_type": getattr(raw_data, "source_type", ""),
                            "raw_id": raw_data.metadata.get("id"),
                            "raw_source": raw_data.metadata.get("source"),
                        },
                        "prompt": prompt,
                        "raw_output": raw_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.warning("graph debug bundle written: %s", path)
        except Exception as e:
            logger.debug("graph debug bundle write failed: %s", e)

    def _merge_duplicate_nodes(self) -> dict[str, int]:
        """Merge duplicate nodes (case, article, name variations)."""
        nodes_list = list(self.graph.nodes())
        if not nodes_list:
            return {"merged": 0}

        node_map: dict[str, str] = {}

        # Pass 1: Normalize via canonical_map
        for node in nodes_list:
            node_str = str(node)
            if normalized := self._normalize_entity(node_str):
                if normalized != node_str:
                    node_map[node_str] = normalized

        # Pass 2: Case-insensitive duplicates
        node_lower_map: dict[str, str] = {}
        for node in nodes_list:
            node_str = str(node)
            node_lower = node_str.lower()
            if node_lower not in node_lower_map:
                node_lower_map[node_lower] = node_str
            else:
                existing = node_lower_map[node_lower]
                # Prefer canonical_map entry or longer
                if node_str in self.canonical_map.values() or (
                    len(node_str) > len(existing) and existing not in self.canonical_map.values()
                ):
                    node_lower_map[node_lower] = node_str

        for node_lower, preferred in node_lower_map.items():
            variants = [n for n in nodes_list if str(n).lower() == node_lower]
            if len(variants) > 1:
                target = preferred
                for variant in variants:
                    variant_str = str(variant)
                    if variant_str != target and variant_str not in node_map:
                        node_map[variant_str] = target

        # Pass 3: Article variations
        article_groups: dict[str, list[str]] = defaultdict(list)
        for node in nodes_list:
            node_str = str(node)
            no_article = normalize_article(node_str)
            if no_article != node_str.lower():
                article_groups[no_article].append(node_str)

        for no_article, variants in article_groups.items():
            if len(variants) <= 1:
                continue
            # Prefer without article, or canonical_map, or longest
            target = next(
                (v for v in variants if v.lower() == no_article),
                next((v for v in variants if v in self.canonical_map.values()), None)
                or max(variants, key=len),
            )
            for variant in variants:
                if variant != target and variant not in node_map:
                    node_map[variant] = target

        # Pass 4: Partial names
        name_groups: dict[str, list[str]] = defaultdict(list)
        for node in nodes_list:
            node_str = str(node)
            if node_str in node_map:
                continue
            if name_key := extract_name_key(node_str):
                name_groups[name_key].append(node_str)

        for name_key, variants in name_groups.items():
            if len(variants) <= 1:
                continue
            target = next((v for v in variants if v in self.canonical_map.values()), None) or max(
                variants, key=len
            )
            for variant in variants:
                if variant != target and len(target) >= len(variant) + 3:
                    if variant not in node_map:
                        node_map[variant] = target

        # Apply merges
        merge_groups: dict[str, list[str]] = defaultdict(list)
        for variant, canonical in node_map.items():
            merge_groups[canonical].append(variant)

        merged_count = 0
        for canonical, variants in merge_groups.items():
            if len(variants) <= 1:
                continue

            # Ensure canonical exists in graph
            if canonical not in self.graph:
                # Use first variant that exists
                for variant in variants:
                    if variant in self.graph:
                        canonical = variant
                        variants = [v for v in variants if v != variant]
                        break
                else:
                    continue  # No valid canonical found

            for variant in variants:
                if variant == canonical or variant not in self.graph:
                    continue

                # Move edges
                for neighbor in list(self.graph.neighbors(variant)):
                    edge_data = self.graph.get_edge_data(variant, neighbor, {})
                    relation = edge_data.get("relation", "RELATED_TO")

                    if self.graph.has_edge(canonical, neighbor):
                        existing_label = self.graph[canonical][neighbor].get(
                            "relation", "RELATED_TO"
                        )
                        if existing_label == "RELATED_TO" and relation != "RELATED_TO":
                            self.graph[canonical][neighbor]["relation"] = relation
                    else:
                        self.graph.add_edge(canonical, neighbor, relation=relation)

                self.graph.remove_node(variant)
                merged_count += 1

        return {"merged": merged_count}

    def _cleanup_graph(self) -> dict[str, int]:
        """Post-build cleanup: merge duplicates, remove low-quality nodes/edges."""
        # Pass 0: Clean node names with control chars (safety net - should be rare)
        node_renames: dict[str, str] = {}
        for node in list(self.graph.nodes()):
            node_str = str(node)
            if cleaned := clean_entity(node_str):
                if cleaned != node_str:
                    node_renames[node_str] = cleaned
                    logger.warning("Found unclean node: %r -> %r", node_str, cleaned)

        # Apply renames
        for old_name, new_name in node_renames.items():
            if old_name not in self.graph:
                continue
            # Move edges to cleaned name
            for neighbor in list(self.graph.neighbors(old_name)):
                edge_data = self.graph.get_edge_data(old_name, neighbor, {})
                relation = edge_data.get("relation", "RELATED_TO")
                if not self.graph.has_edge(new_name, neighbor):
                    self.graph.add_node(new_name)  # Ensure node exists
                    self.graph.add_edge(new_name, neighbor, relation=relation)
            self.graph.remove_node(old_name)

        stats = {"cleaned_nodes": len(node_renames)}
        merge_stats = self._merge_duplicate_nodes()
        stats.update(merge_stats)

        # Remove isolated nodes
        isolated = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        for node in isolated:
            self.graph.remove_node(node)
        stats["isolated_removed"] = len(isolated)

        # Remove generic nodes
        generic_found = [n for n in self.graph.nodes() if str(n).strip().lower() in GENERIC_NODES]
        for node in generic_found:
            self.graph.remove_node(node)
        stats["generic_removed"] = len(generic_found)

        # Remove label-named nodes
        label_nodes = [
            n for n in self.graph.nodes() if str(n).strip().upper() in self.allowed_relation_labels
        ]
        for node in label_nodes:
            self.graph.remove_node(node)
        stats["label_nodes_removed"] = len(label_nodes)

        # Remove edges where label matches node name
        node_names_lower = {str(n).strip().lower() for n in self.graph.nodes()}
        allowed_labels_lower = {x.lower() for x in self.allowed_relation_labels}
        bad_edges = [
            (u, v)
            for u, v, data in self.graph.edges(data=True)
            if (lbl := str((data or {}).get("relation", "")).strip().lower())
            and lbl not in allowed_labels_lower
            and lbl in node_names_lower
        ]
        for u, v in set(bad_edges):
            try:
                self.graph.remove_edge(u, v)
            except Exception as e:
                logger.warning("Failed to remove edge (%s, %s): %s", u, v, e)
        stats["label_edges_removed"] = len(set(bad_edges))

        # Remove newly isolated nodes
        isolated_after = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        for node in isolated_after:
            self.graph.remove_node(node)
        stats["isolated_removed"] += len(isolated_after)

        return stats

    def build(
        self,
        raw_data_list: list[RawData],
        output_path: Path,
        *,
        incremental: bool = False,
    ) -> None:
        """Build graph from raw data and save to disk."""
        # Load existing graph if incremental
        if incremental and output_path.exists():
            try:
                # Load graph with integrity verification
                self.graph = load_graph_secure(output_path)
                logger.info(
                    "Existing graph loaded securely with %d nodes and %d edges",
                    self.graph.number_of_nodes(),
                    self.graph.number_of_edges(),
                )
            except Exception as e:
                logger.warning("Failed to load existing graph, starting fresh: %s", e)
                self.graph = nx.Graph()
        else:
            if output_path.exists() and not incremental:
                logger.info("Rebuilding graph from scratch (ignoring existing %s)", output_path)
            self.graph = nx.Graph()

        logger.info(
            "Building knowledge graph from %d raw data items (workers=%d model=%s incremental=%s)",
            len(raw_data_list),
            self.max_workers,
            self.model,
            incremental,
        )

        # Extract entities and relations in parallel
        all_entities: set[str] = set()
        all_relations: list[dict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_from_item, data, output_path=output_path): data
                for data in raw_data_list
            }

            for future in as_completed(futures):
                completed = sum(1 for f in futures if f.done())
                logger.info(
                    "Processing graph: %d/%d (%.1f%%)",
                    completed,
                    len(raw_data_list),
                    (completed / len(raw_data_list) * 100) if raw_data_list else 0,
                )

                try:
                    entities, relations = future.result()
                    # Entities are already cleaned in _extract_from_item
                    all_entities.update(e for e in entities if isinstance(e, str) and e.strip())
                    all_relations.extend(relations)
                except Exception as e:
                    logger.warning("Failed to process item: %s", e)

        logger.info(
            "Extracted %d unique entities and %d relations",
            len(all_entities),
            len(all_relations),
        )

        initial_nodes = self.graph.number_of_nodes()
        initial_edges = self.graph.number_of_edges()

        # CRITICAL: Clean ALL entities before adding to graph
        cleaned_entities: set[str] = set()
        for entity in all_entities:
            if cleaned := clean_entity(str(entity)):
                if 3 <= len(cleaned) <= 100:
                    cleaned_entities.add(cleaned)

        # Add only cleaned entities
        for entity in cleaned_entities:
            self.graph.add_node(entity)

        # Filter and clean relations, transform to cleaned tuples
        node_set_lower = {e.lower() for e in cleaned_entities}
        cleaned_relations: list[tuple[str, str, str]] = []
        for rel in all_relations:
            if not isinstance(rel, dict):
                continue
            # CRITICAL: Clean source and target
            source_raw = str(rel.get("source", ""))
            target_raw = str(rel.get("target", ""))
            label_raw = str(rel.get("label", ""))

            source = clean_entity(source_raw)
            target = clean_entity(target_raw)
            label = clean_label(label_raw, self.allowed_relation_labels)

            # Validate AFTER cleaning
            if not source or not target or not label:
                continue
            if len(source) < 3 or len(target) < 3:
                continue
            if len(source) > 100 or len(target) > 100:
                continue
            if source == target:
                continue
            if label.lower() in node_set_lower:
                continue
            if source.lower() in GENERIC_NODES or target.lower() in GENERIC_NODES:
                continue
            cleaned_relations.append((source, target, label))

        logger.info(
            "Filtered relations: %d -> %d (dropped %d)",
            len(all_relations),
            len(cleaned_relations),
            len(all_relations) - len(cleaned_relations),
        )

        # Resolve edge label conflicts deterministically
        edge_labels: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for source, target, label in cleaned_relations:
            a, b = (source, target) if source <= target else (target, source)
            edge_labels[(a, b)][label] += 1

        conflicts = sum(1 for counter in edge_labels.values() if len(counter) > 1)
        for (a, b), counter in sorted(edge_labels.items()):
            chosen = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            # CRITICAL: Final cleaning before adding edge
            a_clean = clean_entity(a)
            b_clean = clean_entity(b)
            if a_clean and b_clean and len(a_clean) >= 3 and len(b_clean) >= 3:
                self.graph.add_edge(a_clean, b_clean, relation=chosen)

        # Remove invalid edges
        invalid_edges = [
            (u, v)
            for u, v, data in self.graph.edges(data=True)
            if not clean_label(str((data or {}).get("relation", "")), self.allowed_relation_labels)
        ]
        for u, v in invalid_edges:
            try:
                self.graph.remove_edge(u, v)
            except Exception as e:
                logger.warning("Failed to remove invalid edge (%s, %s): %s", u, v, e)

        # Cleanup
        cleanup_stats = self._cleanup_graph()

        # Save securely with integrity verification
        save_graph_secure(self.graph, output_path)

        new_nodes = self.graph.number_of_nodes() - initial_nodes
        new_edges = self.graph.number_of_edges() - initial_edges

        logger.info(
            "Graph updated: %d nodes (+%d new), %d edges (+%d new). Saved to %s",
            self.graph.number_of_nodes(),
            new_nodes,
            self.graph.number_of_edges(),
            new_edges,
            output_path,
        )

        if conflicts:
            logger.info(
                "Graph label conflicts resolved deterministically on %d edges",
                conflicts,
            )

        # Label distribution
        label_counts = Counter(
            str((data or {}).get("relation", "")).strip()
            for _, _, data in self.graph.edges(data=True)
        )
        if label_counts:
            top = ", ".join(f"{k}:{v}" for k, v in label_counts.most_common(12))
            logger.info("Graph relation labels (top): %s", top)

        if cleanup_stats:
            parts = []
            if cleaned := cleanup_stats.get("cleaned_nodes", 0):
                parts.append(f"cleaned {cleaned} node names")
            if merged := cleanup_stats.get("merged", 0):
                parts.append(f"merged {merged} duplicates")
            if isolated := cleanup_stats.get("isolated_removed", 0):
                parts.append(f"removed {isolated} isolated")
            if generic := cleanup_stats.get("generic_removed", 0):
                parts.append(f"removed {generic} generic")
            if label_nodes := cleanup_stats.get("label_nodes_removed", 0):
                parts.append(f"removed {label_nodes} label-named")
            if label_edges := cleanup_stats.get("label_edges_removed", 0):
                parts.append(f"removed {label_edges} bad edges")
            if parts:
                logger.info("Graph cleanup: %s", ", ".join(parts))
