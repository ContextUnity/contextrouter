"""Prompts for knowledge graph extraction."""

# Base prompt template - taxonomy context is injected dynamically
# Uses TOON-style schema header for explicit column contract
GRAPH_EXTRACTION_PROMPT = """Extract entities and relationships from this text.

{taxonomy_context}

IMPORTANT: Use canonical terms from taxonomy when possible (e.g., "Wealth" not "riches").

Return in this EXACT format (use real tab characters, not "<TAB>"):

ENTITIES:
Entity1
Entity2
Entity3

RELATIONS:
# Schema: relations[n]{{source,target,label}}:
source<TAB>target<TAB>label
Entity1	Entity2	CAUSES
Entity2	Entity3	LEADS_TO

Rules:
- One entity per line under ENTITIES
- RELATIONS section MUST start with schema header: # Schema: relations[n]{{source,target,label}}:
- One relation per line under RELATIONS: source<TAB>target<TAB>label (STRICT ORDER - do NOT swap)
- No markdown, no bullets, no extra text
- Extract: Concepts, People, Books, Principles, Events
- Relation label MUST be EXACTLY one of these (case-sensitive):
  CAUSES, LEADS_TO, ENABLES, REQUIRES, SUPPORTS, OPPOSES, PREVENTS,
  IS_A, IS_PART_OF, INCLUDES, EXAMPLE_OF, APPLIES_TO, RESULTS_IN,
  IS_ABOUT, RELATED_TO
- If none fit exactly, use RELATED_TO (do NOT invent new labels).
- Never use a topic/entity name as a relation label (e.g., avoid WEALTH, SUCCESS, NAPOLEON_HILL).

Text:
{text}
"""


def build_taxonomy_context(taxonomy: dict | None) -> str:
    """Build taxonomy context string for the extraction prompt.

    Args:
        taxonomy: Taxonomy dictionary or None

    Returns:
        Context string to inject into prompt
    """
    if not taxonomy:
        return "Focus on the domain concepts, principles, and entities."

    categories = taxonomy.get("categories", {})

    context_parts = ["Focus on the following domain concepts and principles:\n"]

    # Explicitly list top-level taxonomy categories (for normalization guidance)
    if isinstance(categories, dict) and categories:
        cat_names = [c.replace("_", " ").title() for c in list(categories.keys())[:12]]
        context_parts.append(f"Taxonomy Categories: {', '.join(cat_names)}\n")

    # Add category descriptions
    for cat_name, cat_data in categories.items():
        description = cat_data.get("description", "")
        keywords = cat_data.get("keywords", [])[:10]  # Limit to 10 per category
        if keywords:
            context_parts.append(f"- **{cat_name.replace('_', ' ').title()}**: {description}")
            context_parts.append(f"  Keywords: {', '.join(keywords)}")

    # Add synonym mapping hints
    canonical_map = taxonomy.get("canonical_map", {})
    if canonical_map:
        # Show a few example mappings
        examples = list(canonical_map.items())[:10]
        if examples:
            context_parts.append("\nSynonym mappings (use canonical terms on the right):")
            for syn, canonical in examples:
                if syn.lower() != canonical.lower():
                    context_parts.append(f"  - '{syn}' → '{canonical}'")

    return "\n".join(context_parts)


def format_extraction_prompt(text: str, taxonomy: dict | None = None) -> str:
    """Format the extraction prompt with text and taxonomy context.

    Args:
        text: Text to analyze
        taxonomy: Taxonomy dictionary or None

    Returns:
        Formatted prompt string
    """
    taxonomy_context = build_taxonomy_context(taxonomy)
    return GRAPH_EXTRACTION_PROMPT.format(
        taxonomy_context=taxonomy_context,
        text=text,
    )
