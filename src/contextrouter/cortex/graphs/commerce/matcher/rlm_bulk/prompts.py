"""Prompts and instructions for RLM Bulk Matcher."""

from typing import Any


def build_matching_prompt(
    supplier_products: list[dict[str, Any]],
    site_products: list[dict[str, Any]],
    taxonomies: dict[str, Any] | None = None,
    manual_matches: list[dict[str, Any]] | None = None,
    wrong_pairs: list[dict[str, Any]] | None = None,
    out_path: str = "",
) -> str:
    """Build RLM prompt referencing pre-loaded REPL variables."""
    import json

    brand_name = ""
    if supplier_products:
        brand_name = supplier_products[0].get("brand", "")

    # Compact samples — just 2 items, truncated
    sup_sample = json.dumps(supplier_products[:2], ensure_ascii=False)[:400]
    site_sample = json.dumps(site_products[:2], ensure_ascii=False)[:400]

    # Wrong pairs info for negative examples
    wrong_info = ""
    if wrong_pairs:
        wrong_sample = json.dumps(wrong_pairs[:5], ensure_ascii=False)[:600]
        wrong_info = f"""
## WRONG MATCHES (NEGATIVE EXAMPLES — DO NOT REPEAT):
- {len(wrong_pairs)} pairs marked as wrong by human operator
- Variable `wrong_pairs` has: supplier_name, wrong_oscar_title, reason
- Before confirming a match, check if this supplier+site pair exists in wrong_pairs
- Sample: {wrong_sample}
"""

    return f"""
{DEEP_MATCHING_INSTRUCTIONS}

## THIS BATCH: brand "{brand_name}"
- {len(supplier_products)} supplier products
- {len(site_products)} site products
- out_path = "{out_path}"
{wrong_info}
Sample supplier: {sup_sample}
Sample site: {site_sample}

Execute your matching code NOW. Save results to out_path. Print "DONE".
"""


DEEP_MATCHING_INSTRUCTIONS = """\
## CRITICAL: ACT IMMEDIATELY
You are an autonomous matching agent. DO NOT ask questions. DO NOT wait.
Write Python code in the REPL and execute it NOW.

## TASK: Match supplier products to site products within a single brand.

You have these REPL variables (already loaded, DO NOT reload):
- `supplier_products` — list of dicts with keys: id, name, brand, sku, ean, price_retail_uah, params (color, size)
- `site_products` — list of dicts with keys: id, title, parent_title, upc, manufacturer_sku, brand, product_type, category, structure
  NOTE: Site products use "title" (not "name") and "upc" / "manufacturer_sku" (not "sku"). Use these Oscar field names!
- `taxonomies` — dict with "colors" (list of color names like ["Blue", "Red"]) and "sizes" (list of size strings like ["S", "M", "L"])
- `manual_matches` — list of previously approved matches (for learning patterns)
- `wrong_pairs` — list of matches rejected by human operator. Each has: supplier_name, wrong_oscar_title, reason. NEVER repeat these.
- `out_path` — file path where you MUST save results

## MATCHING ALGORITHM (execute in ONE code block)

```python
import json, re

# Step 1: Build lookup index with color/size extraction
def extract_model(name, brand):
    # Extract model identifier by stripping brand, colors, sizes, filler.
    text = str(name).lower().strip()
    if brand:
        text = re.sub(r'(?i)\\b' + re.escape(brand.lower()) + r'\\b', '', text)
    # Strip known colors/sizes from taxonomies (plain string lists)
    for cn in taxonomies.get('colors', []):
        if cn and len(cn) > 1:
            text = re.sub(r'(?i)\\b' + re.escape(cn.lower()) + r'\\b', '', text)
    for sn in taxonomies.get('sizes', []):
        if sn:
            text = re.sub(r'(?i)(?:^|\\s|-)' + re.escape(sn.lower()) + r'(?=$|\\s|-)', ' ', text)
    text = re.sub(r'[()\\[\\]/,]+', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def extract_color(name):
    # Extract color from product name using taxonomy
    text_lower = str(name).lower()
    for cn in taxonomies.get('colors', []):
        if cn and len(cn) > 1 and re.search(r'(?i)\\b' + re.escape(cn.lower()) + r'\\b', text_lower):
            return cn.lower()
    return ''

def extract_size(name):
    # Extract size from product name using taxonomy
    text_lower = str(name).lower()
    for sn in taxonomies.get('sizes', []):
        if sn and re.search(r'(?i)(?:^|\\s|-)' + re.escape(sn.lower()) + r'(?=$|\\s|-)', text_lower):
            return sn.lower()
    return ''

# Build wrong-match set for quick lookup
wrong_set = set()
for wp in wrong_pairs:
    key = (str(wp.get('supplier_name','')).lower().strip(), str(wp.get('wrong_oscar_title','')).lower().strip())
    wrong_set.add(key)

# Step 2: Index site products by model (using 'title', not 'name'!)
site_index = {}
site_upc_index = {}  # Also index by UPC and Manufacturer SKU
for sp in site_products:
    # Site products use 'title' and 'upc', NOT 'name' and 'sku'
    title = sp.get('title', '') or sp.get('parent_title', '') or ''
    model = extract_model(title, sp.get('brand', ''))
    if model:
        site_index.setdefault(model, []).append(sp)
    upc = str(sp.get('upc') or '').strip()
    m_sku = str(sp.get('manufacturer_sku') or '').strip()
    if upc:
        site_upc_index[upc] = sp
    if m_sku:
        site_upc_index[m_sku] = sp

# Step 3: Match each supplier product
matches = []
for sup in supplier_products:
    sup_name = sup.get('name', '')
    sup_sku = str(sup.get('sku') or '').strip()
    sup_ean = str(sup.get('ean') or '').strip()
    sup_model = extract_model(sup_name, sup.get('brand', ''))
    sup_color = extract_color(sup_name) or (sup.get('params', {}).get('color', '') or '').lower()
    sup_size = extract_size(sup_name) or (sup.get('params', {}).get('size', '') or '').lower()
    best_match = None
    best_score = 0
    match_type = "unmatched"

    # Strategy 1: SKU / EAN exact match
    if sup_sku and sup_sku in site_upc_index:
        best_match = site_upc_index[sup_sku]
        best_score = 0.98
        match_type = "exact_sku"
    elif sup_ean and sup_ean in site_upc_index:
        best_match = site_upc_index[sup_ean]
        best_score = 0.98
        match_type = "exact_ean"

    # Strategy 2: Exact model match WITH color/size verification
    if not best_match and sup_model and sup_model in site_index:
        for site_p in site_index[sup_model]:
            site_title = site_p.get('title','') or site_p.get('parent_title','') or ''
            site_color = extract_color(site_title)
            site_size = extract_size(site_title)
            # Color/size must match or be absent on one side
            color_ok = (not sup_color or not site_color or sup_color == site_color)
            size_ok = (not sup_size or not site_size or sup_size == site_size)
            if color_ok and size_ok:
                best_score = 0.9
                best_match = site_p
                match_type = "model_decomposition"
                break

    # Strategy 3: Fuzzy model match WITH color/size verification
    if not best_match and sup_model:
        sup_tokens = set(sup_model.split())
        sup_category = str(sup.get('category') or '').lower()
        if len(sup_tokens) >= 2:  # Need at least 2 tokens for meaningful fuzzy
            for model_key, site_list in site_index.items():
                site_tokens = set(model_key.split())
                if site_tokens:
                    overlap = len(sup_tokens & site_tokens) / max(len(sup_tokens), len(site_tokens))
                    if overlap > 0.5 and overlap > best_score:
                        # Color/size verification for fuzzy matches too
                        site_p = site_list[0]
                        site_title = site_p.get('title','') or site_p.get('parent_title','') or ''
                        site_color = extract_color(site_title)
                        site_size = extract_size(site_title)
                        color_ok = (not sup_color or not site_color or sup_color == site_color)
                        size_ok = (not sup_size or not site_size or sup_size == site_size)
                        if not (color_ok and size_ok):
                            continue

                        # Category / Product Type check
                        site_type = str(site_p.get('product_type') or site_p.get('category') or '').lower()
                        cat_boost = 0.05 if (sup_category and (sup_category in site_type or site_type in sup_category)) else 0.0

                        best_score = overlap + cat_boost
                        best_match = site_p
                        match_type = "fuzzy_model"

    # Step 3.5: Check against wrong_pairs — reject known bad pairs
    if best_match:
        site_title = best_match.get('title', '') or best_match.get('parent_title', '') or ''
        check_key = (sup_name.lower().strip(), site_title.lower().strip())
        if check_key in wrong_set:
            best_match = None
            best_score = 0
            match_type = "unmatched"

    if best_match:
        site_title = best_match.get('title', '') or best_match.get('parent_title', '') or ''
        matches.append({
            "supplier_id": str(sup.get('id', '')),
            "supplier_name": sup_name,
            "site_id": str(best_match.get('id', '')),
            "site_name": site_title,
            "confidence": round(best_score, 2),
            "match_type": match_type,
            "notes": f"model: {sup_model}"
        })
    else:
        matches.append({
            "supplier_id": str(sup.get('id', '')),
            "supplier_name": sup_name,
            "site_id": None,
            "site_name": None,
            "confidence": 0,
            "match_type": "unmatched",
            "notes": f"model: {sup_model}, no site match"
        })

# Step 4: SAVE — this is CRITICAL
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(matches, f, ensure_ascii=False)
print(f"DONE: {len([m for m in matches if m['site_id']])} matched, {len([m for m in matches if not m['site_id']])} unmatched")
```

## IMPORTANT RULES
1. The code above is a TEMPLATE. Execute it AS-IS with minimal adaptation. Do NOT restructure or rename variables.
2. You MUST save results to `out_path` using `json.dump()`. This is how results are collected.
3. Do NOT use FINAL_VAR(). Do NOT print the full JSON. Just save to out_path and print "DONE".
4. Each match dict must have: supplier_id, supplier_name, site_id (or null), site_name (or null), confidence (0-1), match_type, notes.
5. Use the variable names exactly as shown: `matches`, `site_index`, `site_upc_index`, `best_match`. Do NOT invent new names like `brand_matches` etc.
6. NEVER use triple-quotes (\\'\\'\\' or \\"\\"\\" ) in any code you generate. Use single-line strings or # comments only.
7. If using llm_query(), pass the prompt as a single-quoted string, never triple-quoted.
8. CRITICAL: Color and size MUST match between supplier and site products. "black XL" != "red S". Only match if colors AND sizes are compatible.
9. CRITICAL: Check `wrong_pairs` before confirming any match. Never match a pair that was rejected by the operator.
"""
