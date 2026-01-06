import json


def test_build_ontology_from_taxonomy_writes_file(tmp_path):
    from contextrouter.modules.ingestion.rag.settings import RagIngestionConfig
    from contextrouter.modules.transformers.ontology import build_ontology_from_taxonomy

    assets = tmp_path
    (assets / "output" / "_processing").mkdir(parents=True, exist_ok=True)
    (assets / "taxonomy.json").write_text("{}", encoding="utf-8")

    cfg = RagIngestionConfig.model_validate(
        {
            "paths": {
                "assets_folder": str(assets),
                "source_dir": "source",
                "clean_text_dir": "clean_text",
                "shadow_dir": "shadow",
                "upload_dir": "output",
                "jsonl_dir": "jsonl",
                "processing_dir": "_processing",
            }
        }
    )

    out = build_ontology_from_taxonomy(config=cfg, overwrite=True)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert "relations" in data and "allowed_labels" in data["relations"]
