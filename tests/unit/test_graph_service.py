import pickle


def test_graph_service_case_insensitive_lookup(tmp_path):
    import networkx as nx

    from contextrouter.cortex.services.graph import GraphService

    g = nx.Graph()
    g.add_edge("Alpha", "Beta", relation="CAUSES")
    p = tmp_path / "knowledge_graph.pickle"
    with open(p, "wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    svc = GraphService(graph_path=p, taxonomy_path=None)
    assert "Alpha" in svc.get_context("alpha")
    facts = svc.get_facts(["ALPHA"])
    assert any("Alpha" in f and "CAUSES" in f for f in facts)


def test_graph_service_ontology_filters_facts(tmp_path):
    import json

    import networkx as nx

    from contextrouter.cortex.services.graph import GraphService

    g = nx.Graph()
    g.add_edge("Alpha", "Beta", relation="RELATED_TO")
    gp = tmp_path / "knowledge_graph.pickle"
    with open(gp, "wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    onto = {
        "version": "1.0",
        "relations": {"runtime_fact_labels": ["CAUSES"]},
    }
    op = tmp_path / "ontology.json"
    op.write_text(json.dumps(onto), encoding="utf-8")

    svc = GraphService(graph_path=gp, taxonomy_path=None, ontology_path=op)
    assert svc.get_facts(["Alpha"]) == []
