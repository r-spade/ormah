from eval.maintenance.report import agreement


def test_agreement_and_gate():
    single = {"p1": "none", "p2": "supports", "p3": "none", "p4": "related_to"}
    batched = {"p1": "none", "p2": "supports", "p3": "related_to", "p4": "related_to"}
    r = agreement(single, batched)
    assert r["n"] == 4
    assert r["agree_rate"] == 0.75
    assert r["none_to_edge_rate"] == 0.5      # 1 of the 2 'none' singles flipped
    assert r["gate_pass"] is False


def test_gate_passes_at_thresholds():
    single = {f"p{i}": "none" for i in range(10)}
    batched = dict(single, p0="related_to")   # 0.9 agreement, 0.1 none->edge
    r = agreement(single, batched)
    assert r["agree_rate"] == 0.9 and r["none_to_edge_rate"] == 0.1
    assert r["gate_pass"] is True
