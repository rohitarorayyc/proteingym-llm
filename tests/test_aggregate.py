from src.aggregate import macro_headline, nested_macro


def test_nested_macro_deduplicates_proteins_and_equal_weights_groups():
    meta = {
        "a1": {"uniprot_id": "P1", "function": "Activity"},
        "a2": {"uniprot_id": "P1", "function": "Activity"},
        "a3": {"uniprot_id": "P2", "function": "Activity"},
        "a4": {"uniprot_id": "P3", "function": "Stability"},
    }
    by_group = nested_macro({"a1": 0.2, "a2": 0.4, "a3": 0.9, "a4": 0.1}, meta)
    assert abs(by_group["Activity"]["mean_rho"] - 0.6) < 1e-12
    assert abs(by_group["Stability"]["mean_rho"] - 0.1) < 1e-12
    assert abs(macro_headline(by_group) - 0.35) < 1e-12


def test_nested_macro_drops_missing_metadata():
    meta = {
        "good": {"uniprot_id": "P1", "function": "Activity"},
        "no_id": {"uniprot_id": "", "function": "Activity"},
        "no_group": {"uniprot_id": "P2", "function": "?"},
    }
    result = nested_macro({"good": 0.5, "no_id": 0.9, "no_group": 0.9}, meta)
    assert set(result) == {"Activity"}
    assert result["Activity"]["n_assays"] == 1
