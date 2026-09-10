from rag_project.intelligence.entity_coverage import extract_query_entities


def test_query_entity_inventory_retains_abbreviation_alias_and_canonical_term():
    entities = extract_query_entities("Compare HbA1c 7% with HTA")
    assert "hba1c" in entities
    assert "7%" in entities
    assert "hta" in entities
    assert "hypertension" in entities


def test_query_entity_inventory_does_not_replace_other_explicit_aliases():
    entities = extract_query_entities("HTA and HbA1c")
    assert "hta" in entities
    assert "hypertension" in entities
    assert "hba1c" in entities
