from src import prompt


def test_frozen_prompt_contract():
    meta = {
        "target_name": "TEST_PROTEIN",
        "organism": "Test organism",
        "fitness_description": "measured activity",
    }
    mutants = [("A1V", "MUTANT_A", 1.0), ("A1G", "MUTANT_B", 0.0)]
    user, ids = prompt.build_user_prompt(meta, "WILDTYPE", mutants)

    assert prompt.PROMPT_VERSION == "ranking-v1"
    assert ids == ["M01", "M02"]
    assert "**Assay (what is measured):** measured activity" in user
    assert (
        user.count("**Higher experimental fitness = HIGHER value of the measured property.**") == 1
    )
    assert "A1V" not in user
    assert "M01: MUTANT_A" in user


def test_ranking_parser_and_score_direction():
    ids = [f"M{i:02d}" for i in range(1, 11)]
    parsed = prompt.parse_ranking(
        '{"ranking":["M10","M9","M8","M7","M6","M5","M4","M3","M2","M1"]}',
        ids,
    )
    mutants = [(f"v{i}", "SEQ", float(i)) for i in range(10)]
    assert parsed is not None
    assert prompt.score_ranking(parsed, ids, mutants) == 1.0


def test_partial_or_duplicate_rankings_are_not_scored():
    ids = ["M01", "M02", "M03"]
    assert prompt.parse_ranking('{"ranking":["M01","M02"]}', ids) is None
    assert prompt.parse_ranking('{"ranking":["M01","M02","M02","M03"]}', ids) is None


def test_spearman_ties_and_constants():
    assert abs(prompt.spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12
    assert abs(prompt.spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
    assert prompt.spearman([1, 2, 3], [5, 5, 5]) == 0.0


def test_parse_ranking_rejects_out_of_range_adjacent_digit_ids():
    ids = [f"M{i:02d}" for i in range(1, 101)]
    # An out-of-range "M1000" must not be mis-tokenized as the valid id "M100"
    # and silently complete an otherwise-99-item ranking.
    tokens = ids[:99] + ["M1000"]
    blob = '{"ranking": [' + ", ".join(f'"{token}"' for token in tokens) + "]}"
    assert prompt.parse_ranking(blob, ids) is None
    # An exact ranking (including the three-digit M100) still parses.
    good = '{"ranking": [' + ", ".join(f'"{token}"' for token in ids) + "]}"
    assert prompt.parse_ranking(good, ids) == ids
