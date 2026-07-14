from src.subsample import stratified_sample


def _rows(n: int):
    return [(f"v{i}", f"SEQ{i}", float(i)) for i in range(n)]


def test_stratified_sample_is_deterministic_and_shuffled():
    first = stratified_sample(_rows(1000), 50, strata=10, seed=2)
    second = stratified_sample(_rows(1000), 50, strata=10, seed=2)
    assert first == second
    assert len(first) == len({variant for variant, _, _ in first}) == 50
    assert [score for _, _, score in first] != sorted(score for _, _, score in first)


def test_stratified_sample_spans_the_distribution():
    sample = stratified_sample(_rows(1000), 50, strata=10, seed=1)
    scores = [score for _, _, score in sample]
    assert min(scores) < 100
    assert max(scores) >= 900


def test_small_pool_returns_every_variant():
    sample = stratified_sample(_rows(7), 50, seed=1)
    assert {variant for variant, _, _ in sample} == {f"v{i}" for i in range(7)}
