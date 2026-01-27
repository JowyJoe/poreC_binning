from __future__ import annotations

from porebin.pairwise_baseline import expand_to_pairs


def test_expand_to_pairs_k2() -> None:
    edges = list(expand_to_pairs(["A", "B"]))
    assert edges == [("A", "B", 1.0)]


def test_expand_to_pairs_k5_weights_sum_to_one() -> None:
    edges = list(expand_to_pairs(["A", "B", "C", "D", "E"]))
    assert len(edges) == 10
    weights = [w for _, _, w in edges]
    assert abs(sum(weights) - 1.0) < 1e-12
    assert all(abs(w - 0.1) < 1e-12 for w in weights)


def test_expand_to_pairs_dedup_and_no_self_loops() -> None:
    edges = list(expand_to_pairs(["A", "A", "B", "B", "C"]))
    pairs = {(a, b) for a, b, _ in edges}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}
    assert all(a < b for a, b, _ in edges)
    assert all(a != b for a, b, _ in edges)
    assert all(abs(w - (1.0 / 3.0)) < 1e-12 for _, _, w in edges)

