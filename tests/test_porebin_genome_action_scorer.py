from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from tests.porebin_genome_testkit import read_json, read_tsv_rows, write_reassign_refine_fixture


class ConstantProbModel:
    def __init__(self, score: float) -> None:
        self.score = float(score)

    def predict_proba(self, X):
        return [[1.0 - self.score, self.score] for _row in X]


def _write_model_dir(tmp_path: Path, *, score: float, action_type: str = "reassign") -> Path:
    model_dir = tmp_path / "action_scorer"
    model_dir.mkdir()
    (model_dir / "thresholds.json").write_text(
        json.dumps({action_type: 0.95}),
        encoding="utf-8",
    )
    with (model_dir / "model.joblib").open("wb") as fh:
        pickle.dump({action_type: ConstantProbModel(score)}, fh)
    return model_dir


def test_conservative_scorer_never_revives_rule_rejected_action() -> None:
    from porebin_genome.refine.action_features import ActionFeatureRow
    from porebin_genome.refine.action_scorer import ConservativeActionScorer

    scorer = ConservativeActionScorer(
        models={"reassign": ConstantProbModel(0.99)},
        thresholds={"reassign": 0.95},
        feature_names=("confidence", "delta_contact"),
    )
    row = ActionFeatureRow(
        action_type="reassign",
        contig_id="x",
        source_bin="0",
        target_bin="1",
        rule_reason="move_does_not_improve_contact_coherence",
        rule_accepted=False,
        confidence=0.0,
        delta_contact=-0.1,
        scg_status="abstain",
        support_edge_count=5,
        support_weight_sum=5.0,
        support_reliability_median=1.0,
        support_k_eff_median=2.0,
        current_support=1.0,
        target_support=2.0,
        runner_up_support=1.0,
        cross_support=None,
        source_bin_n_contigs=3,
        target_bin_n_contigs=3,
        contig_length=1000,
        feature_status="pass",
        coverage_status="pass",
        candidate_method="boundary_contig_move",
        candidate_reasons="synthetic",
    )

    score = scorer.score(row)

    assert score.ml_score is None
    assert score.final_accepted is False
    assert score.final_reason == "move_does_not_improve_contact_coherence"
    assert score.scorer_status == "rule_rejected"


def test_action_scorer_low_score_vetoes_rule_accepted_reassign(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for scorer integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_reassign_refine_fixture(tmp_path)
    model_dir = _write_model_dir(tmp_path, score=0.10)
    out_dir = tmp_path / "out"
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        enable_scg=False,
        action_scorer_mode="score",
        action_scorer_model=model_dir,
        out_dir=out_dir,
    )

    assignment = {row["contig_id"]: row["bin_id"] for row in read_tsv_rows(result.bins_refined_tsv)}
    actions = read_tsv_rows(result.refine_actions_tsv)
    scores = read_tsv_rows(result.refine_action_scores_tsv)
    meta = read_json(result.refine_meta_json)

    assert assignment["e"] == "0"
    assert any(row["reason"] == "ml_veto_low_score" for row in actions if row["action_type"] == "reassign")
    assert any(row["ml_decision"] == "veto" for row in scores)
    assert meta["n_ml_veto_actions"] >= 1


def test_action_scorer_high_score_keeps_rule_accepted_reassign(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for scorer integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_reassign_refine_fixture(tmp_path)
    model_dir = _write_model_dir(tmp_path, score=0.99)
    out_dir = tmp_path / "out"
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        enable_scg=False,
        action_scorer_mode="score",
        action_scorer_model=model_dir,
        out_dir=out_dir,
    )

    assignment = {row["contig_id"]: row["bin_id"] for row in read_tsv_rows(result.bins_refined_tsv)}
    actions = read_tsv_rows(result.refine_actions_tsv)
    scores = read_tsv_rows(result.refine_action_scores_tsv)

    assert assignment["e"] == assignment["c"]
    assert any(row["reason"] == "move_to_target_bin" for row in actions if row["action_type"] == "reassign")
    assert any(row["ml_decision"] == "pass" for row in scores)
