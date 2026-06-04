"""Conservative ML scoring for candidate refine actions."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from porebin_genome.io.tables import write_tsv_rows
from porebin_genome.refine.action_features import ActionFeatureRow


ACTION_SCORER_MODES = ("off", "features_only", "score")
DEFAULT_ACTION_THRESHOLDS = {
    "split": 0.90,
    "reassign": 0.95,
    "merge": 0.95,
    "recruit": 0.90,
}
DEFAULT_FEATURE_NAMES = (
    "confidence",
    "delta_contact",
    "support_edge_count",
    "support_weight_sum",
    "support_reliability_median",
    "support_k_eff_median",
    "current_support",
    "target_support",
    "runner_up_support",
    "cross_support",
    "source_bin_n_contigs",
    "target_bin_n_contigs",
    "contig_length",
)
ACTION_SCORE_COLUMNS = (
    "action_type",
    "contig_id",
    "source_bin",
    "target_bin",
    "rule_reason",
    "rule_accepted",
    "ml_score",
    "threshold",
    "ml_decision",
    "final_accepted",
    "final_reason",
    "scorer_status",
)


class ActionScorerError(RuntimeError):
    """Raised when an action scorer artifact is invalid or cannot score."""


@dataclass(frozen=True)
class ActionScoreRow:
    """Audit row for one rule decision after optional ML scoring."""

    action_type: str
    contig_id: str
    source_bin: str
    target_bin: str
    rule_reason: str
    rule_accepted: bool
    ml_score: float | None
    threshold: float | None
    ml_decision: str
    final_accepted: bool
    final_reason: str
    scorer_status: str


class ConservativeActionScorer:
    """Score candidate actions but only allow ML to veto accepted rule decisions."""

    def __init__(
        self,
        *,
        models: Mapping[str, object],
        thresholds: Mapping[str, float],
        feature_names: tuple[str, ...],
    ) -> None:
        self.models = {str(key): value for key, value in dict(models).items()}
        self.thresholds = {
            str(key): float(value)
            for key, value in dict(DEFAULT_ACTION_THRESHOLDS, **dict(thresholds)).items()
        }
        self.feature_names = tuple(str(value) for value in feature_names)
        if not self.feature_names:
            raise ActionScorerError("feature_schema.json must define at least one feature.")

    @classmethod
    def load(cls, model_dir: Path) -> "ConservativeActionScorer":
        """Load a conservative scorer artifact directory."""
        model_dir = model_dir.resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"Action scorer model directory not found: {model_dir}")
        if not model_dir.is_dir():
            raise ActionScorerError(f"Action scorer model path is not a directory: {model_dir}")

        thresholds = _load_thresholds(model_dir / "thresholds.json")
        feature_names = _load_feature_names(model_dir / "feature_schema.json")
        model_obj = _load_model_object(model_dir / "model.joblib")
        models = _normalize_models(model_obj)
        return cls(models=models, thresholds=thresholds, feature_names=feature_names)

    def score(self, row: ActionFeatureRow) -> ActionScoreRow:
        """Return the final conservative decision for one action feature row."""
        action_type = str(row.action_type)
        if not bool(row.rule_accepted):
            return ActionScoreRow(
                action_type=action_type,
                contig_id=row.contig_id,
                source_bin=row.source_bin,
                target_bin=row.target_bin,
                rule_reason=row.rule_reason,
                rule_accepted=False,
                ml_score=None,
                threshold=None,
                ml_decision="not_evaluated",
                final_accepted=False,
                final_reason=row.rule_reason,
                scorer_status="rule_rejected",
            )

        model = self.models.get(action_type)
        threshold = float(self.thresholds.get(action_type, 1.0))
        if model is None:
            return ActionScoreRow(
                action_type=action_type,
                contig_id=row.contig_id,
                source_bin=row.source_bin,
                target_bin=row.target_bin,
                rule_reason=row.rule_reason,
                rule_accepted=True,
                ml_score=None,
                threshold=threshold,
                ml_decision="not_evaluated",
                final_accepted=True,
                final_reason=row.rule_reason,
                scorer_status="model_disabled_for_action",
            )

        score = _clip01(_predict_score(model, _feature_vector(row, self.feature_names)))
        final_accepted = bool(score >= threshold)
        return ActionScoreRow(
            action_type=action_type,
            contig_id=row.contig_id,
            source_bin=row.source_bin,
            target_bin=row.target_bin,
            rule_reason=row.rule_reason,
            rule_accepted=True,
            ml_score=float(score),
            threshold=threshold,
            ml_decision=("pass" if final_accepted else "veto"),
            final_accepted=final_accepted,
            final_reason=(row.rule_reason if final_accepted else "ml_veto_low_score"),
            scorer_status="scored",
        )


def normalize_action_scorer_mode(value: str) -> str:
    """Normalize public action-scorer mode strings."""
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in ACTION_SCORER_MODES:
        raise ValueError("--action-scorer-mode must be off, features-only, or score.")
    return mode


def write_action_scores_tsv(*, rows: list[ActionScoreRow], out_path: Path) -> None:
    """Write action scoring audit rows."""
    write_tsv_rows(
        out_path,
        ACTION_SCORE_COLUMNS,
        (
            (
                row.action_type,
                row.contig_id,
                row.source_bin,
                row.target_bin,
                row.rule_reason,
                int(row.rule_accepted),
                _format_optional_float(row.ml_score),
                _format_optional_float(row.threshold),
                row.ml_decision,
                int(row.final_accepted),
                row.final_reason,
                row.scorer_status,
            )
            for row in rows
        ),
    )


def append_ml_note(note: str, score: ActionScoreRow) -> str:
    """Append compact ML audit information to a refine action note."""
    if score.scorer_status != "scored":
        return str(note)
    return (
        f"{note};ml_score={float(score.ml_score or 0.0):.3f};"
        f"ml_threshold={float(score.threshold or 0.0):.3f};"
        f"ml_decision={score.ml_decision}"
    )


def _load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_ACTION_THRESHOLDS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ActionScorerError("thresholds.json must contain an object.")
    return {str(key): float(value) for key, value in payload.items()}


def _load_feature_names(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return tuple(DEFAULT_FEATURE_NAMES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get("feature_names")
    else:
        values = payload
    if not isinstance(values, list):
        raise ActionScorerError("feature_schema.json must contain feature_names as a list.")
    return tuple(str(value) for value in values)


def _load_model_object(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Action scorer model.joblib not found: {path}")
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as fh:
            return pickle.load(fh)


def _normalize_models(model_obj: object) -> dict[str, object]:
    if isinstance(model_obj, Mapping):
        if "models" in model_obj and isinstance(model_obj["models"], Mapping):
            return {str(key): value for key, value in dict(model_obj["models"]).items()}
        return {str(key): value for key, value in dict(model_obj).items()}
    return {action_type: model_obj for action_type in DEFAULT_ACTION_THRESHOLDS.keys()}


def _feature_vector(row: ActionFeatureRow, feature_names: tuple[str, ...]) -> list[float]:
    payload = asdict(row)
    return [_coerce_float(payload.get(name)) for name in feature_names]


def _coerce_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _predict_score(model: object, vector: list[float]) -> float:
    X = [vector]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        row = _first_row(probs)
        if len(row) >= 2:
            return float(row[1])
        if row:
            return float(row[0])
    if hasattr(model, "predict"):
        pred = model.predict(X)
        return float(_first_value(pred))
    if callable(model):
        pred = model(X)
        return float(_first_value(pred))
    raise ActionScorerError("Action model must implement predict_proba, predict, or be callable.")


def _first_row(values: object) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        return [float(value) for value in values[0]]
    if isinstance(values, tuple) and values and isinstance(values[0], (list, tuple)):
        return [float(value) for value in values[0]]
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def _first_value(values: object) -> float:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        if not values:
            return 0.0
        first = values[0]
        if isinstance(first, (list, tuple)):
            return float(first[-1] if first else 0.0)
        return float(first)
    return float(values)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.8g}"
