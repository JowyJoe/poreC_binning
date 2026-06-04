"""Conservative refine scoring from feature-anchored HG-VAE embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from porebin_genome.coarse.hyperedge_embedding import load_hyperedge_embedding_tsv
from porebin_genome.io.tables import write_tsv_rows
from porebin_genome.refine.action_features import ActionFeatureRow
from porebin_genome.refine.models import RefineState, SplitCandidate


EMBEDDING_SCORER_MODES = ("off", "report", "veto")
DEFAULT_EMBEDDING_THRESHOLDS = {
    "split": 0.05,
    "reassign": 0.03,
    "merge": 0.00,
    "recruit": 0.03,
}
EMBEDDING_SCORE_COLUMNS = (
    "action_type",
    "contig_id",
    "source_bin",
    "target_bin",
    "rule_reason",
    "rule_accepted",
    "embedding_score",
    "threshold",
    "embedding_decision",
    "final_accepted",
    "final_reason",
    "scorer_status",
    "within_similarity",
    "between_similarity",
    "margin",
    "note",
)


class EmbeddingScorerError(RuntimeError):
    """Raised when embedding scoring cannot be configured or evaluated."""


@dataclass(frozen=True)
class EmbeddingScoreRow:
    """Audit row for one candidate action after optional embedding scoring."""

    action_type: str
    contig_id: str
    source_bin: str
    target_bin: str
    rule_reason: str
    rule_accepted: bool
    embedding_score: float | None
    threshold: float | None
    embedding_decision: str
    final_accepted: bool
    final_reason: str
    scorer_status: str
    within_similarity: float | None
    between_similarity: float | None
    margin: float | None
    note: str


class HyperedgeEmbeddingActionScorer:
    """Use unsupervised HG-VAE embeddings to conservatively check refine actions."""

    def __init__(
        self,
        *,
        contig_ids: list[str],
        embeddings: object,
        support_weight_by_contig: dict[str, float],
        thresholds: dict[str, float] | None = None,
    ) -> None:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise EmbeddingScorerError("Embedding scoring requires numpy.") from exc

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise EmbeddingScorerError("Embedding matrix must be two-dimensional.")
        if len(contig_ids) != int(matrix.shape[0]):
            raise EmbeddingScorerError("Embedding row count does not match contig ids.")

        norms = np.linalg.norm(matrix, axis=1)
        mask = norms > 0.0
        matrix = matrix.copy()
        matrix[mask] = (matrix[mask].T / norms[mask]).T

        self.contig_to_idx = {str(contig_id): idx for idx, contig_id in enumerate(contig_ids)}
        self.embeddings = matrix
        self.support_weight_by_contig = {
            str(contig_id): float(value)
            for contig_id, value in support_weight_by_contig.items()
        }
        self.thresholds = dict(DEFAULT_EMBEDDING_THRESHOLDS)
        if thresholds is not None:
            self.thresholds.update({str(key): float(value) for key, value in thresholds.items()})

    @classmethod
    def load(cls, embedding_tsv: Path) -> "HyperedgeEmbeddingActionScorer":
        """Load a scorer from the coarse HG-VAE embedding TSV."""
        contigs, embeddings, support = load_hyperedge_embedding_tsv(embedding_tsv)
        return cls(contig_ids=contigs, embeddings=embeddings, support_weight_by_contig=support)

    def score_reassign(
        self,
        *,
        row: ActionFeatureRow,
        state: RefineState,
        mode: str,
        upstream_accepted: bool,
    ) -> EmbeddingScoreRow:
        """Score a candidate contig reassignment."""
        if not upstream_accepted:
            return _upstream_rejected(row)
        z = self._contig_vector(row.contig_id)
        target = self._bin_centroid(state, row.target_bin)
        source = self._bin_centroid(state, row.source_bin, exclude={row.contig_id})
        if z is None or target is None or source is None:
            return _insufficient(row, upstream_accepted=True, note="missing contig/source/target embedding")
        within = _cosine(z, target)
        between = _cosine(z, source)
        return _decision_row(
            row=row,
            action_type="reassign",
            mode=mode,
            score=float(within - between),
            threshold=float(self.thresholds["reassign"]),
            within_similarity=within,
            between_similarity=between,
            note="target_bin_similarity_minus_source_bin_similarity",
        )

    def score_recruit(
        self,
        *,
        row: ActionFeatureRow,
        state: RefineState,
        mode: str,
        upstream_accepted: bool,
    ) -> EmbeddingScoreRow:
        """Score recruitment of an unbinned contig into a target bin."""
        if not upstream_accepted:
            return _upstream_rejected(row)
        z = self._contig_vector(row.contig_id)
        target = self._bin_centroid(state, row.target_bin)
        if z is None or target is None:
            return _insufficient(row, upstream_accepted=True, note="missing contig/target embedding")
        target_sim = _cosine(z, target)
        runner_up = self._nearest_other_bin_similarity(state, vector=z, exclude_bins={row.target_bin})
        other_sim = 0.0 if runner_up is None else float(runner_up)
        return _decision_row(
            row=row,
            action_type="recruit",
            mode=mode,
            score=float(target_sim - other_sim),
            threshold=float(self.thresholds["recruit"]),
            within_similarity=target_sim,
            between_similarity=other_sim,
            note="target_bin_similarity_minus_runner_up_bin_similarity",
        )

    def score_merge(
        self,
        *,
        row: ActionFeatureRow,
        state: RefineState,
        mode: str,
        upstream_accepted: bool,
    ) -> EmbeddingScoreRow:
        """Score a candidate bin merge."""
        if not upstream_accepted:
            return _upstream_rejected(row)
        source = self._bin_centroid(state, row.source_bin)
        target = self._bin_centroid(state, row.target_bin)
        if source is None or target is None:
            return _insufficient(row, upstream_accepted=True, note="missing source/target bin embedding")
        pair_sim = _cosine(source, target)
        source_runner = self._nearest_other_bin_similarity(
            state,
            vector=source,
            exclude_bins={row.source_bin, row.target_bin},
        )
        target_runner = self._nearest_other_bin_similarity(
            state,
            vector=target,
            exclude_bins={row.source_bin, row.target_bin},
        )
        alternatives = [value for value in (source_runner, target_runner) if value is not None]
        runner_up = max(alternatives) if alternatives else 0.0
        return _decision_row(
            row=row,
            action_type="merge",
            mode=mode,
            score=float(pair_sim - runner_up),
            threshold=float(self.thresholds["merge"]),
            within_similarity=pair_sim,
            between_similarity=runner_up,
            note="source_target_bin_similarity_minus_nearest_alternative_similarity",
        )

    def score_split(
        self,
        *,
        row: ActionFeatureRow,
        candidate: SplitCandidate,
        state: RefineState,
        mode: str,
        upstream_accepted: bool,
    ) -> EmbeddingScoreRow:
        """Score whether proposed child groups are separated in embedding space."""
        if not upstream_accepted:
            return _upstream_rejected(row)
        groups = [tuple(group) for group in candidate.groups if group]
        if len(groups) != 2:
            return _insufficient(row, upstream_accepted=True, note="split candidate is not two-way")
        left = self._group_profile(state, groups[0])
        right = self._group_profile(state, groups[1])
        if left is None or right is None:
            return _insufficient(row, upstream_accepted=True, note="missing child-group embedding")
        left_centroid, left_cohesion, left_weight = left
        right_centroid, right_cohesion, right_weight = right
        total = max(float(left_weight + right_weight), 1.0e-12)
        within = float((left_weight / total) * left_cohesion + (right_weight / total) * right_cohesion)
        between = _cosine(left_centroid, right_centroid)
        return _decision_row(
            row=row,
            action_type="split",
            mode=mode,
            score=float(within - between),
            threshold=float(self.thresholds["split"]),
            within_similarity=within,
            between_similarity=between,
            note="weighted_child_cohesion_minus_child_centroid_similarity",
        )

    def _contig_vector(self, contig_id: str) -> object | None:
        import numpy as np

        idx = self.contig_to_idx.get(str(contig_id))
        if idx is None:
            return None
        if float(self.support_weight_by_contig.get(str(contig_id), 0.0)) <= 0.0:
            return None
        vector = self.embeddings[idx]
        if float(np.linalg.norm(vector)) <= 0.0:
            return None
        return vector

    def _bin_centroid(
        self,
        state: RefineState,
        bin_id: str,
        *,
        exclude: set[str] | None = None,
    ) -> object | None:
        members = state.bin_to_contigs().get(str(bin_id), [])
        if exclude:
            members = [contig_id for contig_id in members if contig_id not in exclude]
        profile = self._group_profile(state, members)
        return None if profile is None else profile[0]

    def _group_profile(
        self,
        state: RefineState,
        contig_ids: Iterable[str],
    ) -> tuple[object, float, float] | None:
        import numpy as np

        vectors = []
        weights = []
        for contig_id in contig_ids:
            vector = self._contig_vector(str(contig_id))
            if vector is None:
                continue
            weight = max(1.0, float(state.contig_lengths.get(str(contig_id), 1)))
            vectors.append(vector)
            weights.append(weight)
        if not vectors:
            return None
        matrix = np.vstack(vectors)
        weight_values = np.asarray(weights, dtype=np.float64)
        centroid = np.average(matrix, axis=0, weights=weight_values)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0.0:
            return None
        centroid = (centroid / norm).astype(np.float32, copy=False)
        similarities = matrix @ centroid
        cohesion = float(np.average(similarities, weights=weight_values))
        return centroid, cohesion, float(weight_values.sum())

    def _nearest_other_bin_similarity(
        self,
        state: RefineState,
        *,
        vector: object,
        exclude_bins: set[str],
    ) -> float | None:
        values: list[float] = []
        for bin_id in sorted(state.bin_to_contigs().keys()):
            if str(bin_id) in {str(value) for value in exclude_bins}:
                continue
            centroid = self._bin_centroid(state, str(bin_id))
            if centroid is None:
                continue
            values.append(_cosine(vector, centroid))
        if not values:
            return None
        return float(max(values))


def normalize_embedding_scorer_mode(value: str) -> str:
    """Normalize public embedding-scorer mode strings."""
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in EMBEDDING_SCORER_MODES:
        raise ValueError("--embedding-scorer-mode must be off, report, or veto.")
    return mode


def write_embedding_scores_tsv(*, rows: list[EmbeddingScoreRow], out_path: Path) -> None:
    """Write embedding action-score audit rows."""
    write_tsv_rows(
        out_path,
        EMBEDDING_SCORE_COLUMNS,
        (
            (
                row.action_type,
                row.contig_id,
                row.source_bin,
                row.target_bin,
                row.rule_reason,
                int(row.rule_accepted),
                _format_optional_float(row.embedding_score),
                _format_optional_float(row.threshold),
                row.embedding_decision,
                int(row.final_accepted),
                row.final_reason,
                row.scorer_status,
                _format_optional_float(row.within_similarity),
                _format_optional_float(row.between_similarity),
                _format_optional_float(row.margin),
                row.note,
            )
            for row in rows
        ),
    )


def append_embedding_note(note: str, score: EmbeddingScoreRow | None) -> str:
    """Append compact embedding audit information to a refine action note."""
    if score is None or score.scorer_status != "scored":
        return str(note)
    return (
        f"{note};embedding_score={float(score.embedding_score or 0.0):.3f};"
        f"embedding_threshold={float(score.threshold or 0.0):.3f};"
        f"embedding_decision={score.embedding_decision}"
    )


def _decision_row(
    *,
    row: ActionFeatureRow,
    action_type: str,
    mode: str,
    score: float,
    threshold: float,
    within_similarity: float,
    between_similarity: float,
    note: str,
) -> EmbeddingScoreRow:
    passed = bool(float(score) >= float(threshold))
    if passed:
        decision = "pass"
        final_accepted = True
        final_reason = row.rule_reason
    elif mode == "veto":
        decision = "veto"
        final_accepted = False
        final_reason = "embedding_veto_low_margin"
    else:
        decision = "would_veto"
        final_accepted = True
        final_reason = row.rule_reason
    return EmbeddingScoreRow(
        action_type=action_type,
        contig_id=row.contig_id,
        source_bin=row.source_bin,
        target_bin=row.target_bin,
        rule_reason=row.rule_reason,
        rule_accepted=True,
        embedding_score=float(score),
        threshold=float(threshold),
        embedding_decision=decision,
        final_accepted=bool(final_accepted),
        final_reason=final_reason,
        scorer_status="scored",
        within_similarity=float(within_similarity),
        between_similarity=float(between_similarity),
        margin=float(score),
        note=note,
    )


def _upstream_rejected(row: ActionFeatureRow) -> EmbeddingScoreRow:
    return EmbeddingScoreRow(
        action_type=row.action_type,
        contig_id=row.contig_id,
        source_bin=row.source_bin,
        target_bin=row.target_bin,
        rule_reason=row.rule_reason,
        rule_accepted=False,
        embedding_score=None,
        threshold=None,
        embedding_decision="not_evaluated",
        final_accepted=False,
        final_reason=row.rule_reason,
        scorer_status="upstream_rejected",
        within_similarity=None,
        between_similarity=None,
        margin=None,
        note="rule_or_previous_scorer_rejected",
    )


def _insufficient(row: ActionFeatureRow, *, upstream_accepted: bool, note: str) -> EmbeddingScoreRow:
    return EmbeddingScoreRow(
        action_type=row.action_type,
        contig_id=row.contig_id,
        source_bin=row.source_bin,
        target_bin=row.target_bin,
        rule_reason=row.rule_reason,
        rule_accepted=bool(upstream_accepted),
        embedding_score=None,
        threshold=None,
        embedding_decision="not_evaluated",
        final_accepted=bool(upstream_accepted),
        final_reason=row.rule_reason,
        scorer_status="insufficient_embedding",
        within_similarity=None,
        between_similarity=None,
        margin=None,
        note=str(note),
    )


def _cosine(left: object, right: object) -> float:
    import numpy as np

    return float(np.dot(left, right))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.8g}"
