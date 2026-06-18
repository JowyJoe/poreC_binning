from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from porebin_genome.evidence.canonical import CanonicalContact
from porebin_genome.refinement.contact_index import (
    build_contact_index_from_contacts,
)
from porebin_genome.refinement.profiles import (
    EvidenceInputs,
    EvidenceProfileError,
    EvidenceProfileState,
    load_evidence_inputs,
)
from porebin_genome.evidence.scg.panel import DEFAULT_SCG_MARKERS
from porebin_genome.refinement.state import AssignmentChange, RefineState


def _fixture():
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(
                0,
                ["a", "b", "c"],
                [0.4, 0.3, 0.3],
                3,
                3,
                1.0,
            ),
            CanonicalContact(
                1,
                ["b", "c", "d"],
                [0.2, 0.5, 0.3],
                3,
                3,
                0.8,
            ),
        ],
        contig_names=("a", "b", "c", "d", "e"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray([0, 0, 1, 1, -1], dtype=np.int32),
        contig_lengths=np.asarray(
            [1000, 2000, 3000, 4000, 5000],
            dtype=np.int64,
        ),
        n_bins=2,
    )
    embedding = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 3.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    tnf = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    coverage = np.asarray(
        [9.0, 11.0, 99.0, 101.0, np.nan],
        dtype=np.float64,
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=embedding,
        tnf=tnf,
        coverage=coverage,
        scg_markers_by_contig={
            "a": ("GrpE",),
            "b": ("GrpE", "PGK"),
            "c": ("PGK", "TIGR00388"),
        },
        scg_orf_counts_by_contig={
            "a": {"GrpE": 2},
            "b": {"GrpE": 1, "PGK": 1},
            "c": {"PGK": 1, "TIGR00388": 2},
        },
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)
    return index, state, inputs, profiles


def test_bin_profiles_follow_the_four_explicit_formulas() -> None:
    _index, _state, _inputs, profiles = _fixture()
    bin0 = profiles.profile(0)

    expected_centroid = np.asarray([0.5, 0.5])
    expected_embedding_dispersion = math.sqrt(0.5)
    assert bin0.embedding.observed_count == 2
    assert bin0.embedding.centroid == pytest.approx(expected_centroid)
    assert bin0.embedding.dispersion == pytest.approx(
        expected_embedding_dispersion
    )
    assert bin0.embedding.distance_mad == pytest.approx(0.0)
    assert bin0.embedding.radius == pytest.approx(
        expected_embedding_dispersion
    )

    tnf_centroid = np.asarray([0.9, 0.1])
    tnf_distances = np.asarray(
        [
            _cosine_distance(np.asarray([1.0, 0.0]), tnf_centroid),
            _cosine_distance(np.asarray([0.8, 0.2]), tnf_centroid),
        ]
    )
    expected_tnf_median = float(np.median(tnf_distances))
    assert bin0.tnf.centroid == pytest.approx(tnf_centroid)
    assert bin0.tnf.distance_median == pytest.approx(expected_tnf_median)
    assert bin0.tnf.distance_mad == pytest.approx(
        float(np.median(np.abs(tnf_distances - expected_tnf_median)))
    )

    log_coverage = np.log1p(np.asarray([9.0, 11.0]))
    expected_cov_median = float(np.median(log_coverage))
    assert bin0.coverage.log_median == pytest.approx(expected_cov_median)
    assert bin0.coverage.log_mad == pytest.approx(
        float(np.median(np.abs(log_coverage - expected_cov_median)))
    )

    assert bin0.scg.unique_marker_count == 2
    assert bin0.scg.completeness_proxy == pytest.approx(2.0 / 107.0)
    assert bin0.scg.duplicate_burden == 1
    assert bin0.scg.duplicate_marker_ids == ("GrpE",)
    grpe_idx = DEFAULT_SCG_MARKERS.index("GrpE")
    assert bin0.scg.marker_contig_counts[grpe_idx] == 2
    assert bin0.scg.marker_orf_counts[grpe_idx] == 3


def test_same_contig_multiple_orfs_do_not_create_scg_duplication() -> None:
    _index, _state, _inputs, profiles = _fixture()

    one_contig = profiles.profile_for_members(9, [0])

    grpe_idx = DEFAULT_SCG_MARKERS.index("GrpE")
    assert one_contig.scg.marker_contig_counts[grpe_idx] == 1
    assert one_contig.scg.marker_orf_counts[grpe_idx] == 2
    assert one_contig.scg.duplicate_burden == 0


def test_scg_aliases_are_canonicalized_before_counting() -> None:
    _index, _state, _inputs, profiles = _fixture()
    bin1 = profiles.profile(1)

    assert "TIGR00389" in {
        marker
        for marker, count in zip(
            DEFAULT_SCG_MARKERS,
            bin1.scg.marker_contig_counts,
            strict=True,
        )
        if count
    }
    assert "TIGR00388" not in DEFAULT_SCG_MARKERS


def test_action_profiles_recompute_only_affected_bins_and_match_membership() -> None:
    _index, state, _inputs, profiles = _fixture()
    profiles.counters.reset()
    contact_delta = state.evaluate(
        [AssignmentChange(contig_idx=1, old_bin_idx=0, new_bin_idx=1)]
    )

    update = profiles.evaluate(contact_delta)

    assert profiles.counters.bin_profiles_built == 2
    assert profiles.counters.contigs_visited == 4
    assert tuple(item.bin_idx for item in update.transitions) == (0, 1)
    source, target = update.transitions
    assert source.before.member_count == 2
    assert source.after.member_count == 1
    assert source.scg_duplicate_change == -1
    assert target.before.member_count == 2
    assert target.after.member_count == 3
    assert target.scg_duplicate_change == 1
    assert source.embedding_gain is not None
    assert target.embedding_gain is not None


def test_profile_apply_and_rollback_track_contact_state_versions() -> None:
    _index, state, _inputs, profiles = _fixture()
    contact_delta = state.evaluate(
        [AssignmentChange(contig_idx=1, old_bin_idx=0, new_bin_idx=1)]
    )
    evidence_update = profiles.evaluate(contact_delta)
    original_source = profiles.profile(0)
    original_target = profiles.profile(1)

    with pytest.raises(EvidenceProfileError, match="contact ActionDelta"):
        profiles.apply(evidence_update)

    state.apply(contact_delta)
    profiles.apply(evidence_update)
    assert profiles.version == state.version == 1
    assert profiles.profile(0) == evidence_update.transitions[0].after
    assert profiles.profile(1) == evidence_update.transitions[1].after

    profiles.rollback(evidence_update)
    state.rollback(contact_delta)
    assert profiles.version == state.version == 0
    assert profiles.profile(0) == original_source
    assert profiles.profile(1) == original_target


def test_new_bin_profile_is_removed_after_rollback() -> None:
    _index, state, _inputs, profiles = _fixture()
    contact_delta = state.evaluate(
        [AssignmentChange(contig_idx=1, old_bin_idx=0, new_bin_idx=2)]
    )
    evidence_update = profiles.evaluate(contact_delta)

    state.apply(contact_delta)
    profiles.apply(evidence_update)
    assert profiles.profile(2).member_count == 1

    profiles.rollback(evidence_update)
    state.rollback(contact_delta)
    assert profiles.profile(2).member_count == 0


def test_sequential_profile_updates_follow_incremental_assignment() -> None:
    _index, state, inputs, profiles = _fixture()
    first_delta = state.evaluate(
        [AssignmentChange(contig_idx=1, old_bin_idx=0, new_bin_idx=1)]
    )
    first_evidence = profiles.evaluate(first_delta)
    state.apply(first_delta)
    profiles.apply(first_evidence)

    second_delta = state.evaluate(
        [AssignmentChange(contig_idx=4, old_bin_idx=-1, new_bin_idx=0)]
    )
    second_evidence = profiles.evaluate(second_delta)
    state.apply(second_delta)
    profiles.apply(second_evidence)

    rebuilt = EvidenceProfileState(refine_state=state, inputs=inputs)
    for bin_idx in range(state.n_bins):
        observed = profiles.profile(bin_idx)
        expected = rebuilt.profile(bin_idx)
        assert observed.member_count == expected.member_count
        assert observed.total_length == expected.total_length
        assert observed.embedding.dispersion == pytest.approx(
            expected.embedding.dispersion
        )
        assert observed.tnf.distance_median == pytest.approx(
            expected.tnf.distance_median
        )
        assert observed.coverage.log_mad == pytest.approx(
            expected.coverage.log_mad
        )
        assert observed.scg == expected.scg

    profiles.rollback(second_evidence)
    state.rollback(second_delta)
    profiles.rollback(first_evidence)
    state.rollback(first_delta)
    assert profiles.version == state.version == 0


def test_contig_bin_metrics_remain_independent_and_unthresholded() -> None:
    _index, _state, _inputs, profiles = _fixture()

    metrics = profiles.contig_bin_metrics(1, 1)

    assert metrics.embedding_distance is not None
    assert metrics.tnf_distance is not None
    assert metrics.coverage_distance is not None
    assert metrics.conflicting_scg_markers == ("PGK",)


def test_missing_evidence_is_reported_as_unavailable_not_zero_support() -> None:
    _index, _state, _inputs, profiles = _fixture()

    empty = profiles.profile_for_members(7, [4])
    metrics = profiles.contig_bin_metrics(4, 0)

    assert empty.embedding.dispersion is None
    assert empty.tnf.distance_median is None
    assert empty.coverage.log_mad is None
    assert empty.scg.unique_marker_count == 0
    assert metrics.embedding_distance is None
    assert metrics.tnf_distance is None
    assert metrics.coverage_distance is None


def test_invalid_feature_shapes_and_unknown_markers_are_rejected() -> None:
    index, _state, _inputs, _profiles = _fixture()

    with pytest.raises(EvidenceProfileError, match="Embedding matrix"):
        EvidenceInputs.from_arrays(
            contact_index=index,
            embedding=np.zeros((2, 2)),
        )
    with pytest.raises(EvidenceProfileError, match="non-negative"):
        EvidenceInputs.from_arrays(
            contact_index=index,
            coverage=np.asarray([1.0, -1.0, 1.0, 1.0, 1.0]),
        )
    with pytest.raises(EvidenceProfileError, match="Unknown SCG marker"):
        EvidenceInputs.from_arrays(
            contact_index=index,
            scg_markers_by_contig={"a": ("not_a_marker",)},
        )
    valid = EvidenceInputs.from_arrays(contact_index=index)
    with pytest.raises(EvidenceProfileError, match="fixed embedded"):
        EvidenceInputs(
            embedding=valid.embedding,
            embedding_present=valid.embedding_present,
            tnf=valid.tnf,
            tnf_present=valid.tnf_present,
            coverage_log1p=valid.coverage_log1p,
            scg_markers=valid.scg_markers,
            scg_orf_counts=valid.scg_orf_counts,
            marker_ids=DEFAULT_SCG_MARKERS[:-1],
        )


def test_load_evidence_inputs_aligns_public_files(tmp_path: Path) -> None:
    index, _state, _inputs, _profiles = _fixture()
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(
        "".join(
            f">{name}\n{'ACGT' * (20 + idx)}\n"
            for idx, name in enumerate(index.contig_names)
        ),
        encoding="utf-8",
    )
    coverage = tmp_path / "coverage.tsv"
    coverage.write_text(
        "contig_name\tcoverage\n"
        "a\t9\nb\t11\nc\t99\nd\t101\n",
        encoding="utf-8",
    )
    embedding = tmp_path / "embedding.tsv"
    embedding.write_text(
        "contig_id\tz0\tz1\tsupport_weight\n"
        "c\t1\t1\t2\n"
        "a\t2\t0\t2\n"
        "d\t2\t2\t2\n"
        "b\t0\t3\t2\n",
        encoding="utf-8",
    )
    scg = tmp_path / "contig_scg_index.json"
    scg.write_text(
        """{
          "profiles": {
            "a": {
              "marker_ids": ["GrpE"],
              "marker_orf_counts": {"GrpE": 2}
            }
          }
        }""",
        encoding="utf-8",
    )

    loaded = load_evidence_inputs(
        contact_index=index,
        embedding_tsv=embedding,
        contigs_fasta=fasta,
        coverage_tsv=coverage,
        scg_index_json=scg,
    )

    assert loaded.embedding is not None
    assert loaded.embedding[index.contig_index("a")] == pytest.approx([1.0, 0.0])
    assert not loaded.embedding_present[index.contig_index("e")]
    assert loaded.tnf is not None
    assert loaded.tnf.shape == (5, 136)
    assert loaded.coverage_log1p[index.contig_index("a")] == pytest.approx(
        math.log1p(9.0)
    )
    assert math.isnan(loaded.coverage_log1p[index.contig_index("e")])
    assert loaded.scg_markers[index.contig_index("a")] == (
        DEFAULT_SCG_MARKERS.index("GrpE"),
    )


def test_load_evidence_inputs_rejects_missing_explicit_scg_index(
    tmp_path: Path,
) -> None:
    index, _state, _inputs, _profiles = _fixture()
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(
        "".join(f">{name}\nACGTACGT\n" for name in index.contig_names),
        encoding="utf-8",
    )
    coverage = tmp_path / "coverage.tsv"
    coverage.write_text(
        "contig_name\tcoverage\n"
        + "".join(f"{name}\t1\n" for name in index.contig_names),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="SCG contig index"):
        load_evidence_inputs(
            contact_index=index,
            embedding_tsv=None,
            contigs_fasta=fasta,
            coverage_tsv=coverage,
            scg_index_json=tmp_path / "missing.json",
        )


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        1.0
        - np.dot(left, right)
        / (np.linalg.norm(left) * np.linalg.norm(right))
    )
