from __future__ import annotations


def test_porebin_genome_imports_smoke() -> None:
    from porebin_genome import __version__
    from porebin_genome.cli import app
    from porebin_genome.coarse.cluster import hdbscan_cluster, write_coarse_bins_tsv
    from porebin_genome.coarse.contact import ContactIncidence, build_contact_incidence_from_parquet
    from porebin_genome.coarse.embed import auto_embedding_dim, spectral_embed_joint
    from porebin_genome.coarse.features import FeatureMatrix, build_feature_matrix
    from porebin_genome.coarse.operator import (
        build_adaptive_feature_knn_edges,
        build_feature_incidence,
        build_feature_knn_edges,
        make_theta_operator,
    )
    from porebin_genome.coarse.orchestrate import CoarseRunResult, run_coarse_discovery
    from porebin_genome.evidence.bam import BamEvidenceStats, bam_to_contact_evidence
    from porebin_genome.evidence.canonical import CanonicalContact, canonicalize_contact
    from porebin_genome.evidence.preflight import EvidencePreflightReport, run_evidence_preflight
    from porebin_genome.evidence.scg import (
        ContigScgProfile,
        ensure_scg_toolchain_available,
        load_scg_panel,
    )
    from porebin_genome.evidence.tnf import TNF136_LIST, compute_tnf136_features
    from porebin_genome.export.orchestrate import ExportRunResult, export_final_results
    from porebin_genome.io.contracts import (
        BIN_QC_COLUMNS,
        COARSE_BINS_COLUMNS,
        CONTACTS_PARQUET_CORE_FIELDS,
        COVERAGE_TSV_COLUMNS,
        FINAL_BINS_COLUMNS,
        REFINE_ACTIONS_COLUMNS,
        UNBINNED_COLUMNS,
    )
    from porebin_genome.refinement import (
        ActionEvaluator,
        ActionPolicy,
        generate_merge_candidates,
    )
    from porebin_genome.refinement.contact_index import ContactIndex
    from porebin_genome.refinement.orchestrate import (
        RefineRunResult,
        run_refinement,
    )
    from porebin_genome.refinement.profiles import EvidenceProfileState
    from porebin_genome.refinement.state import RefineState

    assert __version__
    assert app is not None
    assert CanonicalContact.__name__ == "CanonicalContact"
    assert callable(canonicalize_contact)
    assert BamEvidenceStats.__name__ == "BamEvidenceStats"
    assert callable(bam_to_contact_evidence)
    assert EvidencePreflightReport.__name__ == "EvidencePreflightReport"
    assert callable(run_evidence_preflight)
    assert len(TNF136_LIST) == 136
    assert callable(compute_tnf136_features)
    assert ContactIncidence.__name__ == "ContactIncidence"
    assert callable(build_contact_incidence_from_parquet)
    assert FeatureMatrix.__name__ == "FeatureMatrix"
    assert callable(build_feature_matrix)
    assert callable(build_feature_knn_edges)
    assert callable(build_adaptive_feature_knn_edges)
    assert callable(build_feature_incidence)
    assert callable(make_theta_operator)
    assert callable(auto_embedding_dim)
    assert callable(spectral_embed_joint)
    assert callable(hdbscan_cluster)
    assert callable(write_coarse_bins_tsv)
    assert callable(run_coarse_discovery)
    assert CoarseRunResult.__name__ == "CoarseRunResult"
    assert callable(run_refinement)
    assert RefineRunResult.__name__ == "RefineRunResult"
    assert ContactIndex.__name__ == "ContactIndex"
    assert RefineState.__name__ == "RefineState"
    assert EvidenceProfileState.__name__ == "EvidenceProfileState"
    assert ActionEvaluator.__name__ == "ActionEvaluator"
    assert ActionPolicy.__name__ == "ActionPolicy"
    assert ContigScgProfile.__name__ == "ContigScgProfile"
    assert callable(ensure_scg_toolchain_available)
    assert callable(load_scg_panel)
    assert callable(generate_merge_candidates)
    assert callable(export_final_results)
    assert ExportRunResult.__name__ == "ExportRunResult"
    assert CONTACTS_PARQUET_CORE_FIELDS[:3] == ("contact_id", "contigs", "contig_weights")
    assert COVERAGE_TSV_COLUMNS == ("contig_name", "coverage")
    assert COARSE_BINS_COLUMNS == ("contig_name", "bin_id")
    assert FINAL_BINS_COLUMNS[:2] == ("contig_id", "bin_id")
    assert UNBINNED_COLUMNS[0] == "contig_id"
    assert BIN_QC_COLUMNS[0] == "bin_id"
    assert REFINE_ACTIONS_COLUMNS[0] == "action_type"
