"""CLI command for the genome-centric binning mainline."""

from __future__ import annotations

from pathlib import Path

import typer

from porebin_genome.coarse.orchestrate import run_coarse_discovery
from porebin_genome.io.runtime import record_run
from porebin_genome.refine.orchestrate import run_refinement


def bin_command(
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    contacts: Path = typer.Option(..., "--contacts", help="Canonical contacts.parquet evidence file."),
    coverage_tsv: Path = typer.Option(..., "--coverage-tsv", help="Coverage table used in the default genome-binning contract."),
    disable_scg: bool = typer.Option(
        False,
        "--disable-scg",
        help="Disable internal SCG veto. By default refine requires external `prodigal` and `hmmsearch`.",
    ),
    scg_hmm: Path | None = typer.Option(
        None,
        "--scg-hmm",
        help="Optional override for the bundled bacterial core SCG HMM database.",
    ),
    knn_k: str = typer.Option(
        "adaptive",
        "--knn-k",
        help="Feature graph neighborhood size: 'adaptive' by default, or a positive integer for fixed k.",
    ),
    out: Path = typer.Option(..., "--out", help="Output directory."),
) -> None:
    """Run coarse candidate-bin discovery followed by the genome-centric refine MVP."""
    out = out.resolve()
    params = {
        "contigs": str(contigs),
        "contacts": str(contacts),
        "coverage_tsv": str(coverage_tsv),
        "disable_scg": bool(disable_scg),
        "scg_hmm": (str(scg_hmm) if scg_hmm is not None else None),
        "knn_k": str(knn_k),
        "out": str(out),
    }
    with record_run(out, command="bin", params=params) as run_record:
        coarse_result = run_coarse_discovery(
            contigs_fasta=contigs,
            contacts_parquet=contacts,
            coverage_tsv=coverage_tsv,
            out_dir=out,
            feature_knn_k=knn_k,
        )
        refine_result = run_refinement(
            contigs_fasta=contigs,
            coarse_bins_tsv=coarse_result.bins_tsv,
            contacts_parquet=contacts,
            coverage_tsv=coverage_tsv,
            enable_scg=(not disable_scg),
            scg_hmm_path=scg_hmm,
            out_dir=out,
        )
        run_record["outputs"] = {
            "coarse_bins_tsv": str(coarse_result.bins_tsv),
            "coarse_run_json": str(coarse_result.run_json),
            "refined_bins_tsv": str(refine_result.bins_refined_tsv),
            "unbinned_tsv": str(refine_result.unbinned_tsv),
            "bin_qc_tsv": str(refine_result.bin_qc_tsv),
            "refine_actions_tsv": str(refine_result.refine_actions_tsv),
            "refine_meta_json": str(refine_result.refine_meta_json),
        }
        run_record["notes"] = {
            "coarse_semantics": "candidate_genome_bins",
            "refine_semantics": "final_genome_bins_and_unresolved_contigs",
            "coarse_implemented": bool(coarse_result.implemented),
            "refine_implemented": bool(refine_result.implemented),
            "current_validation_scope": "coarse_candidate_genome_bins_plus_refine_mvp",
        }
