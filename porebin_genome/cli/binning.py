"""CLI command for the genome-centric binning mainline."""

from __future__ import annotations

from pathlib import Path

import typer

from porebin_genome.coarse.hyperedge_embedding import (
    DEFAULT_HYPEREDGE_EMBEDDING_DIM,
    DEFAULT_HYPEREDGE_EMBEDDING_EPOCHS,
    DEFAULT_HYPEREDGE_EMBEDDING_LR,
    DEFAULT_HYPEREDGE_FEATURE_GUARD,
    DEFAULT_HYPEREDGE_VAE_BETA,
    DEFAULT_HYPEREDGE_VAE_BATCH_SIZE,
    DEFAULT_HYPEREDGE_VAE_LAMBDA,
)
from porebin_genome.coarse.orchestrate import DEFAULT_COARSE_METHOD, run_coarse_discovery
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
    coarse_method: str = typer.Option(
        DEFAULT_COARSE_METHOD,
        "--coarse-method",
        help="Coarse discovery method: spectral or hgvae.",
    ),
    contact_weight_mode: str = typer.Option(
        "hypergraph-native",
        "--contact-weight-mode",
        help="Contact hyperedge weighting mode: 'original' or 'hypergraph-native'.",
    ),
    hypergraph_weight_eta: float = typer.Option(
        0.5,
        "--hypergraph-weight-eta",
        help="Eta exponent for hypergraph-native k_eff evidence-budget weighting.",
    ),
    pairwise_baseline: bool = typer.Option(
        False,
        "--pairwise-baseline/--no-pairwise-baseline",
        help="Also run the raw clique-expanded pairwise Leiden baseline for comparison.",
    ),
    pairwise_alpha_min: float = typer.Option(
        0.05,
        "--pairwise-alpha-min",
        help="Minimum contig evidence share retained when clique-expanding Pore-C contacts.",
    ),
    action_scorer_mode: str = typer.Option(
        "features-only",
        "--action-scorer-mode",
        help="Refine action scorer mode: off, features-only, or score.",
    ),
    action_scorer_model: Path | None = typer.Option(
        None,
        "--action-scorer-model",
        help="Action scorer model directory used when --action-scorer-mode=score.",
    ),
    hyperedge_embedding: bool = typer.Option(
        False,
        "--hyperedge-embedding/--no-hyperedge-embedding",
        help="Train feature-anchored hypergraph VAE embeddings from TNF/coverage and Pore-C contacts.",
    ),
    hyperedge_embedding_dim: int = typer.Option(
        DEFAULT_HYPEREDGE_EMBEDDING_DIM,
        "--hyperedge-embedding-dim",
        help="Latent dimension of the hypergraph VAE embedding.",
    ),
    hyperedge_embedding_epochs: int = typer.Option(
        DEFAULT_HYPEREDGE_EMBEDDING_EPOCHS,
        "--hyperedge-embedding-epochs",
        help="Training epochs for the hypergraph VAE embedding.",
    ),
    hyperedge_embedding_lr: float = typer.Option(
        DEFAULT_HYPEREDGE_EMBEDDING_LR,
        "--hyperedge-embedding-lr",
        help="Learning rate for the hypergraph VAE optimizer.",
    ),
    hyperedge_vae_beta: float = typer.Option(
        DEFAULT_HYPEREDGE_VAE_BETA,
        "--hyperedge-vae-beta",
        help="Beta multiplier for the VAE KL regularization term.",
    ),
    hyperedge_vae_lambda: float = typer.Option(
        DEFAULT_HYPEREDGE_VAE_LAMBDA,
        "--hyperedge-vae-lambda",
        help="Lambda multiplier for the Pore-C hyperedge latent regularization term.",
    ),
    hyperedge_vae_batch_size: int = typer.Option(
        DEFAULT_HYPEREDGE_VAE_BATCH_SIZE,
        "--hyperedge-vae-batch-size",
        help="Number of Pore-C hyperedges sampled per HG-VAE epoch; use 0 for all edges.",
    ),
    hyperedge_feature_guard: bool = typer.Option(
        DEFAULT_HYPEREDGE_FEATURE_GUARD,
        "--hyperedge-feature-guard/--no-hyperedge-feature-guard",
        help="Downweight Pore-C hyperedges whose members disagree in TNF/coverage feature space.",
    ),
    embedding_scorer_mode: str = typer.Option(
        "off",
        "--embedding-scorer-mode",
        help="Refine embedding scorer mode: off, report, or veto.",
    ),
    embedding_tsv: Path | None = typer.Option(
        None,
        "--embedding-tsv",
        help="Optional precomputed hyperedge embedding TSV for refine embedding scoring.",
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
        "coarse_method": str(coarse_method),
        "contact_weight_mode": str(contact_weight_mode),
        "hypergraph_weight_eta": float(hypergraph_weight_eta),
        "pairwise_baseline": bool(pairwise_baseline),
        "pairwise_alpha_min": float(pairwise_alpha_min),
        "action_scorer_mode": str(action_scorer_mode),
        "action_scorer_model": (str(action_scorer_model) if action_scorer_model is not None else None),
        "hyperedge_embedding": bool(hyperedge_embedding),
        "hyperedge_embedding_dim": int(hyperedge_embedding_dim),
        "hyperedge_embedding_epochs": int(hyperedge_embedding_epochs),
        "hyperedge_embedding_lr": float(hyperedge_embedding_lr),
        "hyperedge_vae_beta": float(hyperedge_vae_beta),
        "hyperedge_vae_lambda": float(hyperedge_vae_lambda),
        "hyperedge_vae_batch_size": int(hyperedge_vae_batch_size),
        "hyperedge_feature_guard": bool(hyperedge_feature_guard),
        "embedding_scorer_mode": str(embedding_scorer_mode),
        "embedding_tsv": (str(embedding_tsv) if embedding_tsv is not None else None),
        "out": str(out),
    }
    with record_run(out, command="bin", params=params) as run_record:
        coarse_result = run_coarse_discovery(
            contigs_fasta=contigs,
            contacts_parquet=contacts,
            coverage_tsv=coverage_tsv,
            out_dir=out,
            feature_knn_k=knn_k,
            coarse_method=coarse_method,
            contact_weight_mode=contact_weight_mode,
            hypergraph_weight_eta=hypergraph_weight_eta,
            run_pairwise_baseline=pairwise_baseline,
            pairwise_alpha_min=pairwise_alpha_min,
            run_hyperedge_embedding=hyperedge_embedding,
            hyperedge_embedding_dim=hyperedge_embedding_dim,
            hyperedge_embedding_epochs=hyperedge_embedding_epochs,
            hyperedge_embedding_lr=hyperedge_embedding_lr,
            hyperedge_vae_beta=hyperedge_vae_beta,
            hyperedge_vae_lambda=hyperedge_vae_lambda,
            hyperedge_vae_batch_size=hyperedge_vae_batch_size,
            hyperedge_feature_guard=hyperedge_feature_guard,
        )
        resolved_embedding_tsv = embedding_tsv or coarse_result.hyperedge_embedding_tsv
        refine_result = run_refinement(
            contigs_fasta=contigs,
            coarse_bins_tsv=coarse_result.bins_tsv,
            contacts_parquet=contacts,
            coverage_tsv=coverage_tsv,
            enable_scg=(not disable_scg),
            scg_hmm_path=scg_hmm,
            action_scorer_mode=action_scorer_mode,
            action_scorer_model=action_scorer_model,
            embedding_scorer_mode=embedding_scorer_mode,
            embedding_tsv=resolved_embedding_tsv,
            contact_weight_mode=contact_weight_mode,
            hypergraph_weight_eta=hypergraph_weight_eta,
            out_dir=out,
        )
        outputs = {
            "coarse_bins_tsv": str(coarse_result.bins_tsv),
            "coarse_run_json": str(coarse_result.run_json),
            "refined_bins_tsv": str(refine_result.bins_refined_tsv),
            "unbinned_tsv": str(refine_result.unbinned_tsv),
            "bin_qc_tsv": str(refine_result.bin_qc_tsv),
            "refine_actions_tsv": str(refine_result.refine_actions_tsv),
            "refine_meta_json": str(refine_result.refine_meta_json),
        }
        if refine_result.refine_action_features_tsv is not None:
            outputs["refine_action_features_tsv"] = str(refine_result.refine_action_features_tsv)
        if refine_result.refine_action_scores_tsv is not None:
            outputs["refine_action_scores_tsv"] = str(refine_result.refine_action_scores_tsv)
        if coarse_result.hyperedge_embedding_tsv is not None:
            outputs["hyperedge_embedding_tsv"] = str(coarse_result.hyperedge_embedding_tsv)
        if coarse_result.hyperedge_embedding_meta_json is not None:
            outputs["hyperedge_embedding_meta_json"] = str(coarse_result.hyperedge_embedding_meta_json)
        if refine_result.refine_embedding_scores_tsv is not None:
            outputs["refine_embedding_scores_tsv"] = str(refine_result.refine_embedding_scores_tsv)
        run_record["outputs"] = outputs
        run_record["notes"] = {
            "coarse_semantics": "candidate_genome_bins",
            "refine_semantics": "final_genome_bins_and_unresolved_contigs",
            "coarse_implemented": bool(coarse_result.implemented),
            "refine_implemented": bool(refine_result.implemented),
            "current_validation_scope": "coarse_candidate_genome_bins_plus_refine_mvp",
        }
