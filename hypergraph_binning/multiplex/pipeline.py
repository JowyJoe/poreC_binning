from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.contigs import read_contigs_fasta
from ..io.bam import iterate_porec_hyperedges, PoreCFilter, PoreCStats
from ..hypergraph.build import build_from_porec
from ..clustering.kmeans_cluster import kmeans_labels

from .features import compute_tnf, build_knn_graph
from .graph import build_phy_laplacian, build_chem_laplacian, build_supra_laplacian
from .embedding import run_multiplex_embedding
from .auto_tune import auto_select_beta_k
from .confidence import assess_and_filter

@dataclass
class MultiplexConfig:
    contigs_fasta: str
    porec_bam: str
    output_dir: str
    k: int
    beta: float = 0.5
    knn_k: int = 10
    mapq_min: int = 30
    segment_min_bases: int = 1000
    min_segments_per_read: int = 3
    max_hyperedge_size: int = 20
    min_contig_len: int = 2000
    maxiter: int = 300
    seed: int = 42
    # Quality filtering options
    enable_quality_filter: bool = True
    confidence_threshold: float = 0.3

def load_multiplex_config(config_path: Path, override_k: Optional[int] = None, override_beta: Optional[float] = None) -> MultiplexConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    k = override_k if override_k is not None else int(cfg["spectral"].get("k", 50))
    beta = override_beta if override_beta is not None else float(cfg.get("multiplex", {}).get("beta", 0.5))

    # Quality filtering options
    quality_cfg = cfg.get("quality", {})
    enable_quality_filter = quality_cfg.get("enable_filter", True)  # Default on
    confidence_threshold = float(quality_cfg.get("confidence_threshold", 0.3))

    return MultiplexConfig(
        contigs_fasta=cfg["inputs"]["contigs_fasta"],
        porec_bam=cfg["inputs"]["porec_bam"],
        output_dir=cfg["outputs"]["output_dir"],
        k=k,
        beta=beta,
        knn_k=int(cfg.get("multiplex", {}).get("knn_k", 10)),
        mapq_min=int(cfg["filters"].get("mapq_min", 30)),
        segment_min_bases=int(cfg["filters"].get("segment_min_bases", 1000)),
        min_segments_per_read=int(cfg["filters"].get("min_segments_per_read", 3)),
        max_hyperedge_size=int(cfg["filters"].get("max_hyperedge_size", 20)),
        min_contig_len=int(cfg["filters"].get("min_contig_len", 2000)),
        maxiter=int(cfg["spectral"].get("maxiter", 300)),
        seed=int(cfg["spectral"].get("seed", 42)),
        enable_quality_filter=enable_quality_filter,
        confidence_threshold=confidence_threshold,
    )

def run_multiplex_pipeline(config_path: Path, override_k: Optional[int] = None, override_beta: Optional[float] = None) -> None:
    cfg = load_multiplex_config(config_path, override_k, override_beta)
    out_dir = Path(cfg.output_dir) / "multiplex"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting HyperBin-X Multiplex Pipeline")
    print(f"Output directory: {out_dir}")
    print(f"Coupling beta: {cfg.beta}")
    
    # 1. Chemical Layer (TNF + KNN) & Contig Filtering
    # We use compute_tnf to filter contigs and generate features simultaneously.
    # This ensures tnf_names is our "Universe" of valid contigs.
    print(f"Computing TNF and filtering contigs (min_len={cfg.min_contig_len})...")
    
    tnf_names, tnf_features = compute_tnf(cfg.contigs_fasta, min_length=cfg.min_contig_len)
    
    if len(tnf_names) == 0:
        raise ValueError(f"No contigs found longer than {cfg.min_contig_len} bp!")
        
    print(f"Retained {len(tnf_names)} contigs after filtering.")
    
    # Update names/lengths to match filtered set
    # We need lengths for some stats, but read_contigs_fasta gave us all.
    # Let's just use tnf_names as the master list.
    names = tnf_names
    name_to_idx = {n: i for i, n in enumerate(names)}
    n_contigs = len(names)

    # Build KNN hypergraph (true hypergraph structure)
    print("Building Chemical Layer (KNN hypergraph)...")
    H_chem, w_chem, de_chem, dv_chem = build_knn_graph(
        tnf_features,
        k=cfg.knn_k,
        weight_scheme="gaussian"
    )
    L_chem = build_chem_laplacian(H_chem, w_chem, de_chem, dv_chem)
    
    # 2. Physical Layer (Pore-C)
    print("Building Physical Layer (Pore-C)...")
    flt = PoreCFilter(
        mapq_min=cfg.mapq_min,
        segment_min_bases=cfg.segment_min_bases,
        min_segments_per_read=cfg.min_segments_per_read,
        max_hyperedge_size=cfg.max_hyperedge_size,
    )
    
    contig_set = set(names)
    porec_stats = PoreCStats()
    
    def edge_iter():
        for members_names, q in iterate_porec_hyperedges(
            cfg.porec_bam,
            contig_set,
            flt,
            stats=porec_stats,
            log_every=100000,
        ):
            members_idx = [name_to_idx[n] for n in members_names if n in name_to_idx]
            if len(members_idx) >= 2:
                yield members_idx, float(q)
                
    # Build physical hypergraph (in-memory for now, as multiplex usually requires solving eig on 2N)
    # If N is huge, 2N is huge. Streaming not fully supported for multiplex yet in this plan.
    hg = build_from_porec(names, tqdm(edge_iter(), desc="Pore-C reads"))
    L_phy = build_phy_laplacian(hg)

    # Persist Pore-C stats for diagnostics
    stats_path = out_dir / "porec_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(porec_stats.as_dict(), f, indent=2)
    print(
        "Pore-C summary: "
        f"reads_total={porec_stats.reads_total:,}, "
        f"read_pass_contigs={porec_stats.reads_pass_contigs:,}, "
        f"edges_yielded={porec_stats.edges_yielded:,}, "
        f"contig_hits_discarded={porec_stats.contig_hits_discarded:,}, "
        f"reads_filtered_low_quality={porec_stats.reads_filtered_low_quality:,}, "
        f"reads_filtered_small={porec_stats.reads_filtered_small:,}"
    )

    # 4. Auto-tuning or manual parameters
    if cfg.k <= 0:
        # Joint (β, k) auto-selection
        print("Auto-tuning (β, k) using grid search...")
        tune_result = auto_select_beta_k(
            L_phy, L_chem,
            maxiter=cfg.maxiter,
            seed=cfg.seed,
            verbose=True
        )

        best_beta = tune_result.best_beta
        best_k = tune_result.best_k

        # Save auto-tune results
        tune_path = out_dir / "auto_tune_results.json"
        with open(tune_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_beta": best_beta,
                "best_k": best_k,
                "best_score": tune_result.best_score,
                "all_results": [(b, k, s) for b, k, s in tune_result.all_results]
            }, f, indent=2)

        # Use pre-computed eigenvectors from auto-tuning
        U = tune_result.eigenvectors[:, :best_k]
        U_phy = U[:n_contigs, :]
        U_chem = U[n_contigs:, :]
        U_fused = (U_phy + U_chem) / 2.0
        norms = np.linalg.norm(U_fused, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        U_norm = U_fused / norms

        actual_k = best_k
        print(f"Auto-selected: β={best_beta}, k={best_k}")
    else:
        # Use specified β and k
        print("Constructing Supra-Laplacian...")
        L_supra = build_supra_laplacian(L_phy, L_chem, beta=cfg.beta)

        print(f"Running Spectral Embedding (k={cfg.k})...")
        U_norm = run_multiplex_embedding(L_supra, k=cfg.k, maxiter=cfg.maxiter, seed=cfg.seed)

        actual_k = U_norm.shape[1]
        print(f"Using k={actual_k} for clustering.")

    # 5. Clustering
    print("Clustering...")
    labels = kmeans_labels(U_norm, k=actual_k, seed=cfg.seed)

    # 6. Quality Assessment and Filtering
    if cfg.enable_quality_filter:
        print("Running quality assessment...")
        # Build contact matrix from hypergraph for quality assessment
        contact_matrix = hg.to_adjacency()

        quality_result = assess_and_filter(
            X=U_norm,
            labels=labels,
            contact_matrix=contact_matrix,
            tnf_matrix=tnf_features,
            coverage=None,  # Coverage not available in current pipeline
            confidence_threshold=cfg.confidence_threshold,
            verbose=True
        )

        # Use filtered labels
        final_labels = quality_result.filtered_labels

        # Save quality metrics
        quality_path = out_dir / "quality_metrics.json"
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump({
                "n_bins_original": quality_result.n_bins_original,
                "n_bins_filtered": quality_result.n_bins_filtered,
                "n_unassigned": quality_result.n_unassigned,
                "confidence_threshold": cfg.confidence_threshold,
                "bin_qualities": [
                    {
                        "bin_id": q.bin_id,
                        "n_contigs": q.n_contigs,
                        "mean_silhouette": q.mean_silhouette,
                        "mean_contact_ratio": q.mean_contact_ratio,
                        "tnf_consistency": q.tnf_consistency,
                        "pore_c_connectivity": q.pore_c_connectivity,
                        "overall_quality": q.overall_quality
                    }
                    for q in quality_result.bin_qualities
                ]
            }, f, indent=2)
        print(f"Quality metrics saved to {quality_path}")
    else:
        final_labels = labels

    # 7. Output
    bins_path = out_dir / "bins.tsv"
    print(f"Writing results to {bins_path}")
    with open(bins_path, "w", encoding="utf-8") as f:
        f.write("contig\tbin\n")
        for name, lab in zip(names, final_labels):
            if lab < 0:
                f.write(f"{name}\tunassigned\n")
            else:
                f.write(f"{name}\tbin_{int(lab)}\n")

    # Summary (exclude unassigned)
    assigned_mask = final_labels >= 0
    assigned_labels = final_labels[assigned_mask]
    summary = (
        pd.Series(assigned_labels)
        .value_counts()
        .rename_axis("bin")
        .reset_index(name="num_contigs")
        .sort_values("num_contigs", ascending=False)
    )
    summary.to_csv(out_dir / "bin_sizes.tsv", sep="\t", index=False)

    n_assigned = assigned_mask.sum()
    n_unassigned = (~assigned_mask).sum()
    print(f"Assigned: {n_assigned} contigs, Unassigned: {n_unassigned} contigs")
    print("Multiplex Pipeline Completed.")
