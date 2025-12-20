from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import yaml
import pandas as pd
from tqdm import tqdm

from ..utils.contigs import read_contigs_fasta
from ..io.bam import iterate_porec_hyperedges, PoreCFilter, PoreCStats
from ..hypergraph.build import build_from_porec
from ..clustering.kmeans_cluster import kmeans_labels

from .features import compute_tnf, build_knn_graph
from .graph import build_phy_laplacian, build_chem_laplacian, build_supra_laplacian
from .embedding import run_multiplex_embedding

@dataclass
class MultiplexConfig:
    contigs_fasta: str
    porec_bam: str
    output_dir: str
    k: int
    beta: float = 0.5
    knn_k: int = 5
    mapq_min: int = 30
    segment_min_bases: int = 1000
    min_segments_per_read: int = 3
    max_hyperedge_size: int = 20
    min_contig_len: int = 2000
    maxiter: int = 300
    seed: int = 42

def load_multiplex_config(config_path: Path, override_k: Optional[int] = None, override_beta: Optional[float] = None) -> MultiplexConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    k = override_k if override_k is not None else int(cfg["spectral"].get("k", 50))
    beta = override_beta if override_beta is not None else float(cfg.get("multiplex", {}).get("beta", 0.5))
    
    return MultiplexConfig(
        contigs_fasta=cfg["inputs"]["contigs_fasta"],
        porec_bam=cfg["inputs"]["porec_bam"],
        output_dir=cfg["outputs"]["output_dir"],
        k=k,
        beta=beta,
        knn_k=int(cfg.get("multiplex", {}).get("knn_k", 5)),
        mapq_min=int(cfg["filters"].get("mapq_min", 30)),
        segment_min_bases=int(cfg["filters"].get("segment_min_bases", 1000)),
        min_segments_per_read=int(cfg["filters"].get("min_segments_per_read", 3)),
        max_hyperedge_size=int(cfg["filters"].get("max_hyperedge_size", 20)),
        min_contig_len=int(cfg["filters"].get("min_contig_len", 2000)),
        maxiter=int(cfg["spectral"].get("maxiter", 300)),
        seed=int(cfg["spectral"].get("seed", 42)),
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

    # Build KNN
    print("Building Chemical Graph...")
    H_chem = build_knn_graph(tnf_features, k=cfg.knn_k)
    L_chem = build_chem_laplacian(H_chem)
    
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
        f"edges_yielded={porec_stats.edges_yielded:,}"
    )
    
    # 4. Supra-Laplacian
    print("Constructing Supra-Laplacian...")
    L_supra = build_supra_laplacian(L_phy, L_chem, beta=cfg.beta)
    
    # 5. Embedding
    print(f"Running Spectral Embedding (k={cfg.k})...")
    U_norm = run_multiplex_embedding(L_supra, k=cfg.k, maxiter=cfg.maxiter, seed=cfg.seed)
    
    # Determine actual k used (in case of auto-selection)
    actual_k = U_norm.shape[1]
    print(f"Using k={actual_k} for clustering.")

    # 6. Clustering
    print("Clustering...")
    labels = kmeans_labels(U_norm, k=actual_k, seed=cfg.seed)
    
    # 7. Output
    bins_path = out_dir / "bins.tsv"
    print(f"Writing results to {bins_path}")
    with open(bins_path, "w", encoding="utf-8") as f:
        f.write("contig\tbin\n")
        for name, lab in zip(names, labels):
            f.write(f"{name}\tbin_{int(lab)}\n")
            
    # Summary
    summary = (
        pd.Series(labels)
        .value_counts()
        .rename_axis("bin")
        .reset_index(name="num_contigs")
        .sort_values("num_contigs", ascending=False)
    )
    summary.to_csv(out_dir / "bin_sizes.tsv", sep="\t", index=False)
    
    print("Multiplex Pipeline Completed.")
