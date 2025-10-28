from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from ..utils.contigs import read_contigs_fasta
from ..io.bam import iterate_porec_hyperedges, PoreCFilter
from ..hypergraph.build import build_from_porec
from ..hypergraph.laplacian import make_L_linear_operator
from ..hypergraph.stream import write_edge_chunks
from ..hypergraph.stream_operator import StreamingHypergraphOperator
from ..spectral.eigen import spectral_embedding
from ..spectral.kselect import auto_select_k, KSelectParams
from ..clustering.kmeans_cluster import kmeans_labels


@dataclass
class Config:
    contigs_fasta: str
    porec_bam: str
    output_dir: str
    k: int
    mapq_min: int = 20
    segment_min_bases: int = 500
    min_segments_per_read: int = 3
    read_coverage_min: float = 0.6
    max_hyperedge_size: int = 20
    q_cap: float = 0.95
    maxiter: int = 300
    seed: int = 42
    # streaming
    streaming: bool = True
    edges_per_chunk: int = 200_000


def load_config(config_path: Path, override_k: Optional[int] = None) -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    k = override_k if override_k is not None else int(cfg["spectral"].get("k", 50))
    return Config(
        contigs_fasta=cfg["inputs"]["contigs_fasta"],
        porec_bam=cfg["inputs"]["porec_bam"],
        output_dir=cfg["outputs"]["output_dir"],
        k=k,
        mapq_min=int(cfg["filters"].get("mapq_min", 20)),
        segment_min_bases=int(cfg["filters"].get("segment_min_bases", 500)),
        min_segments_per_read=int(cfg["filters"].get("min_segments_per_read", 3)),
        read_coverage_min=float(cfg["filters"].get("read_coverage_min", 0.6)),
        max_hyperedge_size=int(cfg["filters"].get("max_hyperedge_size", 20)),
        q_cap=float(cfg.get("weights", {}).get("q_cap", 0.95)),
        maxiter=int(cfg["spectral"].get("maxiter", 300)),
        seed=int(cfg["spectral"].get("seed", 42)),
        streaming=bool(cfg.get("streaming", {}).get("enabled", True)),
        edges_per_chunk=int(cfg.get("streaming", {}).get("edges_per_chunk", 200_000)),
    )


def run_pipeline(config_path: Path, override_k: Optional[int] = None) -> None:
    cfg = load_config(config_path, override_k)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read contigs
    names, lengths = read_contigs_fasta(cfg.contigs_fasta)
    name_to_idx = {n: i for i, n in enumerate(names)}

    # 2) Iterate BAM and build hyperedges (streaming preferred for large data)
    flt = PoreCFilter(
        mapq_min=cfg.mapq_min,
        segment_min_bases=cfg.segment_min_bases,
        min_segments_per_read=cfg.min_segments_per_read,
        read_coverage_min=cfg.read_coverage_min,
        max_hyperedge_size=cfg.max_hyperedge_size,
    )

    contig_set = set(names)
    # streaming generator of hyperedges as (member_indices, q)
    def edge_iter() -> Iterable[Tuple[List[int], float]]:
        for members_names, q in iterate_porec_hyperedges(cfg.porec_bam, contig_set, flt):
            members_idx = [name_to_idx[n] for n in members_names]
            yield members_idx, min(q, cfg.q_cap)

    if cfg.streaming:
        edges_dir = out_dir / "edges_chunks"
        edges = write_edge_chunks(
            n_vertices=len(names),
            edge_iter=tqdm(edge_iter(), desc="Pore-C reads -> chunks"),
            out_dir=edges_dir,
            q_cap=cfg.q_cap,
            chunk_size=cfg.edges_per_chunk,
        )
        if len(edges.chunk_paths) == 0:
            raise RuntimeError("No qualified Pore-C hyperedges generated. Please relax filters or check BAM.")
        stats = {
            "n_vertices": int(edges.n_vertices),
            "n_chunks": int(len(edges.chunk_paths)),
            "mean_degree_v": float(edges.dv.mean() if edges.dv.size else 0.0),
        }
        (out_dir / "hypergraph.stats.json").write_text(pd.Series(stats).to_json(indent=2), encoding="utf-8")
        Lop = StreamingHypergraphOperator(n=len(names), dv=edges.dv, chunk_paths=[str(p) for p in edges.chunk_paths])
    else:
        # non-streaming path (may be heavy on memory)
        hg = build_from_porec(names, tqdm(edge_iter(), desc="Pore-C reads -> hyperedges"))
        stats = {
            "n_vertices": int(hg.H.shape[0]),
            "n_hyperedges": int(hg.H.shape[1]),
            "sum_w": float(hg.w.sum()),
            "mean_degree_v": float(hg.dv.mean() if hg.dv.size else 0.0),
        }
        (out_dir / "hypergraph.stats.json").write_text(pd.Series(stats).to_json(indent=2), encoding="utf-8")
        if hg.H.shape[1] == 0:
            raise RuntimeError("No qualified Pore-C hyperedges generated. Please relax filters or check BAM.")
        Lop = make_L_linear_operator(hg.H, hg.w, hg.de, hg.dv)

    # 4) Spectral embedding
    # 4) Spectral embedding or auto-k selection
    if cfg.k <= 0:
        ks_params = KSelectParams(k_min=10, k_max=200, maxiter=cfg.maxiter)
        k_sel, U = auto_select_k(Lop, ks_params)
        k_used = k_sel
    else:
        U = spectral_embedding(Lop, k=cfg.k, maxiter=cfg.maxiter, seed=cfg.seed)
        k_used = cfg.k

    # 5) K-means clustering on embedding
    labels = kmeans_labels(U, k=k_used, n_init=20, seed=cfg.seed)

    # 6) Save bins.tsv
    bins_path = out_dir / "bins.tsv"
    with open(bins_path, "w", encoding="utf-8") as f:
        f.write("contig\tbin\n")
        for name, lab in zip(names, labels):
            f.write(f"{name}\tbin_{int(lab)}\n")

    # 7) Minimal summary
    summary = (
        pd.Series(labels)
        .value_counts()
        .rename_axis("bin")
        .reset_index(name="num_contigs")
        .sort_values("num_contigs", ascending=False)
    )
    summary.to_csv(out_dir / "bin_sizes.tsv", sep="\t", index=False)

    # record k used
    (out_dir / "k_used.txt").write_text(str(k_used), encoding="utf-8")

    print("Pipeline completed. Outputs written to", out_dir)


def self_test_pipeline() -> None:
    """Build a tiny synthetic hypergraph (no BAM) and run spectral+kmeans to verify implementation."""
    # 6 contigs, 3 hyperedges connecting {0,1,2}, {3,4,5}, and a weak cross {2,3}
    names = [f"c{i}" for i in range(6)]
    edges = [
        ([0, 1, 2], 0.9),
        ([3, 4, 5], 0.9),
        ([2, 3], 0.2),
    ]
    hg = build_from_porec(names, edges)
    Lop = make_L_linear_operator(hg.H, hg.w, hg.de, hg.dv)
    U = spectral_embedding(Lop, k=2, maxiter=200)
    labels = kmeans_labels(U, k=2, n_init=10)
    df = pd.DataFrame({"contig": names, "bin": labels})
    print("Self-test labels:\n", df.sort_values("contig").to_string(index=False))
