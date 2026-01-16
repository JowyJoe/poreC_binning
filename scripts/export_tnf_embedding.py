"""Export TNF embeddings (PCA/UMAP) for downstream visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None

from hypergraph_binning.multiplex.features import compute_tnf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute TNF PCA/UMAP coordinates for contigs to aid visualization."
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to contigs FASTA (same as used for binning).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2000,
        help="Minimum contig length to include (bp). Default: 2000",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tnf_embedding.tsv"),
        help="Output TSV path containing PCA/UMAP coordinates.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("tnf_embedding_meta.json"),
        help="JSON file to store explained variance and run metadata.",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Enable UMAP embedding (requires umap-learn).",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    return parser.parse_args()


def compute_gc_lengths(
    fasta_path: str,
    keep: set[str],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    lengths: Dict[str, int] = {}
    gc: Dict[str, float] = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        if rec.id not in keep:
            continue
        seq = str(rec.seq).upper()
        length = len(seq)
        lengths[rec.id] = length
        if length == 0:
            gc[rec.id] = 0.0
        else:
            g = seq.count("G")
            c = seq.count("C")
            gc[rec.id] = float(g + c) / float(length)
    return lengths, gc


def run_pca(features: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, List[float]]:
    scaler = StandardScaler()
    feats = scaler.fit_transform(features)
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(feats)
    explained = pca.explained_variance_ratio_.tolist()
    return coords, explained


def run_umap(
    features: np.ndarray,
    n_neighbors: int,
    min_dist: float,
) -> Optional[np.ndarray]:
    if umap is None:
        return None
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=0,
        metric="euclidean",
    )
    return reducer.fit_transform(features)


def main() -> None:
    args = parse_args()
    print(f"[TNF] computing TNF vectors from {args.fasta} (min_len={args.min_length})...")
    names, tnf_features = compute_tnf(args.fasta, min_length=args.min_length)
    if not names:
        raise SystemExit("No contigs passed the length filter; aborting.")

    name_set = set(names)
    lengths, gc = compute_gc_lengths(args.fasta, name_set)

    print("[TNF] running PCA...")
    pca_coords, explained = run_pca(tnf_features)

    umap_coords = None
    if args.umap:
        if umap is None:
            print("[TNF] umap-learn not available, skipping UMAP embedding.")
        else:
            print("[TNF] running UMAP...")
            umap_coords = run_umap(
                tnf_features,
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
            )

    df = pd.DataFrame(
        {
            "contig": names,
            "length": [lengths.get(n, 0) for n in names],
            "gc": [gc.get(n, 0.0) for n in names],
            "pca1": pca_coords[:, 0],
            "pca2": pca_coords[:, 1],
        }
    )

    if umap_coords is not None:
        df["umap1"] = umap_coords[:, 0]
        df["umap2"] = umap_coords[:, 1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"[TNF] wrote embedding table to {args.output}")

    meta = {
        "fasta": str(args.fasta),
        "min_length": args.min_length,
        "num_contigs": len(names),
        "pca_explained_variance": explained,
        "umap_enabled": args.umap and umap is not None,
        "umap_params": {
            "n_neighbors": args.umap_n_neighbors,
            "min_dist": args.umap_min_dist,
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metadata_output, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[TNF] wrote metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
