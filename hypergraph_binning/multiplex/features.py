from __future__ import annotations
from typing import List, Tuple, Dict
import itertools

import numpy as np
from Bio import SeqIO
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

def _generate_canonical_kmers(k: int = 4) -> List[str]:
    """Generate canonical k-mers (lexicographically first of forward/revcomp)."""
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    canonical = []
    seen = set()
    
    trans = str.maketrans("ACGT", "TGCA")
    
    for kmer in all_kmers:
        rev = kmer.translate(trans)[::-1]
        canon = min(kmer, rev)
        if canon not in seen:
            seen.add(canon)
            canonical.append(canon)
            
    return sorted(canonical)

def compute_tnf(fasta_path: str, min_length: int = 2000) -> Tuple[List[str], np.ndarray]:
    """
    Compute Tetra-Nucleotide Frequency (TNF) vectors for contigs.
    Returns:
        names: List of contig names
        features: (N, 136) normalized feature matrix
    """
    canonical_4mers = _generate_canonical_kmers(4)
    kmer_to_idx = {k: i for i, k in enumerate(canonical_4mers)}
    n_features = len(canonical_4mers) # Should be 136
    
    names = []
    counts_list = []
    
    trans = str.maketrans("ACGT", "TGCA")
    
    for rec in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Computing TNF"):
        if len(rec.seq) < min_length:
            continue
            
        names.append(rec.id)
        seq = str(rec.seq).upper()
        
        # Count k-mers
        counts = np.zeros(n_features, dtype=np.float32)
        
        # We can iterate the sequence. For efficiency in Python, simple loop is okay for now.
        # For very large datasets, might need optimization, but this is standard.
        for i in range(len(seq) - 3):
            kmer = seq[i:i+4]
            if "N" in kmer:
                continue
            
            # Find canonical index
            rev = kmer.translate(trans)[::-1]
            canon = min(kmer, rev)
            
            if canon in kmer_to_idx:
                counts[kmer_to_idx[canon]] += 1
                
        # Normalize (L2)
        norm = np.linalg.norm(counts)
        if norm > 0:
            counts /= norm
        else:
            # Handle zero vector (e.g. all Ns or too short valid region)
            # Keep as zero or uniform? Zero is safer for distance.
            pass
            
        counts_list.append(counts)
        
    if not counts_list:
        return [], np.zeros((0, n_features))
        
    return names, np.array(counts_list, dtype=np.float32)

def build_knn_graph(
    features: np.ndarray,
    k: int = 10,
    weight_scheme: str = "gaussian",
    sigma: float = None,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build chemical layer as a true hypergraph (KNN neighborhoods as hyperedges).

    Each node i's k-nearest neighbors (including itself) form a hyperedge e_i.
    Returns the same structure as physical layer: (H, w, de, dv)

    Parameters
    ----------
    features : (N, D) TNF feature matrix
    k : number of neighbors per hyperedge (including self)
    weight_scheme :
        - "gaussian": h(v,e) = exp(-d^2 / (2*sigma^2))
        - "inverse": h(v,e) = 1 / (1 + d)
        - "binary": h(v,e) = 1
    sigma : Gaussian kernel bandwidth (auto-estimated if None)

    Returns
    -------
    H : (N, N) hypergraph incidence matrix
        H[i, j] = weight of node i in hyperedge j
    w : (N,) hyperedge weights
    de : (N,) hyperedge degrees (sum of node weights in each hyperedge)
    dv : (N,) vertex degrees
    """
    n_samples = features.shape[0]

    # KNN query
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Auto-estimate sigma for Gaussian kernel
    if weight_scheme == "gaussian" and sigma is None:
        sigma = np.median(distances[:, 1:])  # exclude self-distance (0)
        sigma = max(sigma, 1e-6)
        print(f"[Chemical Layer] Auto-estimated sigma={sigma:.6f} for Gaussian kernel")

    # Build incidence matrix H
    # H[i, j] = weight of node i in hyperedge j (centered at node j)
    rows = []
    cols = []
    data = []

    for j in range(n_samples):  # j = hyperedge index (also center node)
        nbr_idx = indices[j]     # k neighbors of j (including j itself)
        nbr_dist = distances[j]

        for idx, dist in zip(nbr_idx, nbr_dist):
            # Compute node weight in this hyperedge
            if weight_scheme == "gaussian":
                h_val = np.exp(-dist**2 / (2 * sigma**2))
            elif weight_scheme == "inverse":
                h_val = 1.0 / (1.0 + dist)
            else:  # binary
                h_val = 1.0

            rows.append(idx)
            cols.append(j)
            data.append(h_val)

    H = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    # Compute hyperedge degree: de[j] = sum_i H[i, j]
    de = np.array(H.sum(axis=0)).ravel()

    # Hyperedge weights: average node weight as edge strength
    w = de / k

    # Vertex degree: dv[i] = sum_j w[j] * H[i, j]
    dv = np.array((H @ w.reshape(-1, 1))).ravel()

    return H, w, de, dv
