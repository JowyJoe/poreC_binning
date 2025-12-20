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

def build_knn_graph(features: np.ndarray, k: int = 5) -> Tuple[csr_matrix, np.ndarray]:
    """
    Build a (mutual) KNN graph and return incidence matrix H_chem.

    H_chem: (N, N) sparse matrix. Column j represents the hyperedge centered at node j.
            H[i, j] > 0 if node i is in the (mutual) KNN neighborhood of j (including j itself).
    """
    n_samples = features.shape[0]

    # Fit KNN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean").fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Pre-compute neighbor sets for mutual kNN filtering:
    # j is considered a valid neighbor of i only if i is also in the KNN list of j.
    neighbor_sets = [set(row) for row in indices]

    # Vectorized construction of H (stored as weighted incidence matrix).
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(n_samples):
        # i is the center (column index of H)
        nbr_indices = indices[i]
        nbr_dists = distances[i]

        for idx, dist in zip(nbr_indices, nbr_dists):
            # Keep only mutual nearest neighbours (mutual kNN)
            if i not in neighbor_sets[idx]:
                continue
            w = 1.0 / (1.0 + dist)
            rows.append(idx)
            cols.append(i)  # Column i is the hyperedge centered at i
            data.append(w)

    H = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    # W_chem (hyperedge weights) can be treated as identity since H already stores pairwise weights.
    return H
