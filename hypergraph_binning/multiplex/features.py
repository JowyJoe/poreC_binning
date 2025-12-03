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
    Build KNN graph and return Incidence Matrix H_chem and Weights W_chem.
    
    H_chem: (N, N) sparse matrix. Column j represents the hyperedge centered at node j.
            H[i, j] = 1 if node i is in the neighborhood of j (including j itself).
    W_chem: (N,) weight vector for each hyperedge (column).
    """
    n_samples = features.shape[0]
    
    # Fit KNN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # Build H_chem (Incidence Matrix)
    # Rows: Nodes (Contigs)
    # Cols: Hyperedges (Centered at each Contig)
    # If contig i is neighbor of j, then H[i, j] = 1
    
    # We use lil_matrix for construction
    H = lil_matrix((n_samples, n_samples), dtype=np.float64)
    
    # Weights for each hyperedge (column j)
    # We can define weight of hyperedge j based on the tightness of the neighborhood.
    # Strategy: Average inverse distance of neighbors? Or sum?
    # The plan suggested: w_ij = 1 / (1 + dist(i, j)) for each connection.
    # But standard hypergraph spectral clustering usually assigns ONE weight to the whole hyperedge.
    # However, Zhou et al. allows weighted H (H(v,e) can be weight).
    # But our formula: Delta = I - Dv^-1/2 H W De^-1 H^T Dv^-1/2
    # Usually H is binary (0/1). W is diagonal matrix of hyperedge weights.
    
    # Let's stick to the plan:
    # "H_chem ... 第 j 列代表以 Contig j 为中心的超边"
    # "填入化学权重 (反距离)" -> This implies H itself can be weighted?
    # Or does it mean we calculate a weight for the hyperedge?
    
    # Re-reading plan: "填入化学权重 (反距离)" under "构建 H_chem".
    # If H is weighted, then H[i, j] = weight between i and j.
    # Then the formula uses H as is.
    # Let's assume H contains the pairwise weights w_ij.
    # And W (hyperedge weight) can be 1.0 (identity).
    
    # Wait, if H is weighted, then degree calculations need to account for it.
    # d(v) = sum_e w(e) H(v,e)
    # delta(e) = sum_v H(v,e)
    # This works.
    
    # Let's implement H[i, j] = 1 / (1 + dist(i, j))
    
    # Vectorized construction of H
    rows = []
    cols = []
    data = []
    
    for i in range(n_samples):
        # i is the center (column index of H)
        # indices[i] are the neighbors (row indices of H)
        # distances[i] are the distances
        
        nbr_indices = indices[i]
        nbr_dists = distances[i]
        
        # Self loop is included in KNN (dist=0)
        
        for idx, dist in zip(nbr_indices, nbr_dists):
            w = 1.0 / (1.0 + dist)
            rows.append(idx)
            cols.append(i) # Column i is the hyperedge centered at i
            data.append(w)
            
    H = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    
    # W_chem (hyperedge weights) - can be uniform 1.0 since we put weights in H
    # Or we can use a global confidence for the hyperedge?
    # For now, let's return H and let graph.py handle W.
    
    return H
