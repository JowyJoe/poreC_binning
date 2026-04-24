"""Genome-centric Pore-C metagenomic binning tool.

This package defines a clean public surface for a genome-centric workflow:

1. construct canonical Pore-C evidence
2. discover candidate genome bins
3. refine candidate bins into final genome bins
4. export final bins and unresolved contigs

The current mainline includes:

- evidence construction from BAM into canonical contact evidence
- coarse candidate genome-bin discovery
- genome-centric refine MVP
- FASTA export for final bins and unresolved contigs
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
