from __future__ import annotations

from pathlib import Path

import typer

from porebin import __version__
from porebin.bam_contacts import BamContactsError, bam_to_contacts_parquet
from porebin.build_graph import GraphBuildError, build_graph
from porebin.cluster import (
    GraphClusterError,
    cluster_leiden_pairwise,
    cluster_spectral_hypergraph,
)
from porebin.export import ExportError, MIN_BIN_BP, export_bins
from porebin.pairwise_baseline import PairwiseBaselineError, build_pairwise_edges_from_bam
from porebin.refine import RefineError, refine_bins_parquet
from porebin.utils import (
    console,
    ensure_dir,
    iter_fasta_names,
    record_run,
    setup_logging,
    write_json,
)

app = typer.Typer(add_completion=False, help="porebin: bin metagenomes from Pore-C multi-way contacts.")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    setup_logging(verbose=verbose)


@app.command("bam2contacts")
def bam2contacts(
    bam: Path = typer.Option(..., "--bam", help="Name-sorted BAM (samtools sort -n)."),
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    parquet_batch_size: int = typer.Option(10_000, "--parquet-batch-size", help="Parquet batch size."),
) -> None:
    """
    Build our internal hyperedge format (contacts.parquet) directly from a name-sorted BAM.

    This is the recommended entrypoint for BAM-based pipelines.
    """
    out = out.resolve()
    params = {
        "bam": str(bam),
        "contigs": str(contigs),
        "out": str(out),
        "parquet_batch_size": parquet_batch_size,
    }
    with record_run(out, command="bam2contacts", params=params, seed=None) as run_record:
        try:
            meta = bam_to_contacts_parquet(
                bam=bam,
                contigs_fasta=contigs,
                out_dir=out,
                parquet_batch_size=parquet_batch_size,
                logger=None,
            )
            run_record["outputs"] = {
                "contacts_parquet": str(meta.get("contacts_parquet")),
                "coverage_tsv": str(meta.get("coverage_tsv")),
                "qc_json": str(meta.get("qc_json")),
            }
            run_record["stats"] = meta.get("stats")
        except (BamContactsError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def build(
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    contacts: Path = typer.Option(..., "--contacts", help="Contacts Parquet file (from bam2contacts)."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    order_norm_method: str = typer.Option(
        "pair", "--order-norm", help="OrderNorm: pair (2/(k*(k-1))) or star (1/(k-1))."
    ),
    parquet_batch_size: int = typer.Option(100_000, "--parquet-batch-size", help="Parquet batch size."),
) -> None:
    out = out.resolve()
    params = {
        "contigs": str(contigs),
        "contacts": str(contacts),
        "out": str(out),
        "order_norm_method": order_norm_method,
        "parquet_batch_size": parquet_batch_size,
    }
    with record_run(out, command="build", params=params, seed=None):
        try:
            build_graph(
                contigs_fasta=contigs,
                contacts_parquet=contacts,
                out_dir=out,
                order_norm_method=order_norm_method,
                parquet_batch_size=parquet_batch_size,
            )
        except (GraphBuildError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def cluster(
    graph: Path = typer.Option(..., "--graph", help="Graph directory (out_dir/graph)."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    method: str = typer.Option(
        "spectral",
        "--method",
        help="Clustering method: spectral (joint hypergraph spectral embedding + HDBSCAN).",
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    threads: int = typer.Option(1, "--threads", help="Threads hint for coarse clustering."),
    bam: Path | None = typer.Option(
        None,
        "--bam",
        help="Optional (kept for backwards compatibility; currently unused by spectral v2).",
    ),
) -> None:
    out = out.resolve()
    params = {
        "graph": str(graph),
        "out": str(out),
        "method": method,
        "seed": seed,
        "threads": threads,
        "bam": str(bam) if bam is not None else None,
    }
    with record_run(out, command="cluster", params=params, seed=seed) as run_record:
        try:
            m = method.strip().lower()
            if m == "spectral":
                meta = cluster_spectral_hypergraph(
                    graph_dir=graph,
                    out_bins_tsv=out / "bins.tsv",
                    seed=seed,
                    bam=bam,
                    threads=threads,
                )
                run_record["decisions"] = {"cluster_method": "spectral", **meta}
            else:
                _die(f"Unknown --method {method!r}. Use 'spectral'.")
        except (GraphClusterError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def export(
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    bins_tsv: Path = typer.Option(..., "--bins-tsv", help="Bins TSV (e.g. coarse/bins.tsv or refined/bins.refined.tsv)."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
) -> None:
    out = out.resolve()
    params = {"contigs": str(contigs), "bins_tsv": str(bins_tsv), "out": str(out)}
    with record_run(out, command="export", params=params, seed=None) as run_record:
        try:
            stats = export_bins(contigs_fasta=contigs, bins_tsv=bins_tsv, out_dir=out)
            run_record["thresholds"] = {"MIN_BIN_BP": MIN_BIN_BP}
            run_record["decisions"] = {
                "export_policy": "keep_bins = total_bp >= MIN_BIN_BP; others -> unbinned.fasta",
                "min_bin_bp": MIN_BIN_BP,
                "min_contig_len": stats.min_contig_len,
                "min_contig_len_ratio_1000_2500_bp": stats.min_contig_len_ratio_1000_2500_bp,
            }
            run_record["stats"] = {
                "contigs_total": stats.contigs_total,
                "bins_total": stats.bins_total,
                "bins_kept": stats.bins_kept,
                "contigs_exported": stats.contigs_exported,
                "contigs_unbinned": stats.contigs_unbinned,
            }
        except (ExportError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def refine(
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    bins_tsv: Path = typer.Option(..., "--bins-tsv", help="Coarse bins TSV (e.g. coarse/bins.tsv)."),
    contacts: Path = typer.Option(..., "--contacts", help="contacts.parquet produced by bam2contacts."),
    coverage_tsv: Path | None = typer.Option(
        None, "--coverage-tsv", help="Optional coverage TSV (contig_name\\tcoverage)."
    ),
    out: Path = typer.Option(..., "--out", help="Output directory (refined/)."),
) -> None:
    out = out.resolve()
    params = {
        "contigs": str(contigs),
        "bins_tsv": str(bins_tsv),
        "contacts": str(contacts),
        "coverage_tsv": str(coverage_tsv) if coverage_tsv is not None else None,
        "out": str(out),
    }
    # refine is a host-assignment inference layer on top of coarse candidate host communities.
    # It writes out/run_refine.json plus:
    #   - bins.refined.tsv (core-like contigs only)
    #   - contig_host_scores.tsv (all contigs)
    #   - accessory_associations.tsv (accessory/MGE-like association head)
    with record_run(out, command="refine", params=params, seed=None):
        try:
            refine_bins_parquet(
                contigs_fasta=contigs,
                contacts_parquet=contacts,
                bins_tsv=bins_tsv,
                coverage_tsv=coverage_tsv,
                out_dir=out,
            )
        except (RefineError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command("run-bam")
def run_bam(
    bam: Path = typer.Option(..., "--bam", help="Name-sorted BAM (samtools sort -n)."),
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    threads: int = typer.Option(1, "--threads", help="Threads hint (currently mostly single-threaded)."),
    pairwise_baseline: bool = typer.Option(
        False,
        "--pairwise-baseline",
        help="Run pairwise clique-expansion baseline from BAM (contig-contig graph) instead of hypergraph pipeline.",
    ),
    pairwise_sort_memory: str | None = typer.Option(
        None,
        "--pairwise-sort-memory",
        help="Pairwise baseline only: GNU sort memory for -S (e.g. 8G or 50%).",
    ),
    pairwise_chunk_lines: int = typer.Option(
        5_000_000,
        "--pairwise-chunk-lines",
        help="Pairwise baseline only: raw pair lines per part before rotating output.",
    ),
    order_norm_method: str = typer.Option(
        "pair", "--order-norm", help="OrderNorm: pair (2/(k*(k-1))) or star (1/(k-1))."
    ),
    contacts_parquet_batch_size: int = typer.Option(
        10_000, "--contacts-parquet-batch-size", help="Parquet batch size for bam2contacts output."
    ),
    build_parquet_batch_size: int = typer.Option(100_000, "--build-parquet-batch-size", help="Parquet batch size for build."),
    coarse_method: str = typer.Option(
        "spectral",
        "--coarse-method",
        help="Coarse clustering method: spectral (joint hypergraph spectral embedding + HDBSCAN).",
    ),
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden resolution parameter (pairwise baseline only)."),
    refine: bool = typer.Option(
        True,
        "--refine/--no-refine",
        help="Run refine after coarse binning (recommended for low contamination).",
    ),
) -> None:
    """
    End-to-end BAM pipeline:
      - default (hypergraph): bam2contacts -> build -> coarse cluster -> (optional) refine
      - baseline (--pairwise-baseline): BAM -> pairwise graph -> Leiden
    """
    out = out.resolve()
    params = {
        "bam": str(bam),
        "contigs": str(contigs),
        "out": str(out),
        "seed": seed,
        "threads": threads,
        "pairwise_baseline": pairwise_baseline,
        "pairwise_sort_memory": pairwise_sort_memory,
        "pairwise_chunk_lines": int(pairwise_chunk_lines),
        "order_norm_method": order_norm_method,
        "contacts_parquet_batch_size": contacts_parquet_batch_size,
        "build_parquet_batch_size": build_parquet_batch_size,
        "coarse_method": coarse_method,
        "resolution": resolution,
        "refine": refine,
    }
    with record_run(out, command="run-bam", params=params, seed=seed) as run_record:
        try:
            if pairwise_baseline:
                if refine:
                    _die("--pairwise-baseline does not support --refine (baseline is intended as a coarse control).")

                graph_dir = out / "graph"
                ensure_dir(graph_dir)

                contig_index_tsv = graph_dir / "contig_index.tsv"
                contig_names: list[str] = []
                contig_to_idx: dict[str, int] = {}
                for name in iter_fasta_names(contigs):
                    if not name:
                        continue
                    if name in contig_to_idx:
                        raise GraphBuildError(f"Duplicate contig name in FASTA: {name}")
                    contig_to_idx[name] = len(contig_names)
                    contig_names.append(name)
                if not contig_names:
                    raise GraphBuildError(f"No contigs found in FASTA: {contigs}")
                with contig_index_tsv.open("w", encoding="utf-8", newline="") as fh:
                    fh.write("contig_name\tcontig_idx\n")
                    for idx, name in enumerate(contig_names):
                        fh.write(f"{name}\t{idx}\n")

                pairwise_edges = graph_dir / "pairwise_edges.tsv.gz"
                stats = build_pairwise_edges_from_bam(
                    bam=bam,
                    contigs_fasta=contigs,
                    out_edges_path=pairwise_edges,
                    tmp_dir=out / "tmp" / "pairwise_baseline",
                    sort_threads=max(1, int(threads)),
                    memory=pairwise_sort_memory,
                    chunk_lines=int(pairwise_chunk_lines),
                    logger=None,
                )

                graph_meta = {
                    "porebin_version": __version__,
                    "pairwise_baseline": True,
                    "weight_formula": "2/(k*(k-1))  # == 1/C(k,2)",
                    "input_bam": str(bam),
                    "input_contigs_fasta": str(contigs),
                    "input_sorted_by_qname": stats.input_sorted_by_qname,
                    "input_sorted_by_qname_verified": stats.input_sorted_by_qname_verified,
                    "num_contigs": len(contig_names),
                    "num_edges": stats.unique_edges,
                    "alignments_total": stats.alignments_total,
                    "alignments_kept": stats.alignments_kept,
                    "reads_total": stats.reads_total,
                    "reads_skipped_k_lt_2": stats.reads_skipped_k_lt_2,
                    "raw_pairs_written": stats.raw_pairs_written,
                    "pairwise_edges_path": str(pairwise_edges),
                    "resolution": resolution,
                    "seed": seed,
                    "sort_threads": max(1, int(threads)),
                    "sort_memory": pairwise_sort_memory,
                }
                write_json(graph_dir / "graph_meta.json", graph_meta)

                cluster_leiden_pairwise(
                    contig_index_tsv=contig_index_tsv,
                    pairwise_edges_tsv_gz=pairwise_edges,
                    out_bins_tsv=out / "bins.tsv",
                    resolution=resolution,
                    seed=seed,
                )

                run_record["decisions"] = {"mode": "pairwise_baseline_bam", "cluster_method": "leiden_pairwise", "resolution": resolution}
                run_record["outputs"] = {
                    "graph_dir": str(graph_dir),
                    "pairwise_edges_tsv_gz": str(pairwise_edges),
                    "coarse_bins_tsv": str(out / "bins.tsv"),
                }
                return

            cmeta = bam_to_contacts_parquet(
                bam=bam,
                contigs_fasta=contigs,
                out_dir=out,
                parquet_batch_size=contacts_parquet_batch_size,
                logger=None,
            )
            contacts_parquet = Path(cmeta["contacts_parquet"])
            coverage_tsv = Path(cmeta["coverage_tsv"])

            build_graph(
                contigs_fasta=contigs,
                contacts_parquet=contacts_parquet,
                out_dir=out,
                order_norm_method=order_norm_method,
                parquet_batch_size=build_parquet_batch_size,
            )

            m = coarse_method.strip().lower()
            if m == "spectral":
                coarse_meta = cluster_spectral_hypergraph(
                    graph_dir=out / "graph",
                    out_bins_tsv=out / "bins.tsv",
                    seed=seed,
                    bam=None,
                    threads=threads,
                )
            else:
                raise GraphClusterError(f"Unknown --coarse-method {coarse_method!r}. Use 'spectral'.")

            refined_bins = None
            if refine:
                refined_dir = out / "refined"
                refined_bins = refine_bins_parquet(
                    contigs_fasta=contigs,
                    contacts_parquet=contacts_parquet,
                    bins_tsv=out / "bins.tsv",
                    coverage_tsv=coverage_tsv if coverage_tsv.exists() else None,
                    out_dir=refined_dir,
                )

            run_record["decisions"] = {
                "contacts_source": "bam2contacts",
                "coarse": coarse_meta,
                "refine": bool(refine),
            }
            run_record["outputs"] = {
                "contacts_parquet": str(contacts_parquet),
                "coverage_tsv": str(coverage_tsv),
                "graph_dir": str(out / "graph"),
                "coarse_bins_tsv": str(out / "bins.tsv"),
                "refined_bins_tsv": (str(refined_bins) if refined_bins is not None else None),
            }
        except (BamContactsError, PairwiseBaselineError, GraphBuildError, GraphClusterError, RefineError, FileNotFoundError) as exc:
            _die(str(exc))


def _die(message: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code)


if __name__ == "__main__":  # pragma: no cover
    app()
