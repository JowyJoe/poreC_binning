from __future__ import annotations

from pathlib import Path

import typer

from porebin import __version__
from porebin.build_graph import GraphBuildError, build_graph
from porebin.cluster import (
    GraphClusterError,
    cluster_leiden,
    cluster_leiden_pairwise,
    cluster_spectral_hypergraph,
)
from porebin.export import ExportError, MIN_BIN_BP, export_bins
from porebin.normalize import NormalizeError, normalize_contacts
from porebin.pairwise_baseline import PairwiseBaselineError, build_pairwise_edges
from porebin.refine import RefineError, refine_bins
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


@app.command()
def normalize(
    ppl_contacts: Path = typer.Option(..., "--ppl-contacts", help="PPL .contacts TSV file."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    threads: int = typer.Option(1, "--threads", help="Threads hint (v0.1 mostly single-threaded)."),
    include_tags: list[str] = typer.Option(
        ["mapq", "AS", "n_segments"],
        "--include-tags",
        help="Evidence to include: mapq, AS, n_segments, all. Repeatable.",
    ),
    keep_status: list[str] = typer.Option(
        ["passed"],
        "--keep-status",
        help="Keep segments with these status labels (default: passed). Use --keep-status all to keep all.",
    ),
    assume_no_header: bool = typer.Option(
        False, "--assume-no-header", help="Treat first line as data (no header)."
    ),
) -> None:
    out = out.resolve()
    params = {
        "ppl_contacts": str(ppl_contacts),
        "out": str(out),
        "threads": threads,
        "include_tags": include_tags,
        "keep_status": keep_status,
        "assume_no_header": assume_no_header,
    }
    with record_run(out, command="normalize", params=params, seed=None):
        try:
            normalize_contacts(
                ppl_contacts=ppl_contacts,
                out_dir=out,
                include_tags=include_tags,
                keep_status=keep_status,
                assume_no_header=assume_no_header,
                threads=threads,
            )
        except (NormalizeError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def build(
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    contacts: Path = typer.Option(..., "--contacts", help="Normalized contacts Parquet file."),
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
        "leiden",
        "--method",
        help="Clustering method: leiden (default) or spectral (experimental hypergraph Laplacian + kmeans).",
    ),
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden resolution parameter."),
    seed: int = typer.Option(0, "--seed", help="Random seed for Leiden."),
) -> None:
    out = out.resolve()
    params = {"graph": str(graph), "out": str(out), "method": method, "resolution": resolution, "seed": seed}
    with record_run(out, command="cluster", params=params, seed=seed) as run_record:
        try:
            m = method.strip().lower()
            if m == "leiden":
                cluster_leiden(
                    graph_dir=graph,
                    out_bins_tsv=out / "bins.tsv",
                    resolution=resolution,
                    seed=seed,
                )
                run_record["decisions"] = {"cluster_method": "leiden", "resolution": resolution}
            elif m == "spectral":
                meta = cluster_spectral_hypergraph(
                    graph_dir=graph,
                    out_bins_tsv=out / "bins.tsv",
                    seed=seed,
                )
                run_record["decisions"] = {"cluster_method": "spectral", **meta}
            else:
                _die(f"Unknown --method {method!r}. Use 'leiden' or 'spectral'.")
        except (GraphClusterError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def export(
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    bins_tsv: Path = typer.Option(..., "--bins-tsv", help="Bins TSV (e.g. coarse/bins.tsv or refined/bins.refined.tsv)."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    threads: int = typer.Option(1, "--threads", help="Threads hint (currently mostly single-threaded)."),
) -> None:
    out = out.resolve()
    params = {"contigs": str(contigs), "bins_tsv": str(bins_tsv), "out": str(out), "threads": threads}
    with record_run(out, command="export", params=params, seed=None) as run_record:
        try:
            stats = export_bins(contigs_fasta=contigs, bins_tsv=bins_tsv, out_dir=out, threads=threads)
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
    ppl_contacts: Path = typer.Option(..., "--ppl-contacts", help="PPL .contacts TSV file (segment-level)."),
    bins_tsv: Path = typer.Option(..., "--bins-tsv", help="Coarse bins TSV (e.g. coarse/bins.tsv)."),
    bam: Path | None = typer.Option(None, "--bam", help="Optional BAM for coverage-aware refine."),
    out: Path = typer.Option(..., "--out", help="Output directory (refined/)."),
    threads: int = typer.Option(1, "--threads", help="Threads hint (currently mostly single-threaded)."),
    seed: int = typer.Option(0, "--seed", help="Random seed (used for split partition if enabled)."),
) -> None:
    out = out.resolve()
    params = {
        "contigs": str(contigs),
        "ppl_contacts": str(ppl_contacts),
        "bins_tsv": str(bins_tsv),
        "bam": str(bam) if bam is not None else None,
        "out": str(out),
        "threads": threads,
        "seed": seed,
    }
    # refine writes out/run_refine.json with all thresholds and decisions.
    with record_run(out, command="refine", params=params, seed=seed):
        try:
            refine_bins(
                contigs_fasta=contigs,
                ppl_contacts=ppl_contacts,
                bins_tsv=bins_tsv,
                bam=bam,
                out_dir=out,
                threads=threads,
                seed=seed,
            )
        except (RefineError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def run(
    ppl_contacts: Path = typer.Option(..., "--ppl-contacts", help="PPL .contacts TSV file."),
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    pairwise_baseline: bool = typer.Option(
        False,
        "--pairwise-baseline",
        help="Run pairwise clique-expansion baseline (contig-contig graph) instead of bipartite hypergraph.",
    ),
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden resolution parameter."),
    seed: int = typer.Option(0, "--seed", help="Random seed for Leiden."),
    threads: int = typer.Option(1, "--threads", help="Threads hint (v0.1 mostly single-threaded)."),
    include_tags: list[str] = typer.Option(
        ["mapq", "AS", "n_segments"],
        "--include-tags",
        help="Evidence to include: mapq, AS, n_segments, all. Repeatable.",
    ),
    keep_status: list[str] = typer.Option(
        ["passed"],
        "--keep-status",
        help="Keep segments with these status labels (default: passed). Use --keep-status all to keep all.",
    ),
    assume_sorted_by_readid: bool = typer.Option(
        False,
        "--assume-sorted-by-readid",
        help="Pairwise baseline only: skip sortedness check and assume input is grouped by readID (column 4).",
    ),
    pairwise_sort_memory: str | None = typer.Option(
        None,
        "--pairwise-sort-memory",
        help="Pairwise baseline only: GNU sort memory for -S (e.g. 8G or 50%).",
    ),
    assume_no_header: bool = typer.Option(
        False, "--assume-no-header", help="Treat first line as data (no header)."
    ),
    order_norm_method: str = typer.Option(
        "pair", "--order-norm", help="OrderNorm: pair (2/(k*(k-1))) or star (1/(k-1))."
    ),
    parquet_batch_size: int = typer.Option(100_000, "--parquet-batch-size", help="Parquet batch size."),
    coarse_method: str = typer.Option(
        "leiden",
        "--coarse-method",
        help="Coarse clustering method for hypergraph pipeline: leiden (default) or spectral (experimental).",
    ),
) -> None:
    out = out.resolve()
    params = {
        "ppl_contacts": str(ppl_contacts),
        "contigs": str(contigs),
        "out": str(out),
        "pairwise_baseline": pairwise_baseline,
        "resolution": resolution,
        "seed": seed,
        "threads": threads,
        "include_tags": include_tags,
        "keep_status": keep_status,
        "assume_sorted_by_readid": assume_sorted_by_readid,
        "pairwise_sort_memory": pairwise_sort_memory,
        "assume_no_header": assume_no_header,
        "order_norm_method": order_norm_method,
        "parquet_batch_size": parquet_batch_size,
        "coarse_method": coarse_method,
    }
    with record_run(out, command="run", params=params, seed=seed) as run_record:
        try:
            if pairwise_baseline:
                if coarse_method.strip().lower() != "leiden":
                    raise GraphClusterError("--coarse-method only applies to the hypergraph pipeline (without --pairwise-baseline).")
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
                stats = build_pairwise_edges(
                    contacts_path=ppl_contacts,
                    out_edges_path=pairwise_edges,
                    tmp_dir=out / "tmp" / "pairwise_baseline",
                    assume_sorted=assume_sorted_by_readid,
                    sort_threads=max(1, int(threads)),
                    memory=pairwise_sort_memory,
                    logger=None,
                )

                graph_meta = {
                    "porebin_version": __version__,
                    "pairwise_baseline": True,
                    "weight_formula": "2/(k*(k-1))  # == 1/C(k,2)",
                    "input_ppl_contacts": str(ppl_contacts),
                    "input_contigs_fasta": str(contigs),
                    "input_sorted_by_readid": stats.input_sorted_by_readid,
                    "input_sorted_by_readid_verified": stats.input_sorted_by_readid_verified,
                    "num_contigs": len(contig_names),
                    "num_edges": stats.unique_edges,
                    "segments_total": stats.segments_total,
                    "segments_kept": stats.segments_kept,
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
                run_record["decisions"] = {"cluster_method": "leiden_pairwise", "resolution": resolution}
            else:
                contacts_parquet = normalize_contacts(
                    ppl_contacts=ppl_contacts,
                    out_dir=out,
                    include_tags=include_tags,
                    keep_status=keep_status,
                    assume_no_header=assume_no_header,
                    threads=threads,
                )
                build_graph(
                    contigs_fasta=contigs,
                    contacts_parquet=contacts_parquet,
                    out_dir=out,
                    order_norm_method=order_norm_method,
                    parquet_batch_size=parquet_batch_size,
                )
                m = coarse_method.strip().lower()
                if m == "leiden":
                    cluster_leiden(
                        graph_dir=out / "graph",
                        out_bins_tsv=out / "bins.tsv",
                        resolution=resolution,
                        seed=seed,
                    )
                    run_record["decisions"] = {"cluster_method": "leiden", "resolution": resolution}
                elif m == "spectral":
                    meta = cluster_spectral_hypergraph(
                        graph_dir=out / "graph",
                        out_bins_tsv=out / "bins.tsv",
                        seed=seed,
                    )
                    run_record["decisions"] = {"cluster_method": "spectral", **meta}
                else:
                    raise GraphClusterError(f"Unknown --coarse-method {coarse_method!r}. Use 'leiden' or 'spectral'.")
        except (
            PairwiseBaselineError,
            NormalizeError,
            GraphBuildError,
            GraphClusterError,
            FileNotFoundError,
        ) as exc:
            _die(str(exc))


def _die(message: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code)


if __name__ == "__main__":  # pragma: no cover
    app()
