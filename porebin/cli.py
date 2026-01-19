from __future__ import annotations

from pathlib import Path

import typer

from porebin.build_graph import GraphBuildError, build_graph
from porebin.cluster import GraphClusterError, cluster_leiden
from porebin.export_bins import ExportBinsError, export_bins_fasta
from porebin.normalize import NormalizeError, normalize_contacts
from porebin.utils import console, record_run, setup_logging

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
        "assume_no_header": assume_no_header,
    }
    with record_run(out, command="normalize", params=params, seed=None):
        try:
            normalize_contacts(
                ppl_contacts=ppl_contacts,
                out_dir=out,
                include_tags=include_tags,
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
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden resolution parameter."),
    seed: int = typer.Option(0, "--seed", help="Random seed for Leiden."),
) -> None:
    out = out.resolve()
    params = {"graph": str(graph), "out": str(out), "resolution": resolution, "seed": seed}
    with record_run(out, command="cluster", params=params, seed=seed):
        try:
            cluster_leiden(
                graph_dir=graph,
                out_bins_tsv=out / "bins.tsv",
                resolution=resolution,
                seed=seed,
            )
        except (GraphClusterError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def export(
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    bins: Path = typer.Option(..., "--bins", help="Bins TSV (out_dir/bins.tsv)."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
) -> None:
    out = out.resolve()
    params = {"contigs": str(contigs), "bins": str(bins), "out": str(out)}
    with record_run(out, command="export", params=params, seed=None):
        try:
            export_bins_fasta(contigs_fasta=contigs, bins_tsv=bins, out_dir=out)
        except (ExportBinsError, FileNotFoundError) as exc:
            _die(str(exc))


@app.command()
def run(
    ppl_contacts: Path = typer.Option(..., "--ppl-contacts", help="PPL .contacts TSV file."),
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden resolution parameter."),
    seed: int = typer.Option(0, "--seed", help="Random seed for Leiden."),
    threads: int = typer.Option(1, "--threads", help="Threads hint (v0.1 mostly single-threaded)."),
    include_tags: list[str] = typer.Option(
        ["mapq", "AS", "n_segments"],
        "--include-tags",
        help="Evidence to include: mapq, AS, n_segments, all. Repeatable.",
    ),
    assume_no_header: bool = typer.Option(
        False, "--assume-no-header", help="Treat first line as data (no header)."
    ),
    order_norm_method: str = typer.Option(
        "pair", "--order-norm", help="OrderNorm: pair (2/(k*(k-1))) or star (1/(k-1))."
    ),
    parquet_batch_size: int = typer.Option(100_000, "--parquet-batch-size", help="Parquet batch size."),
) -> None:
    out = out.resolve()
    params = {
        "ppl_contacts": str(ppl_contacts),
        "contigs": str(contigs),
        "out": str(out),
        "resolution": resolution,
        "seed": seed,
        "threads": threads,
        "include_tags": include_tags,
        "assume_no_header": assume_no_header,
        "order_norm_method": order_norm_method,
        "parquet_batch_size": parquet_batch_size,
    }
    with record_run(out, command="run", params=params, seed=seed):
        try:
            contacts_parquet = normalize_contacts(
                ppl_contacts=ppl_contacts,
                out_dir=out,
                include_tags=include_tags,
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
            cluster_leiden(
                graph_dir=out / "graph",
                out_bins_tsv=out / "bins.tsv",
                resolution=resolution,
                seed=seed,
            )
            export_bins_fasta(contigs_fasta=contigs, bins_tsv=out / "bins.tsv", out_dir=out)
        except (NormalizeError, GraphBuildError, GraphClusterError, ExportBinsError, FileNotFoundError) as exc:
            _die(str(exc))


def _die(message: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code)


if __name__ == "__main__":  # pragma: no cover
    app()

