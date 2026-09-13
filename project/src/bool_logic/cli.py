from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .io_utils import to_plain
from .pipeline import run_audit, run_freeze, run_render, run_report, run_reparse, run_requests, resume_requests
from .preflight import preflight_run
from .run_control import PAUSE_REQUEST_FILE
from .recipe_b_depth_audit import write_recipe_b_depth_audit
from .recipe_candidate_scheduler_probe import write_recipe_candidate_scheduler_probe
from .recipe_seed_audit import write_recipe_seed_audit
from .recipe_scheduler_audit import write_recipe_scheduler_audit
from .recipe_universe_audit import write_recipe_universe_audit

app = typer.Typer(help="Bool Logic benchmark pipeline.")


def _provider_name_from_cli(value: Optional[str], configured: str) -> str:
    if value is None or value == configured:
        return configured
    if value == "mock":
        return value
    raise typer.BadParameter("provider override must match the config provider; only mock is allowed for local dry-runs")


@app.command()
def audit(
    config: Path = typer.Option(Path("configs/experiments/smoke.toml"), "--config", "-c"),
    out: Path = typer.Option(Path("runs/smoke"), "--out", "-o"),
):
    """Build resource audit and feasibility table."""
    result = run_audit(load_config(config), out)
    typer.echo(result)


@app.command()
def freeze(
    config: Path = typer.Option(Path("configs/experiments/smoke.toml"), "--config", "-c"),
    audit_dir: Path = typer.Option(Path("runs/smoke"), "--audit-dir"),
    out: Path = typer.Option(Path("runs/smoke"), "--out", "-o"),
):
    """Generate frozen samples from an existing feasibility table."""
    result = run_freeze(load_config(config), audit_dir, out)
    typer.echo(result)


@app.command()
def render(
    samples: Path = typer.Option(Path("runs/smoke/samples.jsonl"), "--samples"),
    resources: Path = typer.Option(Path("runs/smoke/resource_snapshot.json"), "--resources"),
    out: Path = typer.Option(Path("runs/smoke"), "--out", "-o"),
):
    """Render frozen samples into provider-independent requests."""
    result = run_render(samples, resources, out)
    typer.echo(result)


@app.command()
def run(
    config: Path = typer.Option(Path("configs/experiments/smoke.toml"), "--config", "-c"),
    requests: Path = typer.Option(Path("runs/smoke/requests.jsonl"), "--requests"),
    out: Path = typer.Option(Path("runs/smoke"), "--out", "-o"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    limit: Optional[int] = typer.Option(None, "--limit"),
):
    """Send requests to a provider, preserving existing successful responses."""
    loaded_config = load_config(config)
    provider_name = _provider_name_from_cli(provider, loaded_config.provider.name)
    result = run_requests(loaded_config, requests, out, provider_name, limit=limit)
    typer.echo(result)


@app.command("preflight-run")
def preflight_run_command(
    config: Path = typer.Option(Path("configs/experiments/smoke.toml"), "--config", "-c"),
    requests: Path = typer.Option(Path("runs/smoke/requests.jsonl"), "--requests"),
    out: Path = typer.Option(Path("runs/smoke"), "--out", "-o"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    limit: Optional[int] = typer.Option(None, "--limit"),
):
    """Validate a provider run configuration without sending API requests."""
    loaded_config = load_config(config)
    provider_name = _provider_name_from_cli(provider, loaded_config.provider.name)
    result = preflight_run(loaded_config, requests, out, provider_name, limit=limit)
    typer.echo(to_plain(result))
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command()
def resume(
    config: Path = typer.Option(Path("configs/experiments/smoke.toml"), "--config", "-c"),
    run_dir: Path = typer.Option(Path("runs/smoke"), "--run"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    limit: Optional[int] = typer.Option(None, "--limit"),
):
    """Resume missing requests without changing frozen samples or requests."""
    loaded_config = load_config(config)
    provider_name = _provider_name_from_cli(provider, loaded_config.provider.name)
    result = resume_requests(loaded_config, run_dir, provider_name, limit=limit)
    typer.echo(result)


@app.command()
def pause(
    run_dir: Path = typer.Option(Path("runs/smoke"), "--run"),
):
    """Request a cooperative pause for a running provider run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / PAUSE_REQUEST_FILE
    marker.write_text("pause requested\n", encoding="utf-8")
    typer.echo({"pause_requested": str(marker)})


@app.command()
def reparse(
    config: Path = typer.Option(Path("configs/experiments/smoke.toml"), "--config", "-c"),
    run_dir: Path = typer.Option(Path("runs/smoke"), "--run"),
):
    """Reparse saved responses and score them without sending API requests."""
    result = run_reparse(load_config(config), run_dir)
    typer.echo(result)


@app.command()
def report(run_dir: Path = typer.Option(Path("runs/smoke"), "--run")):
    """Build evaluation_metrics.json and evaluation_report.ipynb from a saved run."""
    result = run_report(run_dir)
    typer.echo(result)


@app.command("recipe-universe-audit")
def recipe_universe_audit(
    config: Path = typer.Option(Path("configs/experiments/test.toml"), "--config", "-c"),
    out_dir: Path = typer.Option(Path("audits/recipe_universe"), "--out-dir"),
    max_a1_per_family: int = typer.Option(0, "--max-a1-per-family"),
):
    """Audit current recipe universe under A1 + OR-branches - B semantics."""
    audit = write_recipe_universe_audit(config, out_dir, max_a1_per_family=max_a1_per_family)
    typer.echo(
        {
            "summary_json": audit["outputs"]["summary"],
            "structural_constraint_core_count": audit["structural_constraint_core_count"],
            "probe_core_rows": audit["probe_core_rows"],
            "test_rendered_requests": audit["sampling_summary"]["test"]["rendered_requests"],
            "main_rendered_requests": audit["sampling_summary"]["main"]["rendered_requests"],
            "full_rendered_requests_estimate": audit["sampling_summary"]["full"]["rendered_requests_estimate"],
        }
    )


@app.command("recipe-seed-audit")
def recipe_seed_audit(
    config: Path = typer.Option(Path("configs/experiments/test.toml"), "--config", "-c"),
    out_dir: Path = typer.Option(Path("audits/recipe_seed"), "--out-dir"),
    max_seed_per_family: int = typer.Option(0, "--max-seed-per-family"),
):
    """Audit seed-object-first current recipe sampling frame."""
    audit = write_recipe_seed_audit(config, out_dir, max_seed_per_family=max_seed_per_family)
    typer.echo(
        {
            "summary_json": audit["outputs"]["summary"],
            "probe_core_rows": audit["probe_core_rows"],
            "test_rendered_requests": audit["sampling_summary"]["test"]["rendered_requests"],
            "main_rendered_requests": audit["sampling_summary"]["main"]["rendered_requests"],
            "full_rendered_requests": audit["sampling_summary"]["full"]["rendered_requests"],
        }
    )


@app.command("recipe-b-depth-audit")
def recipe_b_depth_audit(
    config: Path = typer.Option(Path("configs/experiments/test.toml"), "--config", "-c"),
    probe_frame: Path = typer.Option(Path("audits/recipe_seed/lowest_viable_core_probe_frame.jsonl"), "--probe-frame"),
    out_dir: Path = typer.Option(Path("audits/recipe_b_depth"), "--out-dir"),
):
    """Audit whether deeper B candidates create useful signal or combinatorial growth."""
    audit = write_recipe_b_depth_audit(config, probe_frame, out_dir)
    typer.echo(
        {
            "summary_json": audit["outputs"]["summary"],
            "probe_rows": audit["probe_rows"],
        }
    )


@app.command("recipe-scheduler-audit")
def recipe_scheduler_audit(
    threshold_audit: Path = typer.Option(
        Path("audits/recipe_seed/positive_pool_threshold_audit.json"), "--threshold-audit"
    ),
    out_dir: Path = typer.Option(Path("audits/recipe_scheduler"), "--out-dir"),
    threshold: int = typer.Option(5, "--threshold"),
):
    """Audit candidate realization scheduler options against current capacity data."""
    audit = write_recipe_scheduler_audit(threshold_audit, out_dir, threshold=threshold)
    typer.echo(
        {
            "summary_json": audit["outputs"]["summary"],
            "recommendation": audit["recommendation"],
        }
    )


@app.command("recipe-candidate-scheduler-probe")
def recipe_candidate_scheduler_probe(
    config: Path = typer.Option(Path("configs/experiments/test.toml"), "--config", "-c"),
    probe_frame: Path = typer.Option(Path("audits/recipe_seed/lowest_viable_core_probe_frame.jsonl"), "--probe-frame"),
    out_dir: Path = typer.Option(Path("audits/recipe_candidate_scheduler"), "--out-dir"),
    threshold: int = typer.Option(5, "--threshold"),
    max_rows: int = typer.Option(0, "--max-rows"),
    pool_candidate_limit: int = typer.Option(64, "--pool-candidate-limit"),
):
    """Probe candidate-level scheduler strategies on representative current recipe cores."""
    audit = write_recipe_candidate_scheduler_probe(
        config, probe_frame, out_dir, threshold=threshold, max_rows=max_rows, pool_candidate_limit=pool_candidate_limit
    )
    typer.echo(
        {
            "summary_json": audit["outputs"]["summary"],
            "recommendation": audit["recommendation"],
        }
    )
